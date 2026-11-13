"""
Phase 2B Reward Function — Partial Occlusion, No GRU
=====================================================
v8 — Proximity zone correction (root cause of spinning/timeout).

v7 POST-MORTEM:
  The 5m proximity zone was too large for Level 3 obstacle density.
  Level 3 has 12-18 obstacles in a 40m × 40m map. Average obstacle
  spacing is ~10m, but corridors between obstacles are often 3-6m wide.
  With a 5m zone firing on EVERY forward step near any obstacle, the
  effective cost per step was:
    600 forward steps × avg -1.2/step (obs at 3m avg) = -720 proximity
    668 steps × -0.15 = -100 time
    Approach reward (5m gate, almost never fires at L3) = 0
  → Even SUCCESSFUL episodes had reward = +520 - 820 = -300 (NEGATIVE)

  This broke the reward gradient for MOVE_FORWARD:
    - Moving forward: pays -1.2 to -2.4/step in proximity
    - Turning: pays only -0.15/step
  → Drone correctly learns: NEVER move forward. Spin forever. Timeout.

FIXES in v8:
  1. Proximity zone: 5m → 2.5m.
     At L3 density, ~12% of map area is within 2.5m of an obstacle
     (vs ~40% within 5m). Forward steps now rarely trigger proximity.
  2. Approach gate: 5m → 2.5m (matches proximity zone).
     Approach reward fires whenever the drone is safely clear (> 2.5m).
     At L3 with typical 4-6m corridors, approach fires most of the time.
  3. Approach reward: +1.0 → +2.0*Δd. Restore full magnitude.
  4. Hard brake: <3m → <1.5m. Fires only when genuinely about to crash.
  5. Time penalty: -0.15 → -0.20/step. Stronger timeout deterrent.

EXPECTED ECONOMICS (v8):
  Normal forward step in 4m wide corridor (obs at 4m):
    approach = +2.0 × 0.8 = +1.6, proximity = 0 → NET +1.6 ← POSITIVE
  Forward step close to wall (obs at 2m):
    approach gated (2m < 2.5m gate) = 0, proximity = -2.5*(1-2/2.5) = -0.5
    → NET -0.5 ← small penalty, still worth trying to navigate around
  Forward step too close (obs at 1.5m → hard brake):
    approach = 0, proximity = -1.0, hard brake = -8.0 → NET -9.0 ← STOP
  Spinning (no forward steps):
    -0.20/step time only → 1200 steps = -240 (vs +520 for finding survivor)

All v5 exploit fixes (angle-hunt removal, re-acquisition gating/cap) preserved.

Key terms:
  +20.0          first detection bonus                     [SAME as 2A]
  +8.0           re-acquisition bonus (max 2/ep,           [SAME as v5]
                  requires 15+ steps genuine occlusion)
  +2.0*Δd        approach (visible, gated min_dist >= 2.5m) [RESTORED v8]
  -0.2/step      survivor-lost base penalty                [SAME as 2A]
  [REMOVED]      angle-hunt +0.8 when turning while lost   [WAS v3-v4]
  -0.3*k         anti-spin beyond 3 turns                  [SAME as 2A]
  -0.20/step     time penalty                              [INCREASED v8]
  -2.5*(1-d/2.5) proximity warning <2.5m FORWARD (omni.)  [TIGHTENED v8]
  -8.0           hard brake <1.5m FORWARD                  [TIGHTENED v8]
  -150.0         collision terminal                        [SAME as v7]
  -50.0          altitude floor terminal                   [SAME as 2A]
  +500.0         survivor reached terminal                 [SAME as 2A]
"""

MOVE_FORWARD = 4
TURN_LEFT    = 0
TURN_RIGHT   = 1

# Re-acquisition guard constants
REACQ_MIN_LOST_STEPS   = 15
REACQ_MAX_PER_EPISODE  = 2

# Proximity zone — 2.5m matches typical narrow corridor width at L3
PROXIMITY_ZONE   = 2.5   # warning fires when min_dist < this
HARD_BRAKE_ZONE  = 1.5   # hard brake fires when min_dist < this
APPROACH_GATE    = 2.5   # approach reward blocked when min_dist < this


def compute_reward_2B(env, action, delta_dist, newly_revealed=0):
    r = 0.0

    currently_visible     = env.survivor_in_fov()
    was_visible_last_step = env.survivor_was_visible  # capture BEFORE any state updates

    # ── Track consecutive turn count [SAME as 2A] ────────────────────────────
    if action in (TURN_LEFT, TURN_RIGHT):
        env._consecutive_turns += 1
    else:
        env._consecutive_turns = 0

    # ── One-time first detection bonus [SAME as 2A] ───────────────────────────
    if currently_visible and not env.survivor_detected:
        r += 20.0
        env.survivor_detected    = True
        env.survivor_was_visible = True

    # ── Track genuine occlusion duration ──────────────────────────────────────
    was_lost_long_enough = (env._lost_for_steps >= REACQ_MIN_LOST_STEPS)

    if env.survivor_detected:
        if currently_visible:
            env._lost_for_steps = 0
        else:
            env._lost_for_steps += 1

    # ── RE-ACQUISITION BONUS [SAME as v5] ─────────────────────────────────────
    if (currently_visible and
            env.survivor_detected and
            not env.survivor_was_visible and
            was_lost_long_enough and
            env._reacquisitions < REACQ_MAX_PER_EPISODE):
        r += 8.0
        env._reacquisitions += 1

    env.survivor_was_visible = currently_visible

    # ── Signal-loss charging penalty [NEW v8] ───────────────────────────────────
    # Mid-episode crash pattern: drone detects survivor, it goes behind an obstacle,
    # then the drone charges forward (toward where it last saw it) and hits the obstacle.
    # Penalise MOVE_FORWARD taken in the exact step the signal was lost, near an obstacle.
    if (was_visible_last_step and
            not currently_visible and
            action == MOVE_FORWARD):
        sig_loss_dist = env.get_min_obstacle_dist()
        if sig_loss_dist < 8.0:
            r -= 1.5

    # ── Approach reward — RESTORED in v8 ──────────────────────────────────────
    # Gate at 2.5m: blocked only when the drone is dangerously close.
    # In typical 4-6m wide corridors at L3, this fires freely.
    # At L3 density with avg corridor width ~4m, approach fires ~90% of steps.
    # This restores the positive forward movement gradient missing in v7.
    if currently_visible:
        min_dist_approach = env.get_min_obstacle_dist()
        if min_dist_approach >= APPROACH_GATE:
            r += 2.0 * delta_dist

    # ── Approach slow-down [NEW] — 200-250 step crash fix ─────────────────────
    # When survivor IS visible but an obstacle is within 8m, the drone is in
    # approach mode with something in the way. Penalise MOVE_FORWARD to break
    # the Phase 1.5 "charge at survivor" habit.
    # Gradient: 0.0 at 8m → -1.0 at contact. Open approaches unaffected.
    if env.survivor_detected and currently_visible and action == MOVE_FORWARD:
        min_dist_ap = env.get_min_obstacle_dist()
        if min_dist_ap < 8.0:
            r -= 1.0 * (1.0 - min_dist_ap / 8.0)

    # ── steps_since_signal_loss tracking [NEW] ────────────────────────────────
    if env.survivor_detected:
        if was_visible_last_step and not currently_visible:
            env.steps_since_signal_loss = 0    # just lost signal this step
        elif not currently_visible:
            env.steps_since_signal_loss += 1   # still lost
        else:
            env.steps_since_signal_loss = 0    # visible — reset

    # ── Survivor-lost penalty [angle-hunt REMOVED in v5] ─────────────────────
    if env.survivor_detected and not currently_visible:
        r -= 0.2


    # ── Terminal: survivor reached [SAME as 2A] ───────────────────────────────
    if env.survivor_reached():
        r += 500.0
        env.done = True
        return r

    # ── Anti-spin [SAME as 2A] ────────────────────────────────────────────────
    if env._consecutive_turns > 3:
        r -= 0.3 * (env._consecutive_turns - 3)

    # ── Re-acquisition turn incentive [NEW] ─────────────────────────────────
    # 400-600 step crash fix. In the 30-step window after losing the survivor,
    # reward turning (scan for new angle) and punish forward into obstacles
    # (the Phase 1.5 "charge where I last saw it" habit).
    # After 30 steps it is OK to move forward to find a better vantage point.
    if (env.survivor_detected and
            not currently_visible and
            env.steps_since_signal_loss < 30):
        if action in (TURN_LEFT, TURN_RIGHT):
            r += 0.8
        elif action == MOVE_FORWARD:
            min_dist_reacq = env.get_min_obstacle_dist()
            if min_dist_reacq < 10.0:
                r -= 2.0

    # ── Proximity warning <2.5m FORWARD — omnidirectional [TIGHTENED v8] ──────
    # Zone shrunk from 5m to 2.5m. Now fires rarely (~12% of forward steps
    # at L3 density vs ~40% with the 5m zone). Forward movement is no longer
    # globally penalized, restoring the movement gradient.
    min_dist = env.get_min_obstacle_dist()
    if min_dist < PROXIMITY_ZONE and action == MOVE_FORWARD:
        r -= 2.5 * (1.0 - min_dist / PROXIMITY_ZONE)

    # ── Hard brake <1.5m FORWARD [TIGHTENED v8] ──────────────────────────────
    if min_dist < HARD_BRAKE_ZONE and action == MOVE_FORWARD:
        r -= 8.0

    # ── Collision terminal [SAME as v7: -150] ────────────────────────────────
    if env.collision_detected():
        r -= 150.0
        env.done = True
        return r

    # ── Altitude floor terminal [SAME as 2A] ─────────────────────────────────
    if env.drone_alt < env.MIN_ALTITUDE:
        r -= 50.0
        env.done = True
        return r

    # ── Time penalty [INCREASED v8: -0.20/step] ───────────────────────────────
    # Increased from -0.15 to make timeout more costly vs finding survivor.
    # Spinning for 1200 steps = -240 vs finding survivor = +500. Clear signal.
    r -= 0.20

    return r
