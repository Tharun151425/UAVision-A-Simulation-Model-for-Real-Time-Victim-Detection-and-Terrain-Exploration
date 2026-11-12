"""
Phase 2C Reward Function — Full Occlusion with GRU Memory
===========================================================
v11 — Breaks 2B-transferred SCAN/altitude-jitter habits. Fixes corner trap.

ROOT CAUSE OF EXPLORATION FAILURE (diagnosed from env + logs):
  1. SCAN (action=5) is a complete no-op — no position or yaw change.
     The 2B-transferred policy triggers it constantly (was useful in 2B
     for partially-visible survivors). In 2C it does nothing but waste steps.
  2. MOVE_UP / MOVE_DOWN don't move horizontally → no new fog cells entered.
     The drone oscillates altitude instead of sweeping the map.
  3. Fog-cell bonus (+2) was too weak relative to the safe-do-nothing cost
     (−0.20/step). MOVE_FORWARD into a new cell = +2 per 2 steps = +1/step.
     SCAN = only −0.20/step. The margin was insufficient to break the 2B habit.

v11 FIXES (pre-detection only — post-detection unchanged for 2B parity):
  −0.5   SCAN penalty when survivor not yet detected          [NEW v11]
  −0.3   UP/DOWN penalty in safe altitude range (1.5–4.0m)   [NEW v11]
  +3.0   fog-cell bonus (increased from +2.0)                [NEW v11]
  +0.15  MOVE_FORWARD momentum bonus (pre-detection)         [NEW v11]
  −0.25  boundary penalty when near map edge in open space   [NEW v11]

All 2B-parity terms are IDENTICAL to phase2B_rewards v8+.
Altitude management (−0.15/m above 4m, gated) carried over from v9.

Key terms:
  +20.0          first detection bonus                      [SAME as 2A/2B]
  +8.0           re-acquisition (max 2/ep, >=15 steps lost) [SAME as 2B]
  +2.0*delta     approach (visible, gated min_dist >= 2.5m) [SAME as 2B]
  -0.2/step      survivor-lost base penalty                 [SAME as 2A/2B]
  -0.3*k         anti-spin beyond 3 turns                   [SAME as 2A/2B]
  -0.20/step     time penalty                               [SAME as 2B]
  -2.5*(1-d/2.5) proximity warning <2.5m FORWARD            [SAME as 2B]
  -8.0           hard brake <1.5m FORWARD                   [SAME as 2B]
  -150.0         collision terminal                         [SAME as 2B]
  -50.0          altitude floor terminal                    [SAME as 2A/2B]
  +500.0         survivor reached terminal                  [SAME as 2A/2B]
  -1.5           signal-loss charging (step 0 of loss)      [v7+]
  -1.0*(1-d/8)   approach slow-down (<8m, visible)          [v7+]
  +0.8           re-acq turn incentive (<30 steps lost)     [v7+]
  -2.0           re-acq forward penalty (<30 steps, <10m)   [v7+]
  -0.15*(alt-4m) altitude drift penalty when open (>4m)     [v9+]
  -0.4           MOVE_UP penalty when high+open             [v9+]
  +0.3           MOVE_DOWN bonus when high+open             [v9+]
  +3.0           new fog cell bonus (pre-detect, up from 2) [v11]
  -0.5           SCAN penalty (pre-detect)                  [v11]
  -0.3           UP/DOWN penalty in safe altitude (pre-det) [v11]
  +0.15          MOVE_FORWARD momentum (pre-detect)         [v11]
  -0.25          boundary penalty in open space (pre-det)   [v11]
"""

MOVE_FORWARD = 4
TURN_LEFT    = 0
TURN_RIGHT   = 1
MOVE_UP      = 2
MOVE_DOWN    = 3
SCAN         = 5

REACQ_MIN_LOST_STEPS  = 15
REACQ_MAX_PER_EPISODE = 2

PROXIMITY_ZONE  = 2.5
HARD_BRAKE_ZONE = 1.5
APPROACH_GATE   = 2.5


def compute_reward_2C(env, action, delta_dist,
                      newly_revealed=0, current_cell_was_known=False):
    r = 0.0

    currently_visible     = env.survivor_in_fov()
    was_visible_last_step = env.survivor_was_visible  # capture BEFORE state updates

    # -- Track consecutive turn count -----------------------------------------
    if action in (TURN_LEFT, TURN_RIGHT):
        env._consecutive_turns += 1
    else:
        env._consecutive_turns = 0

    # -- One-time first detection bonus ----------------------------------------
    if currently_visible and not env.survivor_detected:
        r += 20.0
        env.survivor_detected    = True
        env.survivor_was_visible = True

    # -- Track genuine occlusion duration -------------------------------------
    was_lost_long_enough = (env._lost_for_steps >= REACQ_MIN_LOST_STEPS)

    if env.survivor_detected:
        if currently_visible:
            env._lost_for_steps = 0
        else:
            env._lost_for_steps += 1

    # -- Re-acquisition bonus [gated + capped] --------------------------------
    if (currently_visible and
            env.survivor_detected and
            not env.survivor_was_visible and
            was_lost_long_enough and
            env._reacquisitions < REACQ_MAX_PER_EPISODE):
        r += 8.0
        env._reacquisitions += 1

    env.survivor_was_visible = currently_visible

    # -- Signal-loss charging penalty [v7+, mirrors 2B] -----------------------
    if (was_visible_last_step and
            not currently_visible and
            action == MOVE_FORWARD):
        sig_loss_dist = env.get_min_obstacle_dist()
        if sig_loss_dist < 8.0:
            r -= 1.5

    # -- Approach reward -------------------------------------------------------
    if currently_visible:
        min_dist_approach = env.get_min_obstacle_dist()
        if min_dist_approach >= APPROACH_GATE:
            r += 2.0 * delta_dist

    # -- Approach slow-down [v7+, mirrors 2B] ---------------------------------
    if env.survivor_detected and currently_visible and action == MOVE_FORWARD:
        min_dist_ap = env.get_min_obstacle_dist()
        if min_dist_ap < 8.0:
            r -= 1.0 * (1.0 - min_dist_ap / 8.0)

    # -- steps_since_signal_loss tracking [v7+] --------------------------------
    if env.survivor_detected:
        if was_visible_last_step and not currently_visible:
            env.steps_since_signal_loss = 0
        elif not currently_visible:
            env.steps_since_signal_loss += 1
        else:
            env.steps_since_signal_loss = 0

    # -- Survivor-lost penalty -------------------------------------------------
    if env.survivor_detected and not currently_visible:
        r -= 0.2

    # -- Terminal: survivor reached --------------------------------------------
    if env.survivor_reached():
        r += 500.0
        env.done = True
        return r

    # -- Anti-spin ------------------------------------------------------------
    if env._consecutive_turns > 3:
        r -= 0.3 * (env._consecutive_turns - 3)

    # -- Re-acquisition turn incentive [v7+, mirrors 2B] ----------------------
    if (env.survivor_detected and
            not currently_visible and
            env.steps_since_signal_loss < 30):
        if action in (TURN_LEFT, TURN_RIGHT):
            r += 0.8
        elif action == MOVE_FORWARD:
            min_dist_reacq = env.get_min_obstacle_dist()
            if min_dist_reacq < 10.0:
                r -= 2.0

    # ═══════════════════════════════════════════════════════════════════════════
    #  PRE-DETECTION EXPLORATION TERMS [2C only — all gated on not detected]
    # ═══════════════════════════════════════════════════════════════════════════
    if not env.survivor_detected:

        # -- FOG-CELL BONUS (+3 per new cell, up from +2 in v10) ----------------
        # Primary exploration gradient. +3 per new 2x2m cell entered.
        # Increased from +2 because the 2B habit (SCAN+oscillate) needs a
        # stronger counter-signal to be broken.
        if not current_cell_was_known:
            r += 3.0

        # -- SCAN PENALTY [v11: -0.5] ------------------------------------------
        # SCAN (action 5) is a complete no-op in the environment — no position,
        # yaw, or altitude change. The 2B-transferred policy defaults to SCAN
        # when the survivor isn't detected (was useful in 2B for partial
        # occlusion). In 2C it does nothing. Penalise explicitly.
        if action == SCAN:
            r -= 0.5

        # -- ALTITUDE JITTER PENALTY [v11: -0.3] --------------------------------
        # UP/DOWN don't move the drone horizontally, so they don't reveal fog
        # cells. When the drone is already in the safe search altitude (1.5–4m),
        # oscillating altitude is purely wasteful. Gate: don't fire near floor or
        # ceiling where altitude adjustment may genuinely be needed.
        SAFE_ALT_LOW  = env.MIN_ALTITUDE + 0.5   # 1.5m
        SAFE_ALT_HIGH = 4.0                       # 4m
        if action in (MOVE_UP, MOVE_DOWN):
            if SAFE_ALT_LOW <= env.drone_alt <= SAFE_ALT_HIGH:
                r -= 0.3

        # -- FORWARD MOMENTUM BONUS [v11: +0.15] --------------------------------
        # Small per-step bonus for choosing MOVE_FORWARD during exploration.
        # This creates a consistent gradient toward forward movement vs the
        # zero-cost alternatives (SCAN, TURN_LEFT/RIGHT in place, UP/DOWN).
        # Even in known cells (no fog bonus), forward movement is better than
        # staying put.
        if action == MOVE_FORWARD:
            r += 0.15

        # -- BOUNDARY PENALTY [v11: -0.25] --------------------------------------
        # The drone gets trapped at map edges: MOVE_FORWARD clips position, so
        # the drone appears to "move" but stays at the boundary cell.
        # Penalise being near the boundary in genuinely open space (obstacle > 5m)
        # so walls near obstacles are not penalised (obstacle avoidance is fine).
        half = env.MAP_SIZE / 2.0
        near_boundary = (abs(env.drone_x) > half - 3.0 or
                         abs(env.drone_y) > half - 3.0)
        if near_boundary and env.get_min_obstacle_dist() > 5.0:
            r -= 0.25

    # ═══════════════════════════════════════════════════════════════════════════

    # -- Proximity warning <2.5m FORWARD (omnidirectional) --------------------
    min_dist = env.get_min_obstacle_dist()
    if min_dist < PROXIMITY_ZONE and action == MOVE_FORWARD:
        r -= 2.5 * (1.0 - min_dist / PROXIMITY_ZONE)

    # -- Hard brake <1.5m FORWARD ---------------------------------------------
    if min_dist < HARD_BRAKE_ZONE and action == MOVE_FORWARD:
        r -= 8.0

    # -- Collision terminal ---------------------------------------------------
    if env.collision_detected():
        r -= 150.0
        env.done = True
        return r

    # -- Altitude floor terminal ----------------------------------------------
    if env.drone_alt < env.MIN_ALTITUDE:
        r -= 50.0
        env.done = True
        return r

    # -- Time penalty [-0.20/step, 2B parity] ---------------------------------
    r -= 0.20

    # -- Altitude management [v9+: -0.15/m above 4m, gated on open space] ----
    # Prevents altitude escape (climbing above obstacles to avoid crashes),
    # which breaks survivor detection since survivor is at ground level.
    # Gate: obs_clear (>5m) means this only fires in genuinely open space,
    # preserving the obstacle-avoidance skill (keeps crash rate low).
    ALT_SEARCH_MAX = 4.0
    ALT_SAFE_OBS   = 5.0
    obs_clear = (env.get_min_obstacle_dist() > ALT_SAFE_OBS)

    if env.drone_alt > ALT_SEARCH_MAX and obs_clear:
        # Graduated: 0 at 4m, -0.15 at 5m, -0.90 at 10m per step
        r -= 0.15 * (env.drone_alt - ALT_SEARCH_MAX)

        # Discourage climbing when already high and clear
        if action == MOVE_UP:
            r -= 0.4

        # Encourage descending back to search altitude
        if action == MOVE_DOWN:
            r += 0.3

    return r
