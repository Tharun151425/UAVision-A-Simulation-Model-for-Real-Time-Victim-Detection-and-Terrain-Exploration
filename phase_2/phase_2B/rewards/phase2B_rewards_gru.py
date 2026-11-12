"""
Phase 2B GRU Reward Function — Partial Occlusion
==================================================
This is where GRU trained from Phase 1 starts earning its keep.
When survivor disappears, the hidden state already contains:
  - Which direction the survivor was last seen
  - How long since it disappeared
  - Nearby obstacle patterns
The re-acquisition bonus encourages the policy to USE that memory.
"""

TURN_LEFT    = 0
TURN_RIGHT   = 1
MOVE_UP      = 2
MOVE_DOWN    = 3
MOVE_FORWARD = 4
SCAN         = 5

REACQ_MIN_LOST_STEPS  = 15
REACQ_MAX_PER_EPISODE = 2
PROXIMITY_ZONE        = 2.5
HARD_BRAKE_ZONE       = 1.5


def compute_reward_2B_gru(env, action, delta_dist: float,
                           newly_revealed: int = 0) -> float:
    r = 0.0

    currently_visible     = env.survivor_in_fov()
    was_visible_last_step = env.survivor_was_visible

    # ── Anti-spin ─────────────────────────────────────────────────────────────
    if action in (TURN_LEFT, TURN_RIGHT):
        env._consecutive_turns += 1
    else:
        env._consecutive_turns = 0

    # ── First detection bonus ─────────────────────────────────────────────────
    if currently_visible and not env.survivor_detected:
        r += 20.0
        env.survivor_detected    = True
        env.survivor_was_visible = True

    # ── Track occlusion duration ──────────────────────────────────────────────
    was_lost_long_enough = (env._lost_for_steps >= REACQ_MIN_LOST_STEPS)
    if env.survivor_detected:
        if currently_visible:
            env._lost_for_steps = 0
        else:
            env._lost_for_steps += 1

    # ── Re-acquisition bonus ──────────────────────────────────────────────────
    if (currently_visible and env.survivor_detected and
            not env.survivor_was_visible and was_lost_long_enough and
            env._reacquisitions < REACQ_MAX_PER_EPISODE):
        r += 8.0
        env._reacquisitions += 1

    env.survivor_was_visible = currently_visible

    # ── Approach reward (visibility-gated) ───────────────────────────────────
    if currently_visible:
        min_d = env.get_min_obstacle_dist()
        if min_d >= 2.5:
            r += 2.0 * delta_dist

    # ── Signal-loss penalty (flew into occlusion carelessly) ─────────────────
    if was_visible_last_step and not currently_visible and action == MOVE_FORWARD:
        if env.get_min_obstacle_dist() < 8.0:
            r -= 1.5

    # ── steps_since_signal_loss tracking ─────────────────────────────────────
    if env.survivor_detected:
        if was_visible_last_step and not currently_visible:
            env.steps_since_signal_loss = 0
        elif not currently_visible:
            env.steps_since_signal_loss += 1
        else:
            env.steps_since_signal_loss = 0

    # ── Survivor-lost penalty ─────────────────────────────────────────────────
    if env.survivor_detected and not currently_visible:
        r -= 0.2

    # ── Terminal: survivor reached ────────────────────────────────────────────
    if env.survivor_reached():
        r += 500.0; env.done = True; return r

    # ── Anti-spin ─────────────────────────────────────────────────────────────
    if env._consecutive_turns > 3:
        r -= 0.3 * (env._consecutive_turns - 3)

    # ── Re-acquisition turn incentive (<30 steps lost) ───────────────────────
    if (env.survivor_detected and not currently_visible and
            env.steps_since_signal_loss < 30):
        if action in (TURN_LEFT, TURN_RIGHT):
            r += 0.8
        elif action == MOVE_FORWARD and env.get_min_obstacle_dist() < 10.0:
            r -= 2.0

    # ── Coverage bonus (pre-detection) ────────────────────────────────────────
    if not env.survivor_detected and newly_revealed:
        r += 3.0   # stronger than Phase 1 (0.05) to drive search

    # ── SCAN penalty (pre-detection) ─────────────────────────────────────────
    if not env.survivor_detected and action == SCAN:
        r -= 0.5

    # ── Proximity warning ─────────────────────────────────────────────────────
    min_dist = env.get_min_obstacle_dist()
    if min_dist < PROXIMITY_ZONE and action == MOVE_FORWARD:
        r -= 2.5 * (1.0 - min_dist / PROXIMITY_ZONE)
    if min_dist < HARD_BRAKE_ZONE and action == MOVE_FORWARD:
        r -= 8.0

    # ── Collision terminal ────────────────────────────────────────────────────
    if env.collision_detected():
        r -= 150.0; env.done = True; return r

    # ── Altitude floor ────────────────────────────────────────────────────────
    if env.drone_alt < env.MIN_ALTITUDE:
        r -= 50.0; env.done = True; return r

    r -= 0.20   # time penalty
    return r
