"""
Phase 2C GRU Reward Function — Full Occlusion
===============================================
Identical structure to the proven Phase 2C v11 reward that worked
in old codebase, but now the GRU arrives already trained from Phase 1.
The exploration terms (fog bonus, SCAN penalty, forward momentum) remain
because the map is large and search is still required.
"""

TURN_LEFT=0; TURN_RIGHT=1; MOVE_UP=2; MOVE_DOWN=3; MOVE_FORWARD=4; SCAN=5

REACQ_MIN_LOST_STEPS  = 15
REACQ_MAX_PER_EPISODE = 2
PROXIMITY_ZONE        = 2.5
HARD_BRAKE_ZONE       = 1.5
APPROACH_GATE         = 2.5


def compute_reward_2C_gru(env, action, delta_dist: float,
                          newly_revealed: int = 0) -> float:
    r = 0.0

    currently_visible     = env.survivor_in_fov()
    was_visible_last_step = env.survivor_was_visible

    if action in (TURN_LEFT, TURN_RIGHT):
        env._consecutive_turns += 1
    else:
        env._consecutive_turns = 0

    # ── First detection ───────────────────────────────────────────────────────
    if currently_visible and not env.survivor_detected:
        r += 20.0
        env.survivor_detected    = True
        env.survivor_was_visible = True

    # ── Occlusion duration ────────────────────────────────────────────────────
    was_lost_long_enough = (env._lost_for_steps >= REACQ_MIN_LOST_STEPS)
    if env.survivor_detected:
        if currently_visible: env._lost_for_steps = 0
        else:                  env._lost_for_steps += 1

    # ── Re-acquisition bonus ──────────────────────────────────────────────────
    if (currently_visible and env.survivor_detected and
            not env.survivor_was_visible and was_lost_long_enough and
            env._reacquisitions < REACQ_MAX_PER_EPISODE):
        r += 8.0
        env._reacquisitions += 1

    env.survivor_was_visible = currently_visible

    # ── Signal-loss penalty ───────────────────────────────────────────────────
    if (was_visible_last_step and not currently_visible and
            action == MOVE_FORWARD and env.get_min_obstacle_dist() < 8.0):
        r -= 1.5

    # ── Approach reward ───────────────────────────────────────────────────────
    if currently_visible and env.get_min_obstacle_dist() >= APPROACH_GATE:
        r += 2.0 * delta_dist

    # ── steps_since_signal_loss ───────────────────────────────────────────────
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

    # ── Re-acquisition turn incentive ─────────────────────────────────────────
    if (env.survivor_detected and not currently_visible and
            env.steps_since_signal_loss < 30):
        if action in (TURN_LEFT, TURN_RIGHT):
            r += 0.8
        elif action == MOVE_FORWARD and env.get_min_obstacle_dist() < 10.0:
            r -= 2.0

    # ── PRE-DETECTION EXPLORATION ─────────────────────────────────────────────
    if not env.survivor_detected:
        if newly_revealed:
            r += 3.0                     # fog-cell bonus
        if action == SCAN:
            r -= 0.5                     # SCAN is no-op, penalise
        SAFE_ALT_LOW = env.MIN_ALTITUDE + 0.5
        SAFE_ALT_HIGH = 4.0
        if action in (MOVE_UP, MOVE_DOWN):
            if SAFE_ALT_LOW <= env.drone_alt <= SAFE_ALT_HIGH:
                r -= 0.3
        if action == MOVE_FORWARD:
            r += 0.15                    # forward momentum
        half = env.MAP_SIZE / 2.0
        near_boundary = (abs(env.drone_x) > half - 3.0 or
                         abs(env.drone_y) > half - 3.0)
        if near_boundary and env.get_min_obstacle_dist() > 5.0:
            r -= 0.25

    # ── Proximity & hard brake ────────────────────────────────────────────────
    min_dist = env.get_min_obstacle_dist()
    if min_dist < PROXIMITY_ZONE and action == MOVE_FORWARD:
        r -= 2.5 * (1.0 - min_dist / PROXIMITY_ZONE)
    if min_dist < HARD_BRAKE_ZONE and action == MOVE_FORWARD:
        r -= 8.0

    # ── Terminals ─────────────────────────────────────────────────────────────
    if env.collision_detected():
        r -= 150.0; env.done = True; return r
    if env.drone_alt < env.MIN_ALTITUDE:
        r -= 50.0; env.done = True; return r

    r -= 0.20   # time penalty

    # ── Altitude drift penalty (open space) ───────────────────────────────────
    ALT_MAX = 4.0
    if env.drone_alt > ALT_MAX and env.get_min_obstacle_dist() > 5.0:
        r -= 0.15 * (env.drone_alt - ALT_MAX)
        if action == MOVE_UP:   r -= 0.4
        if action == MOVE_DOWN: r += 0.3

    return r
