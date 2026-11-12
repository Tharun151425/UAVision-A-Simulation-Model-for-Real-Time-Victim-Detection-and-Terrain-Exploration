"""
Phase 2A GRU Reward Function
==============================
Same structure as Phase 1.5 reward + stronger anti-spin term.
Survivor is always visible in 2A so distance shaping fires frequently.
Coverage bonus kept at 0.05 (pre-detection only) — GRU continues
associating fog changes with value.
"""

TURN_LEFT  = 0
TURN_RIGHT = 1
MOVE_UP    = 2
MOVE_DOWN  = 3
MOVE_FORWARD = 4
SCAN       = 5

PROXIMITY_ZONE  = 2.5
HARD_BRAKE_ZONE = 1.5


def compute_reward_2A_gru(env, action, delta_dist: float, newly_revealed: int) -> float:
    r = 0.0

    # ── Anti-spin tracking ────────────────────────────────────────────────────
    if action in (TURN_LEFT, TURN_RIGHT):
        env._consecutive_turns += 1
    else:
        env._consecutive_turns = 0

    # ── One-time detection bonus ──────────────────────────────────────────────
    if env.survivor_in_fov() and not env.survivor_detected:
        r += 20.0
        env.survivor_detected = True

    # ── Distance shaping (visibility-gated) ──────────────────────────────────
    if env.survivor_in_fov():
        r += 1.5 * delta_dist   # slightly higher than Phase 1 — 2A is approach training

    # ── Terminal: survivor reached ────────────────────────────────────────────
    if env.survivor_reached():
        r += 500.0
        env.done = True
        return r

    # ── Anti-spin ─────────────────────────────────────────────────────────────
    if env._consecutive_turns > 3:
        r -= 0.3 * (env._consecutive_turns - 3)

    # ── Proximity warning ─────────────────────────────────────────────────────
    min_dist = env.get_min_obstacle_dist()
    if min_dist < PROXIMITY_ZONE and action == MOVE_FORWARD:
        r -= 2.5 * (1.0 - min_dist / PROXIMITY_ZONE)

    # ── Hard brake ───────────────────────────────────────────────────────────
    if min_dist < HARD_BRAKE_ZONE and action == MOVE_FORWARD:
        r -= 8.0

    # ── Collision terminal ────────────────────────────────────────────────────
    if env.collision_detected():
        r -= 100.0
        env.done = True
        return r

    # ── Altitude floor terminal ───────────────────────────────────────────────
    if env.drone_alt < env.MIN_ALTITUDE:
        r -= 50.0
        env.done = True
        return r

    # ── Coverage bonus (pre-detection, continues from Phase 1) ───────────────
    if not env.survivor_detected and newly_revealed:
        r += 0.05 * newly_revealed

    # ── Time penalty ──────────────────────────────────────────────────────────
    r -= 0.05

    return r
