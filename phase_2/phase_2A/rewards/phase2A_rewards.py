"""
Phase 2A Reward Function — Shape Recognition, No Occlusion
===========================================================
FIXED v2 — resolves the spinning exploit present in v1.

ROOT CAUSE of v1 failure at L3:
  newly_revealed = np.sum(ray_grid > 0)  counted ALL rays hitting
  anything each step, not truly-new cells. With 15 obstacles at L3,
  ~150 rays hit something every step → +15.0/step coverage reward.
  Policy discovered: spinning near walls >> reaching survivor.
  Timeout episodes earned ~15,000-21,000 reward purely from spinning.

FIXES applied:
  1. Coverage bonus REMOVED. Phase 2A objective is "fly to survivor",
     not "explore". Survivor is always visible from spawn — no
     exploration bonus is needed or appropriate.
  2. Distance shaping is now CONDITIONAL on survivor being in FOV.
     When the drone is behind a wall and survivor disappears, shaping
     is zeroed so the drone isn't pushed blindly into the wall.
  3. Time penalty increased from -0.05 → -0.15 (3×) so that
     1100-step timeouts cost -165 instead of -55. Timeouts become
     clearly bad even with distance-shaping noise.
  4. Anti-spin penalty escalates for consecutive turns without
     MOVE_FORWARD. Spinning in place is now expensive.
  5. Survivor-lost penalty (-0.2/step) fires when survivor was
     detected but is no longer in FOV, pushing drone to re-angle.

Key terms:
  +20.0          one-time survivor detection bonus
  +2.0*Δd        distance-shaping when survivor visible
  -0.15          time penalty per step
  -0.2           survivor-lost penalty per step (post-detection)
  -0.3×k         anti-spin: k = consecutive turns beyond 3
  -2.0*proximity proximity warning on MOVE_FORWARD < 5m
  -50.0          collision — terminal
  -50.0          altitude floor — terminal
  +500.0         survivor reached — terminal
"""

MOVE_FORWARD = 4
TURN_LEFT    = 0
TURN_RIGHT   = 1


def compute_reward_2A(env, action, delta_dist, newly_revealed=0):
    r = 0.0

    currently_visible = env.survivor_in_fov()

    # ── Track consecutive turn count ──────────────────────────────────────────
    if action in (TURN_LEFT, TURN_RIGHT):
        env._consecutive_turns += 1
    else:
        env._consecutive_turns = 0

    # ── One-time detection bonus ───────────────────────────────────────────────
    if currently_visible and not env.survivor_detected:
        r += 20.0
        env.survivor_detected = True

    # ── Distance shaping — ONLY when survivor currently visible ───────────────
    # When hidden behind a wall, distance shaping would push drone into wall.
    if currently_visible:
        r += 2.0 * delta_dist

    # ── Survivor-lost penalty — fires after first detection, while hidden ──────
    if env.survivor_detected and not currently_visible:
        r -= 0.2

    # ── Terminal: survivor reached ─────────────────────────────────────────────
    if env.survivor_reached():
        r += 500.0
        env.done = True
        return r

    # ── Anti-spin penalty — escalates with consecutive turns ──────────────────
    # Allows up to 3 turns in a row (needed for obstacle avoidance);
    # penalises spinning beyond that.
    if env._consecutive_turns > 3:
        r -= 0.3 * (env._consecutive_turns - 3)

    # ── Proximity warning — graduated, only on MOVE_FORWARD ───────────────────
    min_dist = env.get_min_obstacle_dist()
    if min_dist < 5.0 and action == MOVE_FORWARD:
        r += -2.0 * (1.0 - min_dist / 5.0)

    # ── Collision terminal ─────────────────────────────────────────────────────
    if env.collision_detected():
        r -= 50.0
        env.done = True
        return r

    # ── Altitude floor terminal ────────────────────────────────────────────────
    if env.drone_alt < env.MIN_ALTITUDE:
        r -= 50.0
        env.done = True
        return r

    # ── Time penalty — 3× stronger than v1 ────────────────────────────────────
    # 1100-step timeout now costs -165 instead of -55.
    r -= 0.15

    return r
