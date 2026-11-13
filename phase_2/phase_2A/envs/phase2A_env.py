"""
Phase 2A Environment — Shape Recognition, No Occlusion
=======================================================
Objective: Learn that ALL solid objects are code 2 regardless of shape.
Survivor is ALWAYS ray-visible from spawn.

Obstacle shapes introduced via internal 3-level curriculum:
  L1: cylinders + boxes        (5-10 obstacles)
  L2: cylinders + boxes + trees (8-12 obstacles)
  L3: all shapes (+ rocks, walls) (10-15 obstacles)

Tree shape: cone on top of cylinder trunk — same obstacle code (2) as everything else.

Observation space: 262 values — identical to Phase 1 / Phase 1.5.
  → Direct PPO.load() from Phase 1.5 checkpoint, no architecture surgery.
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import math
import sys
import os
from collections import deque

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _PROJ_ROOT)

from phase_2.phase_2A.sensors.raycast import RaycastSensor
from phase_2.phase_2A.rewards.phase2A_rewards import compute_reward_2A

# ── Curriculum levels ──────────────────────────────────────────────────────────
CURRICULUM_LEVELS = {
    1: {
        "min_obstacles":      5,
        "max_obstacles":      10,
        "min_height":         2.0,
        "max_height":         6.0,
        "shapes":             ["cylinder", "box"],
        "drone_safe_dist":    6.0,
        "survivor_safe_dist": 3.0,
        "promote_success":    0.80,
        "promote_crash":      0.08,
        "promote_timeout":    0.12,
        "promote_window":     100,
        "description":        "2A-L1 — Cylinders + Boxes",
    },
    2: {
        "min_obstacles":      8,
        "max_obstacles":      12,
        "min_height":         2.0,
        "max_height":         7.0,
        "shapes":             ["cylinder", "box", "tree"],
        "drone_safe_dist":    5.0,
        "survivor_safe_dist": 3.0,
        "promote_success":    0.75,
        "promote_crash":      0.10,
        "promote_timeout":    0.15,
        "promote_window":     100,
        "description":        "2A-L2 — Cylinders + Boxes + Trees",
    },
    3: {
        "min_obstacles":      10,
        "max_obstacles":      15,
        "min_height":         2.0,
        "max_height":         7.0,
        "shapes":             ["cylinder", "box", "tree", "rock", "wall"],
        "drone_safe_dist":    5.0,
        "survivor_safe_dist": 3.0,
        "promote_success":    None,   # final level
        "promote_crash":      None,
        "promote_timeout":    None,
        "promote_window":     100,
        "description":        "2A-L3 — All Shapes (pass-criteria level)",
    },
}

# Pass criteria evaluated at L3 over 200 episodes
PASS_CRITERIA = {
    "success_rate": 70.0,
    "crash_rate":   10.0,
    "timeout_rate": 20.0,
}


class Phase2AEnv(gym.Env):
    """Phase 2A SAR drone environment — shape recognition, no occlusion."""

    _LEVEL_MAX_STEPS = {1: 1100, 2: 1100, 3: 1100}
    MAP_SIZE            = 40
    RAY_RANGE           = 20.0
    FORWARD_STEP        = 1.0
    YAW_STEP            = 15.0
    ALTITUDE_STEP       = 0.5
    MIN_ALTITUDE        = 1.0
    MAX_ALTITUDE        = 10.0
    SURVIVOR_REACH_DIST = 2.0
    SURVIVOR_MIN_DIST   = 8
    SURVIVOR_MAX_DIST   = 20

    TURN_LEFT    = 0
    TURN_RIGHT   = 1
    MOVE_UP      = 2
    MOVE_DOWN    = 3
    MOVE_FORWARD = 4
    SCAN         = 5

    PROMOTE_WINDOW = 100

    @property
    def MAX_STEPS(self):
        return self._LEVEL_MAX_STEPS[self.curriculum_level]

    def __init__(self, render=False, curriculum_level=1):
        super().__init__()
        self.render_mode = render
        self.curriculum_level = max(1, min(3, curriculum_level))

        # Rolling windows for promotion tracking
        self._outcome_success = deque(maxlen=self.PROMOTE_WINDOW)
        self._outcome_crash   = deque(maxlen=self.PROMOTE_WINDOW)
        self._outcome_timeout = deque(maxlen=self.PROMOTE_WINDOW)
        self._episode_count   = 0

        # Obs: 256 ray + 5 pose + 1 time = 262 (identical to Phase 1/1.5)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=10.0, shape=(262,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(6)

        if render:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=35, cameraPitch=-50,
                cameraYaw=45, cameraTargetPosition=[0, 0, 0]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        self.sensor = RaycastSensor(
            h_rays=32, v_rays=8, h_fov=120, v_fov=60, max_range=self.RAY_RANGE
        )

        self.drone_x = self.drone_y = self.drone_alt = 0.0
        self.drone_yaw    = 0.0
        self.survivor_x   = self.survivor_y = 0.0
        self.ground_id    = -1
        self.drone_id     = -1
        self.survivor_id  = -1
        self.obstacle_ids = []
        self.survivor_detected    = False
        self.prev_dist       = 0.0
        self.done            = False
        self.step_count      = 0
        self._last_ray_grid  = np.zeros((8, 32), dtype=np.int32)
        self._consecutive_turns = 0

        self.reset()

    # ══════════════════════════════════════════════════════════════════════════
    #  Gym API
    # ══════════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, 0)

        self.step_count         = 0
        self.done               = False
        self.survivor_detected  = False
        self._consecutive_turns = 0

        self.ground_id = p.loadURDF("plane.urdf")

        if seed is not None:
            np.random.seed(seed)

        half = (self.MAP_SIZE / 2) - 1.0
        lvl  = CURRICULUM_LEVELS[self.curriculum_level]

        # Place drone + survivor, ensure survivor ray-visible from spawn
        for _attempt in range(100):
            self.drone_x = np.random.uniform(-half, half)
            self.drone_y = np.random.uniform(-half, half)
            self.drone_yaw = np.random.uniform(0, 360)
            self.drone_alt = np.random.uniform(2.5, 5.0)

            dist = np.random.uniform(self.SURVIVOR_MIN_DIST, self.SURVIVOR_MAX_DIST)
            angle = np.random.uniform(0, 2 * math.pi)
            sx = self.drone_x + dist * math.cos(angle)
            sy = self.drone_y + dist * math.sin(angle)

            if not (-half <= sx <= half and -half <= sy <= half):
                continue

            self.survivor_x = sx
            self.survivor_y = sy
            self.prev_dist  = dist
            break

        self.drone_id    = self._spawn_drone()
        self.survivor_id = self._spawn_survivor()

        # Red nose marker
        p.addUserDebugLine(
            [0.3, 0.0, 0.1], [1.5, 0.0, 0.1],
            lineColorRGB=[1, 0, 0], lineWidth=4,
            parentObjectUniqueId=self.drone_id, parentLinkIndex=-1
        )

        n = np.random.randint(lvl["min_obstacles"], lvl["max_obstacles"] + 1)
        self.obstacle_ids = self._spawn_obstacles(n, lvl)

        # Verify survivor is ray-visible; if not, clear obstacles near LOS
        self._last_ray_grid, _ = self.sensor.fire(
            self.drone_x, self.drone_y, self.drone_alt, self.drone_yaw,
            ground_id=self.ground_id,
            obstacle_ids=self.obstacle_ids,
            survivor_id=self.survivor_id,
            physics_client=self.physics_client
        )

        # If survivor not visible, do a 360 yaw sweep to check
        if not self.survivor_in_fov():
            found_yaw = self._find_clear_yaw()
            if found_yaw is not None:
                self.drone_yaw = found_yaw
                quat = p.getQuaternionFromEuler([0, 0, math.radians(self.drone_yaw)])
                p.resetBasePositionAndOrientation(
                    self.drone_id,
                    [self.drone_x, self.drone_y, self.drone_alt], quat
                )
                self._last_ray_grid, _ = self.sensor.fire(
                    self.drone_x, self.drone_y, self.drone_alt, self.drone_yaw,
                    ground_id=self.ground_id,
                    obstacle_ids=self.obstacle_ids,
                    survivor_id=self.survivor_id,
                    physics_client=self.physics_client
                )

        return self._get_obs(self._last_ray_grid), {}

    def step(self, action):
        self.step_count += 1
        self._apply_action(action)
        p.stepSimulation()

        ray_grid, _ = self.sensor.fire(
            self.drone_x, self.drone_y, self.drone_alt, self.drone_yaw,
            ground_id=self.ground_id,
            obstacle_ids=self.obstacle_ids,
            survivor_id=self.survivor_id,
            physics_client=self.physics_client
        )
        self._last_ray_grid = ray_grid

        current_dist = math.hypot(
            self.drone_x - self.survivor_x,
            self.drone_y - self.survivor_y
        )
        delta_dist     = self.prev_dist - current_dist
        self.prev_dist = current_dist

        # newly_revealed passed for API compat — reward v2 does not use it
        newly_revealed = int(np.sum(ray_grid > 0))

        obs    = self._get_obs(ray_grid)
        reward = compute_reward_2A(self, action, delta_dist, newly_revealed)

        reason  = "Ongoing"
        crashed = False
        found   = False

        if self.done:
            if self.survivor_reached():
                reason = "Survivor Found"
                found  = True
            elif self.collision_detected():
                reason  = "Crashed (Obstacle)"
                crashed = True
            elif self.drone_alt < self.MIN_ALTITUDE:
                reason  = "Crashed (Altitude Floor)"
                crashed = True

        if not self.done and self.step_count >= self.MAX_STEPS:
            reason    = "Timeout"
            self.done = True

        if self.done:
            self._episode_count += 1
            self._outcome_success.append(1 if found else 0)
            self._outcome_crash.append(1 if crashed else 0)
            self._outcome_timeout.append(1 if "Timeout" in reason else 0)
            self._maybe_promote()

        info = {
            "step":               self.step_count,
            "survivor_found":     found,
            "crashed":            crashed,
            "curriculum_level":   self.curriculum_level,
            "termination_reason": reason,
        }
        return obs, reward, self.done, False, info

    # ══════════════════════════════════════════════════════════════════════════
    #  Curriculum
    # ══════════════════════════════════════════════════════════════════════════

    def _maybe_promote(self):
        if self.curriculum_level >= 3:
            return
        lvl = CURRICULUM_LEVELS[self.curriculum_level]
        w   = lvl["promote_window"]
        if len(self._outcome_success) < w:
            return

        sr = sum(self._outcome_success) / w
        cr = sum(self._outcome_crash)   / w
        tr = sum(self._outcome_timeout) / w

        if (sr >= lvl["promote_success"] and
                cr <= lvl["promote_crash"] and
                tr <= lvl["promote_timeout"]):
            old = self.curriculum_level
            self.curriculum_level += 1
            self._outcome_success.clear()
            self._outcome_crash.clear()
            self._outcome_timeout.clear()
            print(
                f"\n[Phase 2A] PROMOTION: Level {old} → {self.curriculum_level}"
                f"  |  success={sr*100:.1f}%  crash={cr*100:.1f}%  "
                f"timeout={tr*100:.1f}%  |  ep#{self._episode_count}\n"
            )

    def get_curriculum_level(self):
        return self.curriculum_level

    def set_curriculum_level(self, level):
        self.curriculum_level = max(1, min(3, level))
        self._outcome_success.clear()
        self._outcome_crash.clear()
        self._outcome_timeout.clear()

    # ══════════════════════════════════════════════════════════════════════════
    #  Action
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_action(self, action):
        if action == self.TURN_LEFT:
            self.drone_yaw = (self.drone_yaw - self.YAW_STEP) % 360
        elif action == self.TURN_RIGHT:
            self.drone_yaw = (self.drone_yaw + self.YAW_STEP) % 360
        elif action == self.MOVE_UP:
            self.drone_alt = min(self.drone_alt + self.ALTITUDE_STEP, self.MAX_ALTITUDE)
        elif action == self.MOVE_DOWN:
            self.drone_alt = max(self.drone_alt - self.ALTITUDE_STEP, self.MIN_ALTITUDE)
        elif action == self.MOVE_FORWARD:
            yaw_rad = math.radians(self.drone_yaw)
            self.drone_x += self.FORWARD_STEP * math.cos(yaw_rad)
            self.drone_y += self.FORWARD_STEP * math.sin(yaw_rad)

        half = self.MAP_SIZE / 2
        self.drone_x = float(np.clip(self.drone_x, -half, half))
        self.drone_y = float(np.clip(self.drone_y, -half, half))

        pos  = [self.drone_x, self.drone_y, self.drone_alt]
        quat = p.getQuaternionFromEuler([0, 0, math.radians(self.drone_yaw)])
        p.resetBasePositionAndOrientation(self.drone_id, pos, quat)
        p.resetBaseVelocity(self.drone_id,
                            linearVelocity=[0, 0, 0],
                            angularVelocity=[0, 0, 0])

    # ══════════════════════════════════════════════════════════════════════════
    #  Observation (262)
    # ══════════════════════════════════════════════════════════════════════════

    def _get_obs(self, ray_grid=None):
        if ray_grid is None:
            ray_grid = np.zeros((8, 32), dtype=np.float32)
        half = self.MAP_SIZE / 2
        return np.concatenate([
            ray_grid.flatten().astype(np.float32),
            np.array([
                self.drone_x   / half,
                self.drone_y   / half,
                self.drone_alt / self.MAX_ALTITUDE,
                math.cos(math.radians(self.drone_yaw)),
                math.sin(math.radians(self.drone_yaw)),
            ], dtype=np.float32),
            np.array([self.step_count / self.MAX_STEPS], dtype=np.float32),
        ])

    # ══════════════════════════════════════════════════════════════════════════
    #  Spawn helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _spawn_drone(self):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.1])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.1],
                                  rgbaColor=[0.2, 0.4, 0.9, 1.0])
        quat = p.getQuaternionFromEuler([0, 0, math.radians(self.drone_yaw)])
        return p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[self.drone_x, self.drone_y, self.drone_alt],
            baseOrientation=quat
        )

    def _spawn_survivor(self):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=1.0)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5,
                                  rgbaColor=[1.0, 0.1, 0.1, 1.0])
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[self.survivor_x, self.survivor_y, 0.5]
        )

    def _spawn_obstacles(self, count, lvl):
        obs_ids          = []
        placed_positions = []
        half = (self.MAP_SIZE / 2) - 0.5

        for _ in range(count):
            for _ in range(300):
                x = np.random.uniform(-half, half)
                y = np.random.uniform(-half, half)

                if math.hypot(self.drone_x - x, self.drone_y - y) < lvl["drone_safe_dist"]:
                    continue
                if math.hypot(self.survivor_x - x, self.survivor_y - y) < lvl["survivor_safe_dist"]:
                    continue
                if any(math.hypot(px - x, py - y) < 2.0
                       for px, py in placed_positions):
                    continue

                h     = np.random.uniform(lvl["min_height"], lvl["max_height"])
                shape = np.random.choice(lvl["shapes"])
                ids   = self._create_obstacle(shape, x, y, h)
                obs_ids.extend(ids)
                placed_positions.append((x, y))
                break

        return obs_ids

    def _create_obstacle(self, shape, x, y, h):
        """Returns a list of body IDs (tree has 2 parts, others have 1)."""
        if shape == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=h)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=h,
                                      rgbaColor=[0.55, 0.35, 0.15, 1.0])
            uid = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[x, y, h / 2])
            return [uid]

        elif shape == "box":
            hw = np.random.uniform(0.5, 2.5)
            hd = np.random.uniform(0.5, 2.5)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hw, hd, h / 2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hw, hd, h / 2],
                                      rgbaColor=[0.45, 0.45, 0.45, 1.0])
            uid = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[x, y, h / 2])
            return [uid]

        elif shape == "tree":
            # Trunk: thin cylinder (same code=2 as everything else)
            trunk_r = 0.2
            trunk_h = h * 0.6
            col_t = p.createCollisionShape(p.GEOM_CYLINDER, radius=trunk_r, height=trunk_h)
            vis_t = p.createVisualShape(p.GEOM_CYLINDER, radius=trunk_r, length=trunk_h,
                                        rgbaColor=[0.4, 0.25, 0.1, 1.0])
            trunk_id = p.createMultiBody(baseMass=0,
                                         baseCollisionShapeIndex=col_t,
                                         baseVisualShapeIndex=vis_t,
                                         basePosition=[x, y, trunk_h / 2])
            # Canopy: cone approximated as sphere on top
            canopy_r = np.random.uniform(0.8, 1.5)
            canopy_z = trunk_h + canopy_r * 0.7
            col_c = p.createCollisionShape(p.GEOM_SPHERE, radius=canopy_r)
            vis_c = p.createVisualShape(p.GEOM_SPHERE, radius=canopy_r,
                                        rgbaColor=[0.15, 0.55, 0.15, 1.0])
            canopy_id = p.createMultiBody(baseMass=0,
                                          baseCollisionShapeIndex=col_c,
                                          baseVisualShapeIndex=vis_c,
                                          basePosition=[x, y, canopy_z])
            # Both trunk and canopy go into obstacle_ids → code 2
            return [trunk_id, canopy_id]

        elif shape == "rock":
            # Flat, wide ellipsoid-like box at ground level
            rx = np.random.uniform(0.6, 1.5)
            ry = np.random.uniform(0.6, 1.5)
            rz = np.random.uniform(0.25, 0.75)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[rx, ry, rz])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[rx, ry, rz],
                                      rgbaColor=[0.5, 0.47, 0.44, 1.0])
            uid = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[x, y, rz])
            return [uid]

        elif shape == "wall":
            # Long thin box, wall_length capped at 3-6m, height 2-5m
            wall_len = np.random.uniform(3.0, 6.0)
            wall_h   = np.random.uniform(2.0, 5.0)
            wall_t   = 0.2
            col = p.createCollisionShape(p.GEOM_BOX,
                                         halfExtents=[wall_len / 2, wall_t, wall_h / 2])
            vis = p.createVisualShape(p.GEOM_BOX,
                                      halfExtents=[wall_len / 2, wall_t, wall_h / 2],
                                      rgbaColor=[0.7, 0.7, 0.65, 1.0])
            uid = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[x, y, wall_h / 2])
            return [uid]

        return []

    # ══════════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _find_clear_yaw(self):
        """Sweep yaw in 15° steps; return first yaw where survivor is visible."""
        for deg in range(0, 360, 15):
            grid, _ = self.sensor.fire(
                self.drone_x, self.drone_y, self.drone_alt, float(deg),
                ground_id=self.ground_id,
                obstacle_ids=self.obstacle_ids,
                survivor_id=self.survivor_id,
                physics_client=self.physics_client
            )
            if int(3) in grid:
                return float(deg)
        return None

    def get_min_obstacle_dist(self):
        min_dist = self.RAY_RANGE
        for oid in self.obstacle_ids:
            pts = p.getClosestPoints(self.drone_id, oid, distance=6.0)
            for pt in pts:
                if pt[8] < min_dist:
                    min_dist = pt[8]
        return min_dist

    def collision_detected(self):
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.drone_id, oid):
                return True
        return False

    def survivor_reached(self):
        return math.hypot(
            self.drone_x - self.survivor_x,
            self.drone_y - self.survivor_y
        ) <= self.SURVIVOR_REACH_DIST

    def survivor_in_fov(self, ray_grid=None):
        if ray_grid is None:
            ray_grid = self._last_ray_grid
        return int(3) in ray_grid

    def render(self):
        pass

    def close(self):
        p.disconnect(self.physics_client)
