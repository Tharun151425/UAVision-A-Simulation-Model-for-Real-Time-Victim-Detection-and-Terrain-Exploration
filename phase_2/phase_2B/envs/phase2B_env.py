"""
Phase 2B Environment — Partial Occlusion, No GRU
=================================================
Objective: Learn to re-acquire survivor after mid-flight occlusion.

RULE: Survivor IS ray-visible at spawn.
      Obstacles may come between drone and survivor as drone moves.

Internal 3-level curriculum:
  L1: ~20% of flights have mid-flight occlusion  (8-12 obstacles, biased away from LOS)
  L2: ~50% of flights have mid-flight occlusion  (10-15 obstacles, random placement)
  L3: ~80% of flights have mid-flight occlusion  (12-18 obstacles, dense random)

Network: MlpPolicy (no GRU). Load Phase 2A final checkpoint.
Obs space: 262 — same as Phase 1/1.5/2A.
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

from phase_2.phase_2B.sensors.raycast import RaycastSensor
from phase_2.phase_2B.rewards.phase2B_rewards import compute_reward_2B

CURRICULUM_LEVELS = {
    1: {
        "min_obstacles":      8,
        "max_obstacles":      12,
        "min_height":         2.0,
        "max_height":         7.0,
        "shapes":             ["cylinder", "box", "tree"],
        "drone_safe_dist":    10.0,  # INCREASED: 10m gives 10+ steps before any obstacle
        "survivor_safe_dist": 3.0,
        # Bias obstacles away from LOS path at L1
        "los_bias":           True,
        "promote_success":    0.65,
        "promote_crash":      0.25,
        "promote_timeout":    0.20,
        "promote_window":     100,
        "description":        "2B-L1 — ~20% mid-flight occlusion",
    },
    2: {
        "min_obstacles":      10,
        "max_obstacles":      15,
        "min_height":         2.0,
        "max_height":         7.0,
        "shapes":             ["cylinder", "box", "tree", "rock", "wall"],
        "drone_safe_dist":    10.0,  # INCREASED from 8.0
        "survivor_safe_dist": 3.0,
        "los_bias":           False,
        "promote_success":    0.60,
        "promote_crash":      0.20,
        "promote_timeout":    0.30,
        "promote_window":     100,
        "description":        "2B-L2 — ~50% mid-flight occlusion",
    },
    3: {
        "min_obstacles":      12,
        "max_obstacles":      18,
        "min_height":         2.0,
        "max_height":         8.0,
        "shapes":             ["cylinder", "box", "tree", "rock", "wall"],
        "drone_safe_dist":    10.0,  # INCREASED from 8.0
        "survivor_safe_dist": 3.0,
        "los_bias":           False,
        "promote_success":    None,
        "promote_crash":      None,
        "promote_timeout":    None,
        "promote_window":     100,
        "description":        "2B-L3 — ~80% mid-flight occlusion (pass-criteria level)",
    },
}

PASS_CRITERIA = {"success_rate": 55.0, "crash_rate": 12.0, "timeout_rate": 33.0}


class Phase2BEnv(gym.Env):
    """Phase 2B SAR drone — partial occlusion, MlpPolicy."""

    _LEVEL_MAX_STEPS = {1: 1200, 2: 1200, 3: 1200}
    MAP_SIZE            = 40
    RAY_RANGE           = 20.0
    FORWARD_STEP        = 1.0
    YAW_STEP            = 15.0
    ALTITUDE_STEP       = 0.5
    MIN_ALTITUDE        = 1.0
    MAX_ALTITUDE        = 10.0
    SURVIVOR_REACH_DIST = 2.0
    SURVIVOR_MIN_DIST   = 10
    SURVIVOR_MAX_DIST   = 25
    CRASH_DIAG          = False  # set True in test script to log crash details

    TURN_LEFT    = 0
    TURN_RIGHT   = 1
    MOVE_UP      = 2
    MOVE_DOWN    = 3
    MOVE_FORWARD = 4
    SCAN         = 5

    @property
    def MAX_STEPS(self):
        return self._LEVEL_MAX_STEPS[self.curriculum_level]

    def __init__(self, render=False, curriculum_level=1):
        super().__init__()
        self.render_mode = render
        self.curriculum_level = max(1, min(3, curriculum_level))

        self._outcome_success = deque(maxlen=100)
        self._outcome_crash   = deque(maxlen=100)
        self._outcome_timeout = deque(maxlen=100)
        self._episode_count   = 0

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=10.0, shape=(262,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(6)

        if render:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=40, cameraPitch=-50,
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
        self.survivor_was_visible = False
        self.prev_dist       = 0.0
        self.done            = False
        self.step_count      = 0
        self._last_ray_grid  = np.zeros((8, 32), dtype=np.int32)
        self._consecutive_turns = 0
        self._lost_for_steps    = 0   # steps survivor has been hidden post-detection
        self._reacquisitions    = 0   # re-acquisition bonuses given this episode

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, 0)

        self.step_count              = 0
        self.done                    = False
        self.survivor_detected       = False
        self.survivor_was_visible    = False
        self._consecutive_turns      = 0
        self._lost_for_steps         = 0
        self._reacquisitions         = 0
        self.steps_since_signal_loss = 0  # for re-acquisition turn incentive

        self.ground_id = p.loadURDF("plane.urdf")

        if seed is not None:
            np.random.seed(seed)

        half = (self.MAP_SIZE / 2) - 1.0
        lvl  = CURRICULUM_LEVELS[self.curriculum_level]

        for _ in range(100):
            self.drone_x = np.random.uniform(-half, half)
            self.drone_y = np.random.uniform(-half, half)
            self.drone_alt = np.random.uniform(2.5, 5.0)
            self.drone_yaw = np.random.uniform(0, 360)

            dist  = np.random.uniform(self.SURVIVOR_MIN_DIST, self.SURVIVOR_MAX_DIST)
            angle = np.random.uniform(0, 2 * math.pi)
            sx = self.drone_x + dist * math.cos(angle)
            sy = self.drone_y + dist * math.sin(angle)

            if -half <= sx <= half and -half <= sy <= half:
                self.survivor_x = sx
                self.survivor_y = sy
                self.prev_dist  = dist
                break

        self.drone_id    = self._spawn_drone()
        self.survivor_id = self._spawn_survivor()

        p.addUserDebugLine(
            [0.3, 0.0, 0.1], [1.5, 0.0, 0.1],
            lineColorRGB=[1, 0, 0], lineWidth=4,
            parentObjectUniqueId=self.drone_id, parentLinkIndex=-1
        )

        n = np.random.randint(lvl["min_obstacles"], lvl["max_obstacles"] + 1)
        self.obstacle_ids = self._spawn_obstacles(n, lvl)

        # Fix spawn yaw: find a forward direction clear of obstacles for 15m
        self.drone_yaw = self._get_safe_spawn_yaw()
        _quat = p.getQuaternionFromEuler([0, 0, math.radians(self.drone_yaw)])
        p.resetBasePositionAndOrientation(
            self.drone_id, [self.drone_x, self.drone_y, self.drone_alt], _quat
        )

        # Ensure survivor visible at spawn — rotate yaw if needed
        self._last_ray_grid, _ = self.sensor.fire(
            self.drone_x, self.drone_y, self.drone_alt, self.drone_yaw,
            ground_id=self.ground_id, obstacle_ids=self.obstacle_ids,
            survivor_id=self.survivor_id, physics_client=self.physics_client
        )

        if not self.survivor_in_fov():
            found_yaw = self._find_clear_yaw()
            if found_yaw is not None:
                self.drone_yaw = found_yaw
                quat = p.getQuaternionFromEuler([0, 0, math.radians(self.drone_yaw)])
                p.resetBasePositionAndOrientation(
                    self.drone_id, [self.drone_x, self.drone_y, self.drone_alt], quat
                )
                self._last_ray_grid, _ = self.sensor.fire(
                    self.drone_x, self.drone_y, self.drone_alt, self.drone_yaw,
                    ground_id=self.ground_id, obstacle_ids=self.obstacle_ids,
                    survivor_id=self.survivor_id, physics_client=self.physics_client
                )

        return self._get_obs(self._last_ray_grid), {}

    def step(self, action):
        self.step_count += 1
        self._apply_action(action)
        p.stepSimulation()

        ray_grid, _ = self.sensor.fire(
            self.drone_x, self.drone_y, self.drone_alt, self.drone_yaw,
            ground_id=self.ground_id, obstacle_ids=self.obstacle_ids,
            survivor_id=self.survivor_id, physics_client=self.physics_client
        )
        self._last_ray_grid = ray_grid

        current_dist   = math.hypot(self.drone_x - self.survivor_x,
                                     self.drone_y - self.survivor_y)
        delta_dist     = self.prev_dist - current_dist
        self.prev_dist = current_dist

        newly_revealed = int(np.sum(ray_grid > 0))  # passed for API compat; reward v2 does not use it
        obs    = self._get_obs(ray_grid)
        reward = compute_reward_2B(self, action, delta_dist, newly_revealed)

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
                if self.CRASH_DIAG:
                    _s = self.step_count
                    _bucket = ("EARLY (<150)"   if _s < 150  else
                               "APPROACH (150-300)" if _s < 300  else
                               "MID (300-500)"  if _s < 500  else
                               "LATE (500+)")
                    print(
                        f"[CRASH] step={_s:4d}  bucket={_bucket}"
                        f"  detected={self.survivor_detected}"
                        f"  visible={self.survivor_in_fov()}"
                        f"  min_obs={self.get_min_obstacle_dist():.1f}m"
                        f"  lost_steps={self.steps_since_signal_loss}"
                    )
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
            "step": self.step_count, "survivor_found": found,
            "crashed": crashed, "curriculum_level": self.curriculum_level,
            "termination_reason": reason,
        }
        return obs, reward, self.done, False, info

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
                f"\n[Phase 2B] PROMOTION: Level {old} → {self.curriculum_level}"
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
                            linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

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

    def _spawn_drone(self):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.1])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.1],
                                  rgbaColor=[0.2, 0.4, 0.9, 1.0])
        quat = p.getQuaternionFromEuler([0, 0, math.radians(self.drone_yaw)])
        return p.createMultiBody(baseMass=1.0,
                                 baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=[self.drone_x, self.drone_y, self.drone_alt],
                                 baseOrientation=quat)

    def _spawn_survivor(self):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=1.0)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5,
                                  rgbaColor=[1.0, 0.1, 0.1, 1.0])
        return p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis,
                                 basePosition=[self.survivor_x, self.survivor_y, 0.5])

    def _spawn_obstacles(self, count, lvl):
        obs_ids          = []
        placed_positions = []
        half = (self.MAP_SIZE / 2) - 0.5

        # Compute LOS direction for L1 bias
        los_dx = self.survivor_x - self.drone_x
        los_dy = self.survivor_y - self.drone_y
        los_len = math.hypot(los_dx, los_dy) + 1e-6

        for _ in range(count):
            for _ in range(300):
                x = np.random.uniform(-half, half)
                y = np.random.uniform(-half, half)

                if math.hypot(self.drone_x - x, self.drone_y - y) < lvl["drone_safe_dist"]:
                    continue
                # Soft exclusion: reject 70% of obstacles within 15m of drone spawn.
                # Prevents dense clusters near the drone even outside the hard radius.
                if math.hypot(self.drone_x - x, self.drone_y - y) < 15.0:
                    if np.random.random() < 0.7:
                        continue
                if math.hypot(self.survivor_x - x, self.survivor_y - y) < lvl["survivor_safe_dist"]:
                    continue
                # Approach corridor exclusion for L1/L2: obstacles in the
                # straight-line path between drone and survivor cause guaranteed
                # approach crashes. Allow them at L3 (drone must handle it).
                if self.curriculum_level in (1, 2):
                    if self._is_in_approach_corridor(x, y):
                        continue
                if any(math.hypot(px - x, py - y) < 2.0 for px, py in placed_positions):
                    continue

                # L1: bias obstacles away from direct LOS
                if lvl.get("los_bias"):
                    # Project obstacle onto LOS vector
                    t = ((x - self.drone_x) * los_dx + (y - self.drone_y) * los_dy) / (los_len ** 2)
                    proj_x = self.drone_x + t * los_dx
                    proj_y = self.drone_y + t * los_dy
                    perp_dist = math.hypot(x - proj_x, y - proj_y)
                    # If too close to LOS path, skip with 80% probability
                    if 0 < t < 1 and perp_dist < 3.0 and np.random.rand() < 0.8:
                        continue

                h     = np.random.uniform(lvl["min_height"], lvl["max_height"])
                shape = np.random.choice(lvl["shapes"])
                ids   = self._create_obstacle(shape, x, y, h)
                obs_ids.extend(ids)
                placed_positions.append((x, y))
                break

        return obs_ids

    def _create_obstacle(self, shape, x, y, h):
        if shape == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=h)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=h,
                                      rgbaColor=[0.55, 0.35, 0.15, 1.0])
            return [p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis, basePosition=[x, y, h/2])]
        elif shape == "box":
            hw = np.random.uniform(0.5, 2.5)
            hd = np.random.uniform(0.5, 2.5)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hw, hd, h/2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hw, hd, h/2],
                                      rgbaColor=[0.45, 0.45, 0.45, 1.0])
            return [p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis, basePosition=[x, y, h/2])]
        elif shape == "tree":
            trunk_h = h * 0.6
            col_t = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=trunk_h)
            vis_t = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=trunk_h,
                                        rgbaColor=[0.4, 0.25, 0.1, 1.0])
            trunk_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_t,
                                         baseVisualShapeIndex=vis_t,
                                         basePosition=[x, y, trunk_h/2])
            cr = np.random.uniform(0.8, 1.5)
            col_c = p.createCollisionShape(p.GEOM_SPHERE, radius=cr)
            vis_c = p.createVisualShape(p.GEOM_SPHERE, radius=cr,
                                        rgbaColor=[0.15, 0.55, 0.15, 1.0])
            canopy_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_c,
                                          baseVisualShapeIndex=vis_c,
                                          basePosition=[x, y, trunk_h + cr * 0.7])
            return [trunk_id, canopy_id]
        elif shape == "rock":
            rx, ry = np.random.uniform(0.6, 1.5), np.random.uniform(0.6, 1.5)
            rz = np.random.uniform(0.25, 0.75)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[rx, ry, rz])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[rx, ry, rz],
                                      rgbaColor=[0.5, 0.47, 0.44, 1.0])
            return [p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis, basePosition=[x, y, rz])]
        elif shape == "wall":
            wl = np.random.uniform(3.0, 6.0)
            wh = np.random.uniform(2.0, 5.0)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wl/2, 0.2, wh/2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[wl/2, 0.2, wh/2],
                                      rgbaColor=[0.7, 0.7, 0.65, 1.0])
            return [p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis, basePosition=[x, y, wh/2])]
        return []

    def _is_in_approach_corridor(self, ox, oy):
        """
        Returns True if (ox, oy) lies within the 3m-wide corridor between
        drone spawn and survivor. Obstacles here cause guaranteed approach crashes
        because the drone flies straight at the survivor and hits the wall.
        Used to soft-exclude placements at L1 and L2.
        """
        dx = self.survivor_x - self.drone_x
        dy = self.survivor_y - self.drone_y
        length = math.hypot(dx, dy)
        if length < 0.001:
            return False
        nx, ny = dx / length, dy / length
        ox_rel = ox - self.drone_x
        oy_rel = oy - self.drone_y
        projection = ox_rel * nx + oy_rel * ny
        if projection < 0 or projection > length:
            return False
        perp_dist = abs(ox_rel * ny - oy_rel * nx)
        return perp_dist < 3.0

    def _get_safe_spawn_yaw(self):
        """
        Find a spawn yaw where the forward direction is obstacle-free for 15m.
        Called AFTER obstacles are spawned so ray results are accurate.
        Tries all 10-degree increments in random order; falls back to random.
        """
        yaw_candidates = list(range(0, 360, 10))
        np.random.shuffle(yaw_candidates)
        for yaw_deg in yaw_candidates:
            yaw_rad = math.radians(yaw_deg)
            fx = math.cos(yaw_rad)
            fy = math.sin(yaw_rad)
            ray_end = [
                self.drone_x + fx * 15.0,
                self.drone_y + fy * 15.0,
                self.drone_alt,
            ]
            hit = p.rayTest(
                [self.drone_x, self.drone_y, self.drone_alt],
                ray_end
            )[0]
            # Safe: ray hits nothing, or only the ground plane
            if hit[0] == -1 or hit[0] == self.ground_id:
                return float(yaw_deg)
        # Fallback (should rarely trigger with 10m clear radius)
        return float(np.random.randint(0, 36) * 10)

    def _find_clear_yaw(self):
        for deg in range(0, 360, 15):
            grid, _ = self.sensor.fire(
                self.drone_x, self.drone_y, self.drone_alt, float(deg),
                ground_id=self.ground_id, obstacle_ids=self.obstacle_ids,
                survivor_id=self.survivor_id, physics_client=self.physics_client
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

    def get_front_obstacle_dist(self):
        front_dist = self.RAY_RANGE
        yaw_rad = math.radians(self.drone_yaw)
        fx = math.cos(yaw_rad)
        fy = math.sin(yaw_rad)
        
        for oid in self.obstacle_ids:
            pts = p.getClosestPoints(self.drone_id, oid, distance=6.0)
            for pt in pts:
                dist = pt[8]
                posA = pt[5]  # on drone
                posB = pt[6]  # on obstacle
                dx = posB[0] - posA[0]
                dy = posB[1] - posA[1]
                
                length = math.hypot(dx, dy)
                if length > 0.01:
                    cos_angle = (dx * fx + dy * fy) / length
                    if cos_angle > 0.5:  # within +/- 60 degrees of front
                        if dist < front_dist:
                            front_dist = dist
        return front_dist

    def collision_detected(self):
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.drone_id, oid):
                return True
        return False

    def survivor_reached(self):
        return math.hypot(self.drone_x - self.survivor_x,
                          self.drone_y - self.survivor_y) <= self.SURVIVOR_REACH_DIST

    def survivor_in_fov(self, ray_grid=None):
        if ray_grid is None:
            ray_grid = self._last_ray_grid
        return int(3) in ray_grid

    def render(self):
        pass

    def close(self):
        p.disconnect(self.physics_client)
