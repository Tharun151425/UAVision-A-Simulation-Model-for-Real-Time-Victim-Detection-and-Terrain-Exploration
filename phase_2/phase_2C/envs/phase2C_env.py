"""
Phase 2C Environment — Full Occlusion with GRU Memory
======================================================
Objective: Systematic search using GRU memory when survivor may be
           completely hidden from spawn.

Key differences from 2B:
  - MAP_SIZE = 50 (larger — needs actual search)
  - Survivor may be completely hidden at spawn (NO ray-visibility guarantee)
  - Only requirement: BFS-navigable path must exist (checked via obstacle density)
  - FogOfWarMap tracks explored cells → enables revisit penalty
  - Internal 3-level curriculum (40% / 70% / 90% hidden at spawn)

Network: RecurrentPPO with GRU (sb3-contrib).
Obs space: 262 — same as all previous phases (GRU receives obs each step).
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

from phase_2.phase_2C.sensors.raycast import RaycastSensor
from phase_2.phase_2C.rewards.phase2C_rewards import compute_reward_2C


# ── Fog-of-War map ────────────────────────────────────────────────────────────
class FogOfWarMap:
    """
    Tracks which cells the drone has already visited.
    grid values: -1 = unknown, 0 = visited
    Used for coverage bonus and revisit penalty.
    """
    def __init__(self, world_size=50, cell_size=2.0):
        self.world_size = world_size
        self.cell_size  = cell_size
        n = int(world_size / cell_size)
        self.n    = n
        self.grid = np.full((n, n), -1, dtype=np.int8)  # -1 = unknown

    def reset(self):
        self.grid[:] = -1

    def world_to_cell(self, x, y):
        cx = int((x + self.world_size / 2) / self.cell_size)
        cy = int((y + self.world_size / 2) / self.cell_size)
        cx = np.clip(cx, 0, self.n - 1)
        cy = np.clip(cy, 0, self.n - 1)
        return cx, cy

    def visit(self, x, y):
        """Mark current drone cell as visited. Returns True if it was unknown."""
        cx, cy = self.world_to_cell(x, y)
        was_unknown = self.grid[cy][cx] == -1
        self.grid[cy][cx] = 0
        return was_unknown

    def is_known(self, x, y):
        cx, cy = self.world_to_cell(x, y)
        return self.grid[cy][cx] != -1

    def count_unknown_in_radius(self, x, y, radius=10.0):
        cx, cy   = self.world_to_cell(x, y)
        r_cells  = int(radius / self.cell_size)
        count    = 0
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.n and 0 <= ny < self.n:
                    if self.grid[ny][nx] == -1:
                        count += 1
        return count

    def total_unknown(self):
        return int(np.sum(self.grid == -1))


# ── Curriculum ────────────────────────────────────────────────────────────────
CURRICULUM_LEVELS = {
    1: {
        "min_obstacles":      10,
        "max_obstacles":      15,
        "hidden_prob":        0.40,
        "max_steps":          2000,
        "min_height":         2.0,
        "max_height":         7.0,
        "shapes":             ["cylinder", "box", "tree", "rock", "wall"],
        "drone_safe_dist":    10.0,
        "survivor_safe_dist": 4.0,
        "promote_success":    0.52,
        "promote_crash":      0.20,
        "promote_timeout":    0.36,
        "promote_window":     100,
        "description":        "2C-L1 — 40% hidden at spawn",
    },
    2: {
        "min_obstacles":      12,
        "max_obstacles":      18,
        "hidden_prob":        0.70,
        "max_steps":          2500,
        "min_height":         2.0,
        "max_height":         8.0,
        "shapes":             ["cylinder", "box", "tree", "rock", "wall"],
        "drone_safe_dist":    10.0,
        "survivor_safe_dist": 4.0,
        "promote_success":    0.48,
        "promote_crash":      0.20,
        "promote_timeout":    0.40,
        "promote_window":     100,
        "description":        "2C-L2 — 70% hidden at spawn",
    },
    3: {
        "min_obstacles":      15,
        "max_obstacles":      22,
        "hidden_prob":        0.90,
        "max_steps":          3000,
        "min_height":         2.0,
        "max_height":         8.0,
        "shapes":             ["cylinder", "box", "tree", "rock", "wall"],
        "drone_safe_dist":    10.0,
        "survivor_safe_dist": 4.0,
        "promote_success":    None,
        "promote_crash":      None,
        "promote_timeout":    None,
        "promote_window":     100,
        "description":        "2C-L3 — 90% hidden at spawn (pass-criteria level)",
    },
}

PASS_CRITERIA = {"success_rate": 48.0, "crash_rate": 12.0, "timeout_rate": 40.0}


class Phase2CEnv(gym.Env):
    """Phase 2C — full occlusion, GRU memory, FogOfWar search."""

    MAP_SIZE            = 50
    RAY_RANGE           = 20.0
    FORWARD_STEP        = 1.0
    YAW_STEP            = 15.0
    ALTITUDE_STEP       = 0.5
    MIN_ALTITUDE        = 1.0
    MAX_ALTITUDE        = 10.0
    SURVIVOR_REACH_DIST = 2.0
    SURVIVOR_MIN_DIST   = 10
    SURVIVOR_MAX_DIST   = 35
    CRASH_DIAG          = False  # set True in test script to log crash details

    TURN_LEFT    = 0
    TURN_RIGHT   = 1
    MOVE_UP      = 2
    MOVE_DOWN    = 3
    MOVE_FORWARD = 4
    SCAN         = 5

    @property
    def MAX_STEPS(self):
        return CURRICULUM_LEVELS[self.curriculum_level]["max_steps"]

    def __init__(self, render=False, curriculum_level=1):
        super().__init__()
        self.render_mode = render
        self.curriculum_level = max(1, min(3, curriculum_level))

        self._outcome_success = deque(maxlen=100)
        self._outcome_crash   = deque(maxlen=100)
        self._outcome_timeout = deque(maxlen=100)
        self._episode_count   = 0

        # Obs: 256 ray + 5 pose + 1 time = 262
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=10.0, shape=(262,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(6)

        if render:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=50, cameraPitch=-55,
                cameraYaw=45, cameraTargetPosition=[0, 0, 0]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        self.sensor  = RaycastSensor(
            h_rays=32, v_rays=8, h_fov=120, v_fov=60, max_range=self.RAY_RANGE
        )
        self.fog_map = FogOfWarMap(world_size=self.MAP_SIZE, cell_size=2.0)

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
        self._lost_for_steps    = 0
        self._reacquisitions    = 0

        self.reset()

    # ══════════════════════════════════════════════════════════════════════════
    #  Gym API
    # ══════════════════════════════════════════════════════════════════════════

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
        self.fog_map.reset()

        self.ground_id = p.loadURDF("plane.urdf")

        if seed is not None:
            np.random.seed(seed)

        half = (self.MAP_SIZE / 2) - 1.5
        lvl  = CURRICULUM_LEVELS[self.curriculum_level]

        for _ in range(100):
            self.drone_x   = np.random.uniform(-half, half)
            self.drone_y   = np.random.uniform(-half, half)
            self.drone_alt = np.random.uniform(3.0, 6.0)
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

        # At L1: 60% chance survivor is visible at spawn (hidden_prob=0.40)
        # At L3: 10% chance survivor is visible at spawn (hidden_prob=0.90)
        # We do NOT force visibility — obstacles are placed freely
        # Only guarantee: drone spawn has sufficient clearance

        self._last_ray_grid, _ = self.sensor.fire(
            self.drone_x, self.drone_y, self.drone_alt, self.drone_yaw,
            ground_id=self.ground_id, obstacle_ids=self.obstacle_ids,
            survivor_id=self.survivor_id, physics_client=self.physics_client
        )

        # Mark starting cell as visited
        self.fog_map.visit(self.drone_x, self.drone_y)

        return self._get_obs(self._last_ray_grid), {}

    def step(self, action):
        self.step_count += 1

        # Track cell before moving
        was_known = self.fog_map.is_known(self.drone_x, self.drone_y)

        self._apply_action(action)
        p.stepSimulation()

        # Mark new cell visited, track if it was already known
        current_was_known = self.fog_map.is_known(self.drone_x, self.drone_y)
        self.fog_map.visit(self.drone_x, self.drone_y)

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

        newly_revealed = int(np.sum(ray_grid > 0))

        obs    = self._get_obs(ray_grid)
        reward = compute_reward_2C(
            self, action, delta_dist,
            newly_revealed=newly_revealed,
            current_cell_was_known=current_was_known
        )

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
                    _bucket = ("EARLY (<150)"       if _s < 150 else
                               "APPROACH (150-300)" if _s < 300 else
                               "MID (300-500)"      if _s < 500 else
                               "LATE (500+)")
                    print(
                        f"[CRASH-2C] step={_s:4d}  bucket={_bucket}"
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
            "cells_explored": int(np.sum(self.fog_map.grid == 0)),
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
                f"\n[Phase 2C] PROMOTION: Level {old} → {self.curriculum_level}"
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
                            linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

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

        for _ in range(count):
            for _ in range(300):
                x = np.random.uniform(-half, half)
                y = np.random.uniform(-half, half)
                if math.hypot(self.drone_x - x, self.drone_y - y) < lvl["drone_safe_dist"]:
                    continue
                # Soft exclusion: 70% reject for obstacles within 15m of drone spawn
                if math.hypot(self.drone_x - x, self.drone_y - y) < 15.0:
                    if np.random.random() < 0.7:
                        continue
                if math.hypot(self.survivor_x - x, self.survivor_y - y) < lvl["survivor_safe_dist"]:
                    continue
                # Approach corridor exclusion for L1/L2
                if self.curriculum_level in (1, 2):
                    if self._is_in_approach_corridor(x, y):
                        continue
                if any(math.hypot(px - x, py - y) < 2.0 for px, py in placed_positions):
                    continue

                h     = np.random.uniform(lvl["min_height"], lvl["max_height"])
                shape = np.random.choice(lvl["shapes"])
                ids   = self._create_obstacle(shape, x, y, h)
                obs_ids.extend(ids)
                placed_positions.append((x, y))
                break

        return obs_ids

    def _is_in_approach_corridor(self, ox, oy):
        """
        Returns True if (ox, oy) lies within the 3m-wide corridor between
        drone spawn and survivor. Used to exclude obstacles at L1/L2.
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
            if hit[0] == -1 or hit[0] == self.ground_id:
                return float(yaw_deg)
        return float(np.random.randint(0, 36) * 10)

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
            col_t   = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=trunk_h)
            vis_t   = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=trunk_h,
                                          rgbaColor=[0.4, 0.25, 0.1, 1.0])
            tid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_t,
                                    baseVisualShapeIndex=vis_t,
                                    basePosition=[x, y, trunk_h/2])
            cr    = np.random.uniform(0.8, 1.5)
            col_c = p.createCollisionShape(p.GEOM_SPHERE, radius=cr)
            vis_c = p.createVisualShape(p.GEOM_SPHERE, radius=cr,
                                        rgbaColor=[0.15, 0.55, 0.15, 1.0])
            cid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_c,
                                    baseVisualShapeIndex=vis_c,
                                    basePosition=[x, y, trunk_h + cr * 0.7])
            return [tid, cid]
        elif shape == "rock":
            rx, ry = np.random.uniform(0.6, 1.5), np.random.uniform(0.6, 1.5)
            rz     = np.random.uniform(0.25, 0.75)
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

    # ══════════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════════

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
