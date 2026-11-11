"""
rl_drone_env_multi.py - Multi-Cone Search Environment
======================================================
Same as rl_drone_env.py but the drone must find ALL cones (e.g. 3/3).
Each cone found gives a reward. Episode ends when all cones are found
or max steps reached.

HOW IT DIFFERS FROM rl_drone_env.py:
  - Places multiple cones (NUM_CONES) at random positions
  - Tracks which cones have been found
  - Episode ends when all cones found OR max steps
  - Extra observation: "found_count" (how many found so far)

ACTION SPACE: Same as rl_drone_env.py (Discrete(6))
  0=Forward, 1=Up, 2=Down, 3=Hover, 4=YawLeft, 5=YawRight

REQUIRES: pip install gymnasium numpy Pillow
"""
import setup_path
import airsim
import numpy as np
import math
import time

import gymnasium as gym
from gymnasium import spaces

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
TARGET_MESH    = "PersonTarget*"
TARGET_LABEL   = "Person Target"
NUM_CONES      = 3             # ← how many cones to find

VEHICLE        = "Drone1"
CAMERA         = "0"
LIDAR_SURR     = "SurroundLidar"
DIST_FRONT     = "FrontDistance"

STEP_LENGTH    = 5.0
VERT_STEP      = 2.0
YAW_STEP       = 45.0
MAX_STEPS      = 300            # more steps since finding 3 targets
DETECT_RADIUS  = 200 * 100
IMAGE_SHAPE    = (84, 84, 1)

SPAWN_MIN_XY   = -30.0
SPAWN_MAX_XY   =  30.0
SPAWN_Z        = -5.0
CONE_MIN_DIST  = 8.0
CONE_MAX_DIST  = 25.0

BOUNDS_MIN     = (-50, -50, -30)
BOUNDS_MAX     = (50, 50, 0)
LIDAR_DANGER   = 2.0
TARGET_FOUND_DIST = 15.0


class AirSimDroneMultiConeEnv(gym.Env):
    """Gymnasium env: FPV drone must find ALL cones in the scene.
    
    The drone gets a big reward for each cone found. Episode ends
    when all NUM_CONES cones are found or max steps reached.
    Cones are placed at random positions every episode.
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(self, ip="127.0.0.1", render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # ── AirSim ──
        self.drone = airsim.MultirotorClient(ip=ip)
        self.drone.confirmConnection()
        
        # ── Spaces ──
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=IMAGE_SHAPE, dtype=np.uint8),
            "lidar": spaces.Box(0, 100, shape=(8,), dtype=np.float32),
            "target_dir": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            "found_ratio": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(6)
        
        # ── State ──
        self.step_count = 0
        self.prev_dist_to_nearest = None
        self.drone_yaw = 0.0
        self.cone_names = []
        self.cube_names = []
        self.cone_positions = {}    # name -> np.array([x,y,z])
        self.cones_found = set()   # names of cones already found
        self._scanned = False
        self._drone_start = (0, 0, SPAWN_Z)
        
        print(f"[RL ENV MULTI] Target: {TARGET_LABEL} × {NUM_CONES}")
        print(f"[RL ENV MULTI] Step: {STEP_LENGTH}m, yaw: {YAW_STEP}°, max {MAX_STEPS} steps")
    
    # ─── SCENE SCANNING ─────────────────────────────────────
    # (same as rl_drone_env.py)
    
    def _scan_scene_objects(self):
        if self._scanned:
            return
        try:
            all_objs = self.drone.simListSceneObjects(".*")
            self.cube_names = [n for n in all_objs if n.lower().startswith("movingcube")]
            self.cone_names = [n for n in all_objs if n.lower().startswith("persontarget")]
            print(f"[RL ENV MULTI] Found {len(self.cube_names)} moving cubes, {len(self.cone_names)} person targets")
            
            if len(self.cone_names) < NUM_CONES:
                print(f"[RL ENV MULTI] WARNING: Need {NUM_CONES} targets but only {len(self.cone_names)} exist!")
                print(f"[RL ENV MULTI] Add more PersonTarget actors in Unreal Editor.")
        except Exception as e:
            print(f"[RL ENV MULTI] Scene scan failed: {e}")
        self._scanned = True
    
    # ─── DOMAIN RANDOMIZATION ───────────────────────────────
    
    def _randomize_scene(self):
        rng = self.np_random
        
        # 1. Random drone spawn
        dx = float(rng.uniform(SPAWN_MIN_XY, SPAWN_MAX_XY))
        dy = float(rng.uniform(SPAWN_MIN_XY, SPAWN_MAX_XY))
        self.drone_yaw = float(rng.uniform(0, 360))
        self._drone_start = (dx, dy, SPAWN_Z)
        
        # 2. Place NUM_CONES cones at random positions
        self.cone_positions = {}
        placed = [(dx, dy)]  # avoid drone position
        
        num_to_place = min(NUM_CONES, len(self.cone_names))
        for i in range(num_to_place):
            for _attempt in range(50):
                angle = float(rng.uniform(0, 2 * math.pi))
                dist = float(rng.uniform(CONE_MIN_DIST, CONE_MAX_DIST))
                cx = dx + dist * math.cos(angle)
                cy = dy + dist * math.sin(angle)
                cx = float(np.clip(cx, BOUNDS_MIN[0] + 5, BOUNDS_MAX[0] - 5))
                cy = float(np.clip(cy, BOUNDS_MIN[1] + 5, BOUNDS_MAX[1] - 5))
                
                # Check minimum separation from all placed objects
                too_close = any(
                    math.sqrt((cx - px)**2 + (cy - py)**2) < 5.0
                    for px, py in placed
                )
                if not too_close:
                    break
            
            cone_name = self.cone_names[i]
            cone_pose = airsim.Pose(
                airsim.Vector3r(cx, cy, 0),
                airsim.to_quaternion(0, 0, 0)
            )
            try:
                self.drone.simSetObjectPose(cone_name, cone_pose, True)
            except:
                pass
            
            self.cone_positions[cone_name] = np.array([cx, cy, 0.0])
            placed.append((cx, cy))
        
        # Hide extra cones far away if scene has more than NUM_CONES
        for i in range(num_to_place, len(self.cone_names)):
            try:
                hide_pose = airsim.Pose(
                    airsim.Vector3r(999, 999, -100),
                    airsim.to_quaternion(0, 0, 0)
                )
                self.drone.simSetObjectPose(self.cone_names[i], hide_pose, True)
            except:
                pass
        
        self.cones_found = set()
        
        # 3. Scatter cubes (same as rl_drone_env.py)
        for cube in self.cube_names:
            for _attempt in range(10):
                qx = float(rng.uniform(SPAWN_MIN_XY - 10, SPAWN_MAX_XY + 10))
                qy = float(rng.uniform(SPAWN_MIN_XY - 10, SPAWN_MAX_XY + 10))
                too_close = any(
                    abs(qx - px) < 3 and abs(qy - py) < 3
                    for px, py in placed
                )
                if not too_close:
                    break
            cube_pose = airsim.Pose(
                airsim.Vector3r(qx, qy, 0),
                airsim.to_quaternion(0, 0, float(rng.uniform(0, 2 * math.pi)))
            )
            try:
                self.drone.simSetObjectPose(cube, cube_pose, True)
            except:
                pass
        
        positions_str = " | ".join(
            f"{n}=({p[0]:.0f},{p[1]:.0f})" for n, p in self.cone_positions.items()
        )
        print(f"[RL ENV MULTI] Drone=({dx:.0f},{dy:.0f}) yaw={self.drone_yaw:.0f}° | Cones: {positions_str}")
    
    # ─── FLIGHT SETUP ───────────────────────────────────────
    # (same as rl_drone_env.py)
    
    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True, VEHICLE)
        self.drone.armDisarm(True, VEHICLE)
        
        self.drone.simSetDetectionFilterRadius(CAMERA, airsim.ImageType.Scene, DETECT_RADIUS, VEHICLE)
        self.drone.simAddDetectionFilterMeshName(CAMERA, airsim.ImageType.Scene, TARGET_MESH, VEHICLE)
        
        sx, sy, sz = self._drone_start
        start_pose = airsim.Pose(
            airsim.Vector3r(sx, sy, sz),
            airsim.to_quaternion(0, 0, math.radians(self.drone_yaw))
        )
        self.drone.simSetVehiclePose(start_pose, True, VEHICLE)
        time.sleep(0.3)
        
        self.drone.enableApiControl(True, VEHICLE)
        self.drone.armDisarm(True, VEHICLE)
        self.drone.moveToPositionAsync(sx, sy, sz, 5, vehicle_name=VEHICLE).join()
        time.sleep(0.2)
        
        self.step_count = 0
        self.prev_dist_to_nearest = self._dist_to_nearest_unfound()
    
    # ─── SENSORS ────────────────────────────────────────────
    # (same as rl_drone_env.py — _get_position, _get_depth_image, _get_lidar_sectors)
    
    def _get_position(self):
        state = self.drone.getMultirotorState(vehicle_name=VEHICLE)
        p = state.kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val])
    
    def _nearest_unfound_cone(self):
        """Get position of nearest cone not yet found."""
        pos = self._get_position()
        best_dist = float('inf')
        best_pos = np.array([0, 0, 0], dtype=np.float32)
        for name, cpos in self.cone_positions.items():
            if name in self.cones_found:
                continue
            d = float(np.linalg.norm(pos[:2] - cpos[:2]))
            if d < best_dist:
                best_dist = d
                best_pos = cpos
        return best_pos, best_dist
    
    def _dist_to_nearest_unfound(self):
        _, d = self._nearest_unfound_cone()
        return d
    
    def _get_depth_image(self):
        try:
            resp = self.drone.simGetImages([
                airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPerspective, True, False)
            ], vehicle_name=VEHICLE)
            if resp[0].width == 0:
                return np.zeros(IMAGE_SHAPE, dtype=np.uint8)
            img = np.array(resp[0].image_data_float, dtype=np.float32)
            img = 255.0 / np.maximum(np.ones_like(img), img)
            img = img.reshape(resp[0].height, resp[0].width)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize((84, 84))
            return np.array(pil_img).reshape(IMAGE_SHAPE).astype(np.uint8)
        except Exception:
            return np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    
    def _get_lidar_sectors(self):
        sectors = np.full(8, 30.0, dtype=np.float32)
        try:
            data = self.drone.getLidarData(LIDAR_SURR, VEHICLE)
            if len(data.point_cloud) < 3:
                return sectors
            pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
            angles = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
            dists = np.linalg.norm(pts[:, :2], axis=1)
            for i, a in enumerate(angles):
                s = int(((a + 180 + 22.5) % 360) / 45) % 8
                sectors[s] = min(sectors[s], dists[i])
        except Exception:
            pass
        return sectors
    
    def _get_target_direction(self):
        pos = self._get_position()
        target, _ = self._nearest_unfound_cone()
        diff = target - pos
        norm = np.linalg.norm(diff)
        if norm < 0.01:
            return np.zeros(3, dtype=np.float32)
        return (diff / norm).astype(np.float32)
    
    def _check_cone_detection(self):
        """Check if any unfound cone is detected and close enough."""
        try:
            dets = self.drone.simGetDetections(CAMERA, airsim.ImageType.Scene, VEHICLE)
            if not dets:
                return False, []
            
            pos = self._get_position()
            newly_found = []
            for det in dets:
                det_name = det.name
                # Check against unfound cones
                for cone_name, cone_pos in self.cone_positions.items():
                    if cone_name in self.cones_found:
                        continue
                    dist = float(np.linalg.norm(pos[:2] - cone_pos[:2]))
                    if dist < TARGET_FOUND_DIST and cone_name.lower() in det_name.lower():
                        newly_found.append(cone_name)
                
                # Also match by proximity if name doesn't match exactly
                if not newly_found:
                    for cone_name, cone_pos in self.cone_positions.items():
                        if cone_name in self.cones_found:
                            continue
                        dist = float(np.linalg.norm(pos[:2] - cone_pos[:2]))
                        if dist < TARGET_FOUND_DIST:
                            newly_found.append(cone_name)
            
            return len(newly_found) > 0, newly_found
        except:
            return False, []
    
    def _get_obs(self):
        found_ratio = len(self.cones_found) / max(len(self.cone_positions), 1)
        return {
            "image": self._get_depth_image(),
            "lidar": self._get_lidar_sectors(),
            "target_dir": self._get_target_direction(),
            "found_ratio": np.array([found_ratio], dtype=np.float32),
        }
    
    # ─── ACTIONS ────────────────────────────────────────────
    # (same as rl_drone_env.py)
    
    def _do_action(self, action):
        pos = self._get_position()
        
        if action == 0:  # Forward
            rad = math.radians(self.drone_yaw)
            nx = pos[0] + STEP_LENGTH * math.cos(rad)
            ny = pos[1] + STEP_LENGTH * math.sin(rad)
            self.drone.moveToPositionAsync(
                nx, ny, pos[2], 8,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()
        elif action == 1:  # Up
            self.drone.moveToPositionAsync(
                pos[0], pos[1], pos[2] - VERT_STEP, 4,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()
        elif action == 2:  # Down
            nz = min(pos[2] + VERT_STEP, -1.0)
            self.drone.moveToPositionAsync(
                pos[0], pos[1], nz, 4,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()
        elif action == 3:  # Hover
            self.drone.hoverAsync(vehicle_name=VEHICLE).join()
            time.sleep(0.1)
        elif action == 4:  # Yaw Left
            self.drone_yaw = (self.drone_yaw - YAW_STEP) % 360
            self.drone.rotateToYawAsync(self.drone_yaw, vehicle_name=VEHICLE).join()
        elif action == 5:  # Yaw Right
            self.drone_yaw = (self.drone_yaw + YAW_STEP) % 360
            self.drone.rotateToYawAsync(self.drone_yaw, vehicle_name=VEHICLE).join()
    
    # ─── REWARD ─────────────────────────────────────────────
    
    def _compute_reward(self):
        collision = self.drone.simGetCollisionInfo(VEHICLE).has_collided
        detected, newly_found = self._check_cone_detection()
        dist = self._dist_to_nearest_unfound()
        sectors = self._get_lidar_sectors()
        pos = self._get_position()
        
        out_of_bounds = any([
            pos[0] < BOUNDS_MIN[0], pos[1] < BOUNDS_MIN[1], pos[2] < BOUNDS_MIN[2],
            pos[0] > BOUNDS_MAX[0], pos[1] > BOUNDS_MAX[1], pos[2] > BOUNDS_MAX[2],
        ])
        
        reward = -0.2
        done = False
        truncated = False
        info = {
            "collision": collision,
            "detected": detected,
            "dist": dist,
            "found": len(self.cones_found),
            "total": len(self.cone_positions),
        }
        
        if collision:
            reward = -100.0
            done = True
            info["reason"] = "collision"
        elif out_of_bounds:
            reward = -50.0
            done = True
            info["reason"] = "out_of_bounds"
        elif self.step_count >= MAX_STEPS:
            reward = -20.0
            truncated = True
            info["reason"] = "max_steps"
        else:
            # Reward for newly found cones
            if newly_found:
                for cone_name in newly_found:
                    if cone_name not in self.cones_found:
                        self.cones_found.add(cone_name)
                        reward += 200.0  # big reward per cone
                        print(f"  🎯 FOUND {cone_name}! ({len(self.cones_found)}/{len(self.cone_positions)})")
                
                # Check if all found
                if len(self.cones_found) >= len(self.cone_positions):
                    reward += 100.0  # bonus for finding all
                    done = True
                    info["reason"] = "all_found"
                    print(f"  ✅ ALL CONES FOUND!")
                else:
                    # Reset distance tracking to next nearest
                    self.prev_dist_to_nearest = self._dist_to_nearest_unfound()
                
                info["found"] = len(self.cones_found)
            else:
                # Distance shaping toward nearest unfound
                if self.prev_dist_to_nearest is not None:
                    delta = self.prev_dist_to_nearest - dist
                    reward += delta * 3.0
                    if delta > 0:
                        reward += 1.0
                
                # LiDAR proximity penalty
                if np.min(sectors) < LIDAR_DANGER:
                    reward -= 5.0
                
                # Partial reward for seeing a cone
                if detected:
                    reward += 15.0
        
        self.prev_dist_to_nearest = dist
        return reward, done, truncated, info
    
    def step(self, action):
        self._do_action(action)
        self.step_count += 1
        obs = self._get_obs()
        reward, done, truncated, info = self._compute_reward()
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._scan_scene_objects()
        self._randomize_scene()
        self._setup_flight()
        obs = self._get_obs()
        return obs, {}
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_depth_image().squeeze()
        return None
    
    def close(self):
        try:
            self.drone.armDisarm(False, VEHICLE)
            self.drone.enableApiControl(False, VEHICLE)
        except:
            pass


if __name__ == "__main__":
    print(f"Testing Multi-Cone RL environment ({NUM_CONES} cones)...")
    env = AirSimDroneMultiConeEnv()
    obs, info = env.reset()
    
    print(f"Observation spaces:")
    print(f"  image:      {obs['image'].shape}")
    print(f"  lidar:      {obs['lidar']}")
    print(f"  target_dir: {obs['target_dir']}")
    print(f"  found:      {obs['found_ratio']}")
    
    action_names = ["Forward", "Up", "Down", "Hover", "Yaw Left", "Yaw Right"]
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: {action_names[action]:10s} R={reward:.2f} found={info.get('found',0)}/{info.get('total',0)}")
        if done or truncated:
            obs, info = env.reset()
    
    env.close()
    print("Test complete!")
