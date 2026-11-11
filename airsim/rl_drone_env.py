"""
rl_drone_env.py - Gymnasium RL Environment for AirSim Drone (v3)
=================================================================
FPV drone that searches for a target (person).
Uses yaw-based turning (no backward/strafe), domain randomization.

FIXED IN V3:
  - Drone now HOVERS properly (takeoff before teleport)
  - Ground contact = DEATH (-200) — teaches model that falling is fatal
  - A*-inspired heading reward — guides drone toward target
  - Altitude enforcement — drone can't go below 2m
  - Grace period after spawn — ignores collisions for 0.5s
  - Exploration bonus — rewards visiting new areas

OBSERVATION SPACE:
  Dict with:
    "image"  : (84, 84, 1) depth image, uint8
    "lidar"  : (8,) float32, min distance per sector
    "target_dir" : (3,) float32, relative direction to target (unit vector)

ACTION SPACE:
  Discrete(5):
    0 = Forward (5m in facing direction)
    1 = Up (ascend 2m)
    2 = Down (descend 2m, floor at -2m altitude)
    3 = Yaw Left (rotate -45 degrees)
    4 = Yaw Right (rotate +45 degrees)

REWARD:
  +200   : Target found (close enough)
  -200   : Ground contact (death — Z > -0.5 NED)
  -100   : Building collision
  -50    : Out of bounds
  -20    : Max steps (timeout)
  -5     : LiDAR proximity warning (<2m)
  +15    : Partial target detection
  +dist  : Distance shaping (closer = more reward)
  +heading: A* heading bonus (facing target = reward)
  +1.0   : Altitude maintenance bonus
  -0.3   : Per-step penalty

REQUIRES: pip install gymnasium numpy Pillow
"""
import setup_path

# ── MSGPACK HOTFIX FOR LARGE IMAGES ──
import msgpack
_old_unpacker = msgpack.Unpacker
def _patched_unpacker(*args, **kwargs):
    kwargs['max_array_len'] = 104857600
    kwargs['max_bin_len'] = 104857600
    kwargs['max_str_len'] = 104857600
    return _old_unpacker(*args, **kwargs)
msgpack.Unpacker = _patched_unpacker

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
TARGET_LABEL   = "Dwight"

VEHICLE        = "Drone1"
CAMERA         = "0"
LIDAR_SURR     = "SurroundLidar"
DIST_FRONT     = "FrontDistance"

STEP_LENGTH    = 5.0       # meters per forward action
VERT_STEP      = 2.0       # meters per up/down action
YAW_STEP       = 45.0      # degrees per yaw action
MAX_STEPS      = 200       # max steps per episode
DETECT_RADIUS  = 200 * 100 # cm
IMAGE_SHAPE    = (84, 84, 1)

# Domain randomization bounds (NED)
SPAWN_MIN_XY   = -15.0
SPAWN_MAX_XY   =  15.0
SPAWN_Z        = -10.0      # drone altitude (NED, -5 = 5m up)
CONE_MIN_DIST  = 20.0
CONE_MAX_DIST  = 40.0

BOUNDS_MIN     = (-50, -50, -30)  # NED min
BOUNDS_MAX     = (50, 50, 0)      # NED max

# Safety thresholds
LIDAR_DANGER       = 2.0
TARGET_FOUND_DIST  = 15.0
GROUND_DEATH_Z     = -0.5   # NED: anything > this = ground contact = death
MIN_ALTITUDE_Z     = -2.0   # NED: drone can't descend above this
GOOD_ALT_MIN       = -8.0   # NED: ideal altitude range
GOOD_ALT_MAX       = -3.0

# Grace period — ignore collisions for this many seconds after spawn
SPAWN_GRACE_SEC    = 1.5


class AirSimDroneSearchEnv(gym.Env):
    """Gymnasium env: FPV drone searches for target, avoids obstacles.
    
    V3: Fixes hover bug, adds ground death, A*-inspired heading rewards.
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(self, ip="127.0.0.1", render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # ── AirSim ──
        self.drone = airsim.MultirotorClient(ip=ip)
        self.drone.confirmConnection()
        
        # ── Spaces (5 actions — removed Hover, it's useless for search) ──
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=IMAGE_SHAPE, dtype=np.uint8),
            "lidar": spaces.Box(0, 100, shape=(8,), dtype=np.float32),
            "target_dir": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(5)
        
        # ── State ──
        self.step_count = 0
        self.prev_dist_to_target = None
        self.target_pos = np.array([10.0, 10.0, -5.0])
        self.drone_yaw = 0.0
        self.cube_names = []
        self.cone_names = []
        self._scanned = False
        self._spawn_time = 0.0       # when episode started (for grace period)
        self._visited_cells = set()  # exploration tracking
        self._cell_size = 5.0        # meters per grid cell
        self._last_action = -1       # track last action for spin detection
        self._consecutive_yaws = 0   # count consecutive yaw-only actions
        
        # ── Setup detection ──
        self.drone.simSetDetectionFilterRadius(CAMERA, airsim.ImageType.Scene, DETECT_RADIUS, VEHICLE)
        self.drone.simAddDetectionFilterMeshName(CAMERA, airsim.ImageType.Scene, TARGET_MESH, VEHICLE)
        
        print(f"[RL ENV] Target: {TARGET_LABEL} (mesh: {TARGET_MESH})")
        print(f"[RL ENV] Step: {STEP_LENGTH}m forward, {YAW_STEP}° yaw, max {MAX_STEPS} steps")
        self.ep_count = 0
        self.last_reason = "started for the first time"

    # ─── SCENE SCAN ───────────────────────────────────────────
    def _scan_scene_objects(self):
        """Find all MovingCubes and PersonTarget actors in scene (once)."""
        if self._scanned:
            return
        try:
            all_objs = self.drone.simListSceneObjects(".*")
            self.cube_names = [n for n in all_objs if n.lower().startswith("movingcube")]
            self.cone_names = [n for n in all_objs if n.lower().startswith("persontarget")]
            print(f"[RL ENV] Found {len(self.cube_names)} moving cubes, {len(self.cone_names)} targets in scene")
        except Exception as e:
            print(f"[RL ENV] Scene scan failed: {e}")
            self.cube_names = []
            self.cone_names = []
        self._scanned = True

    # ─── DOMAIN RANDOMIZATION ─────────────────────────────────
    def _randomize_scene(self):
        """Domain randomization: scatter drone, cone, cubes."""
        rng = self.np_random
        
        # 1. Random drone spawn
        dx = float(rng.uniform(SPAWN_MIN_XY, SPAWN_MAX_XY))
        dy = float(rng.uniform(SPAWN_MIN_XY, SPAWN_MAX_XY))
        self.drone_yaw = float(rng.uniform(0, 360))
        self._drone_start = (dx, dy, SPAWN_Z)
        
        # 2. Random cone position (10-25m from drone)
        angle = float(rng.uniform(0, 2 * math.pi))
        dist = float(rng.uniform(CONE_MIN_DIST, CONE_MAX_DIST))
        cx = dx + dist * math.cos(angle)
        cy = dy + dist * math.sin(angle)
        cx = float(np.clip(cx, BOUNDS_MIN[0] + 5, BOUNDS_MAX[0] - 5))
        cy = float(np.clip(cy, BOUNDS_MIN[1] + 5, BOUNDS_MAX[1] - 5))
        
        if self.cone_names:
            cone_name = self.cone_names[0]
            cone_pose = airsim.Pose(
                airsim.Vector3r(cx, cy, 0),
                airsim.to_quaternion(0, 0, 0)
            )
            try:
                self.drone.simSetObjectPose(cone_name, cone_pose, True)
            except Exception as e:
                print(f"[RL ENV] Failed to move cone: {e}")
        
        self.target_pos = np.array([cx, cy, 0.0])
        
        # 3. Scatter cubes randomly
        for cube in self.cube_names:
            for _attempt in range(20):
                qx = float(rng.uniform(SPAWN_MIN_XY - 10, SPAWN_MAX_XY + 10))
                qy = float(rng.uniform(SPAWN_MIN_XY - 10, SPAWN_MAX_XY + 10))
                if (abs(qx - dx) > 10 and abs(qy - dy) > 10 and
                    abs(qx - cx) > 10 and abs(qy - cy) > 10):
                    break
            cube_yaw = float(rng.uniform(0, 2 * math.pi))
            cube_pose = airsim.Pose(
                airsim.Vector3r(qx, qy, 0),
                airsim.to_quaternion(0, 0, cube_yaw)
            )
            try:
                self.drone.simSetObjectPose(cube, cube_pose, True)
            except:
                pass

        # ═══ 3-LINE SPAWN LOG ═══
        self.ep_count += 1
        print(f"\nRun {self.ep_count} : spawned coz it {self.last_reason}...")
        
        ball_str = ""
        try:
            b_pose = self.drone.simGetObjectPose("OrangeBall")
            if not math.isnan(b_pose.position.x_val):
                ball_str = f" | 🟠 OrangeBall: ({b_pose.position.x_val:.1f}, {b_pose.position.y_val:.1f})"
        except: pass
        
        print(f"locations : 🚁 drone: ({dx:.1f}, {dy:.1f}){ball_str}")
        print(f"🎯 Dwight: ({cx:.1f}, {cy:.1f}) | 📦 Cubes: {len(self.cube_names)} | 📏 Dist: {dist:.1f}m")

    # ─── FLIGHT SETUP (FIXED — DRONE ACTUALLY HOVERS NOW) ────
    def _setup_flight(self):
        """Reset and prepare drone — BULLETPROOF hover sequence."""
        self.drone.reset()
        self.drone.enableApiControl(True, VEHICLE)
        self.drone.armDisarm(True, VEHICLE)
        
        self.drone.simSetDetectionFilterRadius(CAMERA, airsim.ImageType.Scene, DETECT_RADIUS, VEHICLE)
        self.drone.simAddDetectionFilterMeshName(CAMERA, airsim.ImageType.Scene, TARGET_MESH, VEHICLE)
        
        # 1. TAKEOFF FIRST so the propellers are spinning
        self.drone.takeoffAsync(timeout_sec=3).join()
        
        # 2. PAUSE PHYSICS so it doesn't fall during the teleport
        self.drone.simPause(True)
        
        # 3. Teleport
        sx, sy, sz = self._drone_start
        start_pose = airsim.Pose(
            airsim.Vector3r(sx, sy, sz),
            airsim.to_quaternion(0, 0, math.radians(self.drone_yaw))
        )
        self.drone.simSetVehiclePose(start_pose, True, VEHICLE)
        
        # 4. Command it to hold this exact spot
        self.drone.moveToPositionAsync(
            sx, sy, sz, 5, timeout_sec=2,
            yaw_mode=airsim.YawMode(False, self.drone_yaw),
            vehicle_name=VEHICLE
        )
        
        # 5. UNPAUSE and let it settle
        self.drone.simPause(False)
        time.sleep(1.0) # Let physics catch up
        
        self.drone.simGetCollisionInfo(VEHICLE) # Clear false flags
        self._spawn_time = time.time()
        self.step_count = 0
        self._visited_cells = set()
        self.prev_dist_to_target = self._dist_to_target()
        
        # ═══ NEW: Probation tracking ═══
        self._is_airborne = False

    # ─── SENSORS ──────────────────────────────────────────────
    def _get_position(self):
        """Current drone position as numpy array."""
        state = self.drone.getMultirotorState(vehicle_name=VEHICLE)
        p = state.kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val])

    def _dist_to_target(self):
        pos = self._get_position()
        return float(np.linalg.norm(pos - self.target_pos))

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
        diff = self.target_pos - pos
        norm = np.linalg.norm(diff)
        if norm < 0.01:
            return np.zeros(3, dtype=np.float32)
        return (diff / norm).astype(np.float32)

    def _target_in_fov(self):
        try:
            dets = self.drone.simGetDetections(CAMERA, airsim.ImageType.Scene, VEHICLE)
            return len(dets) > 0
        except:
            return False

    def _get_obs(self):
        return {
            "image": self._get_depth_image(),
            "lidar": self._get_lidar_sectors(),
            "target_dir": self._get_target_direction(),
        }

    # ─── ACTIONS (5 discrete — removed useless Hover) ─────────
    def _do_action(self, action):
        """Execute discrete action with yaw-based heading."""
        pos = self._get_position()
        timeout = 2.0
        
        if action == 0:  # Forward
            rad = math.radians(self.drone_yaw)
            nx = pos[0] + STEP_LENGTH * math.cos(rad)
            ny = pos[1] + STEP_LENGTH * math.sin(rad)
            nz = pos[2]
            # Enforce altitude floor
            if nz > MIN_ALTITUDE_Z:
                nz = MIN_ALTITUDE_Z
            self.drone.moveToPositionAsync(
                nx, ny, nz, 8,
                timeout_sec=timeout,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()
            
        elif action == 1:  # Up
            nz = pos[2] - VERT_STEP
            self.drone.moveToPositionAsync(
                pos[0], pos[1], nz, 4,
                timeout_sec=timeout,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()
            
        elif action == 2:  # Down
            nz = min(pos[2] + VERT_STEP, MIN_ALTITUDE_Z)  # can't go below 2m altitude
            self.drone.moveToPositionAsync(
                pos[0], pos[1], nz, 4,
                timeout_sec=timeout,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()
            
        elif action == 3:  # Yaw Left
            self.drone_yaw = (self.drone_yaw - YAW_STEP) % 360
            self.drone.rotateToYawAsync(
                self.drone_yaw, timeout_sec=timeout, margin=5,
                vehicle_name=VEHICLE
            ).join()
            
        elif action == 4:  # Yaw Right
            self.drone_yaw = (self.drone_yaw + YAW_STEP) % 360
            self.drone.rotateToYawAsync(
                self.drone_yaw, timeout_sec=timeout, margin=5,
                vehicle_name=VEHICLE
            ).join()
        
        # ═══ ALTITUDE SAFETY NET ═══
        # After every action, if drone is too low, push it back up
        new_pos = self._get_position()
        if new_pos[2] > MIN_ALTITUDE_Z:
            self.drone.moveToPositionAsync(
                new_pos[0], new_pos[1], SPAWN_Z, 5,
                timeout_sec=2,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()

    # ─── REWARD (with ground death + A* heading) ──────────────
    def _compute_reward(self, action):
        """Compute reward with ground death, heading bonus, exploration."""
        pos = self._get_position()
        dist = self._dist_to_target()
        sectors = self._get_lidar_sectors()
        target_found = self._target_in_fov()
        
        # ═══ THE SMART FILTER ═══
        # 1. Did the drone finally make it into the air? (Z < -2.5 means 2.5 meters UP)
        if pos[2] < -2.5:
            self._is_airborne = True
            
        # 2. Read raw collision sensors
        raw_collision = self.drone.simGetCollisionInfo(VEHICLE).has_collided
        raw_ground_contact = pos[2] > GROUND_DEATH_Z
        
        # 3. Probationary Period: Ignore all crashes until it successfully takes off
        if not self._is_airborne:
            collision = False
            ground_contact = False
        else:
            collision = raw_collision
            ground_contact = raw_ground_contact
        
        # Check bounds
        out_of_bounds = any([
            pos[0] < BOUNDS_MIN[0], pos[1] < BOUNDS_MIN[1], pos[2] < BOUNDS_MIN[2],
            pos[0] > BOUNDS_MAX[0], pos[1] > BOUNDS_MAX[1], pos[2] > BOUNDS_MAX[2],
        ])
        
        reward = -0.3  # per-step cost
        done = False
        truncated = False
        info = {"collision": collision, "target_found": target_found, "dist": dist,
                "altitude": -pos[2]}
        
        # ── Terminal conditions (priority order) ──
        if ground_contact:
            reward = -200.0
            done = True
            info["reason"] = "ground_death"
        elif collision:
            reward = -100.0
            done = True
            info["reason"] = "collision"
        elif target_found and dist < TARGET_FOUND_DIST:
            reward = 200.0
            done = True
            info["reason"] = "target_found"
        elif out_of_bounds:
            reward = -50.0
            done = True
            info["reason"] = "out_of_bounds"
        elif self.step_count >= MAX_STEPS:
            reward = -20.0
            truncated = True
            info["reason"] = "max_steps"
        else:
            # ── Distance shaping ──
            if self.prev_dist_to_target is not None:
                delta = self.prev_dist_to_target - dist
                reward += delta * 3.0   # positive if closer
                if delta > 0:
                    reward += 1.0       # approach bonus
            
            # ── A*-inspired HEADING reward (ONLY on Forward action) ──
            # Only reward facing target when drone MOVES forward
            # This prevents the spinning-in-circles exploit
            target_dir = self._get_target_direction()
            heading_rad = math.radians(self.drone_yaw)
            heading_vec = np.array([math.cos(heading_rad), math.sin(heading_rad)])
            target_vec_2d = target_dir[:2]
            target_norm = np.linalg.norm(target_vec_2d)
            if target_norm > 0.01:
                cos_angle = np.dot(heading_vec, target_vec_2d / target_norm)
                if action == 0:  # Forward — full heading reward
                    reward += 3.0 * cos_angle  # +3 moving toward target
                # Small yaw alignment bonus (much smaller than forward)
                elif action in (3, 4) and cos_angle > 0.7:
                    reward += 0.3  # tiny reward for aligning toward target
            
            # ── Anti-spin penalty ──
            # Penalize consecutive yaw-only actions (spinning)
            if self._consecutive_yaws >= 3:
                reward -= 2.0  # stop spinning!
            elif self._consecutive_yaws >= 6:
                reward -= 5.0  # heavy penalty for excessive spinning
            
            # ── Altitude maintenance ──
            if GOOD_ALT_MIN <= pos[2] <= GOOD_ALT_MAX:
                reward += 1.0  # good altitude bonus
            elif pos[2] > -1.5:
                reward -= 3.0  # dangerously low penalty
            
            # ── LiDAR proximity penalty ──
            if np.min(sectors) < LIDAR_DANGER:
                reward -= 5.0
            
            # ── Partial target detection ──
            if target_found:
                reward += 15.0
            
            # ── Exploration bonus ──
            cell = (int(pos[0] // self._cell_size), int(pos[1] // self._cell_size))
            if cell not in self._visited_cells:
                self._visited_cells.add(cell)
                reward += 0.5  # reward for exploring new area
        if done or truncated:
            self.last_reason = info["reason"]
        self.prev_dist_to_target = dist
        return reward, done, truncated, info

    # ─── STEP & RESET ─────────────────────────────────────────
    def step(self, action):
        self._do_action(action)
        self.step_count += 1
        
        # Track yaw spinning
        if action in (3, 4):  # Yaw Left / Yaw Right
            self._consecutive_yaws += 1
        else:
            self._consecutive_yaws = 0
        self._last_action = action
        
        obs = self._get_obs()
        reward, done, truncated, info = self._compute_reward(action)
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


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Testing RL environment (v3 — hover fix)...")
    env = AirSimDroneSearchEnv()
    obs, info = env.reset()
    
    print(f"Observation spaces:")
    print(f"  image:  {obs['image'].shape}")
    print(f"  lidar:  {obs['lidar']}")
    print(f"  target: {obs['target_dir']}")
    
    action_names = ["Forward", "Up", "Down", "Yaw Left", "Yaw Right"]
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: {action_names[action]:10s} reward={reward:.2f} done={done} alt={info.get('altitude','?')} info={info}")
        if done or truncated:
            print(f"  → Episode ended: {info.get('reason', '?')}")
            obs, info = env.reset()
    
    env.close()
    print("Test complete!")