"""
rl_play_phase.py — Run Phase 1 / Phase 1.5 Trained Models in AirSim
=====================================================================
Loads your friend's PPO models (trained in PyBullet) and bridges them
to AirSim so you can watch the drone navigate in Unreal Engine.

The models use:
  - Observation: 262-dim vector (256 raycast + 5 ego + 1 timestep)
  - Actions: 6 discrete (Turn Left, Turn Right, Up, Down, Forward, Scan)

This script TRANSLATES between:
  - AirSim sensors → 262-dim observation (matching PyBullet format)
  - Model's action output → AirSim drone movement commands

HOW TO RUN:
  python rl_play_phase.py

  Then select:
    1 = Phase 1 (start → goal, no obstacles)
    2 = Phase 1.5 (start → goal, with obstacles)

  You'll be asked for the .zip model path.

REQUIRES:
  pip install stable-baselines3 gymnasium numpy
  (AirSim must be running with Unreal Editor → Press Play)
"""
import setup_path
import airsim
import numpy as np
import math
import time
import sys
import os

try:
    from stable_baselines3 import PPO
except ImportError:
    print("ERROR: pip install stable-baselines3")
    sys.exit(1)

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ═══════════════════════════════════════════════════════════════
# CONFIG — matches your friend's PyBullet environment
# ═══════════════════════════════════════════════════════════════
VEHICLE        = "Drone1"
CAMERA         = "0"
LIDAR_SURR     = "SurroundLidar"

# Match PyBullet env constants
WORLD_SIZE     = 50.0
RAY_RANGE      = 20.0
FORWARD_STEP   = 1.0       # meters per forward action
YAW_STEP       = 15.0      # degrees per turn
ALT_STEP       = 0.5       # meters per up/down
MIN_ALT        = 1.0       # meters
MAX_ALT        = 10.0      # meters
MAX_STEPS      = 800       # Phase 1 max steps
MAX_STEPS_1_5  = 1100      # Phase 1.5 Level 3 max steps
REACH_DIST     = 2.0       # meters to count as "found"

ACTION_NAMES = {
    0: "TURN_LEFT", 1: "TURN_RIGHT",
    2: "MOVE_UP",   3: "MOVE_DOWN",
    4: "FORWARD",   5: "SCAN"
}

# Detection setup
DETECT_RADIUS  = 200 * 100
TARGET_MESH    = "PersonTarget*"

# Pygame window
WIN_W, WIN_H = 800, 600


class AirSimPhaseBridge:
    """Bridges a PyBullet-trained PPO model to AirSim.
    
    Translates AirSim sensor data into the same 262-dim observation
    format used by the PyBullet training envs, then executes the model's
    chosen action as AirSim movement commands.
    """
    
    def __init__(self, phase, model_path, episodes=10):
        self.phase = phase
        self.episodes = episodes
        
        print("=" * 60)
        print(f"  PHASE {phase} MODEL → AirSim Bridge")
        print(f"  Running pre-trained PyBullet model in Unreal Engine!")
        print("=" * 60)
        
        # ── Load model ──
        print(f"\n[1/3] Loading PPO model: {model_path}")
        self.model = PPO.load(model_path)
        print("[✓] Model loaded!")
        
        # ── Connect to AirSim ──
        print("[2/3] Connecting to AirSim...")
        self.client = airsim.MultirotorClient(ip="127.0.0.1")
        self.client.confirmConnection()
        
        # ── Setup detection (for finding target position) ──
        self.client.simSetDetectionFilterRadius(
            CAMERA, airsim.ImageType.Scene, DETECT_RADIUS, VEHICLE)
        self.client.simAddDetectionFilterMeshName(
            CAMERA, airsim.ImageType.Scene, TARGET_MESH, VEHICLE)
        
        # ── Scan for targets in scene ──
        all_objs = self.client.simListSceneObjects(".*")
        self.target_names = [n for n in all_objs if n.lower().startswith("persontarget")]
        self.cube_names = [n for n in all_objs if n.lower().startswith("movingcube")]
        print(f"[✓] Found {len(self.target_names)} targets, {len(self.cube_names)} obstacles")
        
        # ── State ──
        self.drone_yaw = 0.0
        self.drone_alt = 3.0       # NED: this is the actual altitude in meters
        self.target_pos = None     # (x, y) of survivor
        self.step_count = 0
        self.max_steps = MAX_STEPS if phase == 1 else MAX_STEPS_1_5
        
        # ── Pygame ──
        self.use_pygame = _HAS_PYGAME
        if self.use_pygame:
            pygame.init()
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption(f"Phase {phase} Model — AirSim Bridge")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 14)
            self.font_big = pygame.font.SysFont("consolas", 20, bold=True)
        
        print(f"[3/3] Ready! Phase {phase}, {episodes} episodes\n")

    def _get_position(self):
        """Get drone position (x, y, z) in AirSim NED coordinates."""
        state = self.client.getMultirotorState(vehicle_name=VEHICLE)
        p = state.kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val])

    def _randomize_and_setup(self):
        """Randomize target position and setup drone for new episode."""
        rng = np.random
        half = (WORLD_SIZE / 2) - 1.0
        
        # Random drone position
        dx = rng.uniform(-half/2, half/2)
        dy = rng.uniform(-half/2, half/2)
        self.drone_yaw = float(rng.uniform(0, 360))
        self.drone_alt = 3.0
        
        # Random target at 15-40m
        prob = rng.rand()
        if prob < 0.20:
            target_dist = rng.uniform(15, 20)
        elif prob < 0.60:
            target_dist = rng.uniform(20, 30)
        else:
            target_dist = rng.uniform(30, 40)
        
        angle = rng.uniform(0, 2 * math.pi)
        tx = dx + target_dist * math.cos(angle)
        ty = dy + target_dist * math.sin(angle)
        tx = float(np.clip(tx, -half, half))
        ty = float(np.clip(ty, -half, half))
        self.target_pos = np.array([tx, ty])
        
        # Move target in AirSim
        if self.target_names:
            target_pose = airsim.Pose(
                airsim.Vector3r(tx, ty, 0),
                airsim.to_quaternion(0, 0, 0)
            )
            try:
                self.client.simSetObjectPose(self.target_names[0], target_pose, True)
            except:
                pass
        
        # Scatter cubes (obstacles) for phase 1.5
        if self.phase >= 2:  # 1.5
            for cube in self.cube_names:
                qx = float(rng.uniform(-half, half))
                qy = float(rng.uniform(-half, half))
                cube_pose = airsim.Pose(
                    airsim.Vector3r(qx, qy, 0),
                    airsim.to_quaternion(0, 0, rng.uniform(0, 2 * math.pi))
                )
                try:
                    self.client.simSetObjectPose(cube, cube_pose, True)
                except:
                    pass
        
        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True, VEHICLE)
        self.client.armDisarm(True, VEHICLE)
        self.client.takeoffAsync(vehicle_name=VEHICLE).join()
        time.sleep(0.3)
        
        # Teleport drone
        spawn_z = -self.drone_alt  # NED: negative = up
        start_pose = airsim.Pose(
            airsim.Vector3r(dx, dy, spawn_z),
            airsim.to_quaternion(0, 0, math.radians(self.drone_yaw))
        )
        self.client.simSetVehiclePose(start_pose, True, VEHICLE)
        self.client.moveToPositionAsync(
            dx, dy, spawn_z, 5,
            timeout_sec=3,
            yaw_mode=airsim.YawMode(False, self.drone_yaw),
            vehicle_name=VEHICLE
        ).join()
        time.sleep(0.3)
        
        self.step_count = 0
        actual_dist = math.hypot(dx - tx, dy - ty)
        print(f"  Spawned: drone=({dx:.0f},{dy:.0f}) yaw={self.drone_yaw:.0f}° | "
              f"target=({tx:.0f},{ty:.0f}) dist={actual_dist:.0f}m")

    def _build_observation(self):
        """Build the 262-dim observation vector matching PyBullet format.
        
        Format: [256 raycast values, 5 ego state, 1 timestep fraction]
        
        Since AirSim doesn't have the same raycast system, we use:
        - LiDAR data mapped to a 32x8 grid (approximate the PyBullet rays)
        - AirSim detection API to find the target
        """
        pos = self._get_position()
        drone_x, drone_y = pos[0], pos[1]
        drone_alt = -pos[2]  # Convert NED to altitude
        
        # ── Build raycast grid (8 vertical × 32 horizontal) ──
        # Map AirSim LiDAR data into the same format as PyBullet rays
        # Values: 0=air, 1=ground, 2=obstacle, 3=survivor
        ray_grid = np.zeros((8, 32), dtype=np.float32)
        
        try:
            data = self.client.getLidarData(LIDAR_SURR, VEHICLE)
            if len(data.point_cloud) >= 3:
                pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
                for pt in pts:
                    # Convert point to horizontal angle from drone heading
                    h_angle = math.degrees(math.atan2(pt[1], pt[0])) - self.drone_yaw
                    h_angle = ((h_angle + 180) % 360) - 180  # normalize to [-180, 180]
                    
                    # Only points in front FOV (120°)
                    if abs(h_angle) > 60:
                        continue
                    
                    # Map to grid indices
                    h_idx = int((h_angle + 60) / 120 * 32)
                    h_idx = max(0, min(31, h_idx))
                    
                    # Vertical angle
                    dist_2d = math.sqrt(pt[0]**2 + pt[1]**2)
                    v_angle = math.degrees(math.atan2(pt[2], max(dist_2d, 0.01)))
                    v_idx = int((v_angle + 30) / 60 * 8)
                    v_idx = max(0, min(7, v_idx))
                    
                    # Mark as obstacle (2)
                    ray_grid[v_idx, h_idx] = 2
        except:
            pass
        
        # ── Check if target is visible (mark as 3 in grid) ──
        try:
            dets = self.client.simGetDetections(
                CAMERA, airsim.ImageType.Scene, VEHICLE)
            if dets:
                for det in dets:
                    rel = det.relative_pose.position
                    h_angle = math.degrees(math.atan2(rel.y_val, rel.x_val))
                    if abs(h_angle) <= 60:
                        h_idx = int((h_angle + 60) / 120 * 32)
                        h_idx = max(0, min(31, h_idx))
                        # Mark center rows as survivor
                        for v in range(3, 6):
                            ray_grid[v, h_idx] = 3
        except:
            pass
        
        # ── Build ego state (5 values) ──
        ego = np.array([
            drone_x / (WORLD_SIZE / 2),      # normalized x
            drone_y / (WORLD_SIZE / 2),      # normalized y
            drone_alt / MAX_ALT,              # normalized altitude
            math.cos(math.radians(self.drone_yaw)),  # heading cos
            math.sin(math.radians(self.drone_yaw)),  # heading sin
        ], dtype=np.float32)
        
        # ── Timestep fraction (1 value) ──
        time_frac = np.array([self.step_count / self.max_steps], dtype=np.float32)
        
        # ── Concatenate: 256 + 5 + 1 = 262 ──
        obs = np.concatenate([ray_grid.flatten(), ego, time_frac])
        return obs

    def _execute_action(self, action):
        """Translate PyBullet action into AirSim drone command."""
        pos = self._get_position()
        timeout = 2.0
        
        if action == 0:  # TURN_LEFT
            self.drone_yaw = (self.drone_yaw - YAW_STEP) % 360
            self.client.rotateToYawAsync(
                self.drone_yaw, timeout_sec=timeout, margin=5,
                vehicle_name=VEHICLE
            ).join()
            
        elif action == 1:  # TURN_RIGHT
            self.drone_yaw = (self.drone_yaw + YAW_STEP) % 360
            self.client.rotateToYawAsync(
                self.drone_yaw, timeout_sec=timeout, margin=5,
                vehicle_name=VEHICLE
            ).join()
            
        elif action == 2:  # MOVE_UP
            self.drone_alt = min(self.drone_alt + ALT_STEP, MAX_ALT)
            nz = -self.drone_alt  # NED
            self.client.moveToPositionAsync(
                pos[0], pos[1], nz, 4,
                timeout_sec=timeout,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()
            
        elif action == 3:  # MOVE_DOWN
            self.drone_alt = max(self.drone_alt - ALT_STEP, MIN_ALT)
            nz = -self.drone_alt  # NED
            self.client.moveToPositionAsync(
                pos[0], pos[1], nz, 4,
                timeout_sec=timeout,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()
            
        elif action == 4:  # MOVE_FORWARD
            rad = math.radians(self.drone_yaw)
            nx = pos[0] + FORWARD_STEP * math.cos(rad)
            ny = pos[1] + FORWARD_STEP * math.sin(rad)
            nz = -self.drone_alt  # maintain altitude
            self.client.moveToPositionAsync(
                nx, ny, nz, 5,
                timeout_sec=timeout,
                yaw_mode=airsim.YawMode(False, self.drone_yaw),
                vehicle_name=VEHICLE
            ).join()
            
        elif action == 5:  # SCAN
            # No movement, just observe
            time.sleep(0.1)

    def _check_survivor_reached(self):
        """Check if drone is close enough to target."""
        pos = self._get_position()
        if self.target_pos is None:
            return False
        dx = pos[0] - self.target_pos[0]
        dy = pos[1] - self.target_pos[1]
        return math.hypot(dx, dy) <= REACH_DIST

    def _check_collision(self):
        return self.client.simGetCollisionInfo(VEHICLE).has_collided

    def _get_camera_frame(self):
        """Get camera image for display."""
        try:
            resp = self.client.simGetImage(CAMERA, airsim.ImageType.Scene, VEHICLE)
            if resp is None:
                return None
            img = np.frombuffer(resp, dtype=np.uint8)
            if _HAS_CV2:
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (WIN_W, WIN_H))
                return img
            return None
        except:
            return None

    def _draw_hud(self, ep, step, action, dist, found):
        """Draw HUD overlay."""
        if not self.use_pygame:
            return
        
        pos = self._get_position()
        lines = [
            f"PHASE {'1' if self.phase == 1 else '1.5'} MODEL — AirSim Bridge",
            f"Episode {ep}/{self.episodes}  Step {step}/{self.max_steps}",
            f"Action: {ACTION_NAMES.get(action, '?')}",
            f"Pos: ({pos[0]:.1f}, {pos[1]:.1f})  Alt: {-pos[2]:.1f}m",
            f"Yaw: {self.drone_yaw:.0f}°",
            f"Target dist: {dist:.1f}m",
        ]
        if found:
            lines.append("★ SURVIVOR FOUND! ★")
        
        # Background
        bg = pygame.Surface((280, len(lines) * 18 + 10))
        bg.set_alpha(170)
        bg.fill((0, 0, 0))
        self.screen.blit(bg, (5, 5))
        
        y = 10
        for i, line in enumerate(lines):
            if i == 0:
                clr = (0, 255, 200)
            elif "FOUND" in line:
                clr = (0, 255, 0)
            else:
                clr = (200, 200, 200)
            self.screen.blit(self.font.render(line, True, clr), (10, y))
            y += 18
        
        # Controls
        help_txt = "ESC=Quit | R=Reset"
        self.screen.blit(self.font.render(help_txt, True, (100, 100, 100)),
                        (8, WIN_H - 18))

    def play(self):
        """Run inference loop."""
        results = []
        
        for ep in range(1, self.episodes + 1):
            print(f"\n{'='*40}")
            print(f"  Episode {ep}/{self.episodes}")
            print(f"{'='*40}")
            
            self._randomize_and_setup()
            done = False
            step = 0
            found = False
            
            while not done and step < self.max_steps:
                # Build observation matching PyBullet format
                obs = self._build_observation()
                
                # Model prediction (deterministic=False matches training behavior)
                action, _ = self.model.predict(obs, deterministic=False)
                action = int(action)
                
                # Execute in AirSim
                self._execute_action(action)
                step += 1
                
                # Check terminal conditions
                pos = self._get_position()
                dist = math.hypot(pos[0] - self.target_pos[0],
                                  pos[1] - self.target_pos[1])
                
                if self._check_survivor_reached():
                    found = True
                    done = True
                    reason = "Survivor Found"
                elif self._check_collision():
                    done = True
                    reason = "Collision"
                elif -pos[2] < 0.3:  # altitude too low
                    done = True
                    reason = "Crashed (Altitude)"
                elif step >= self.max_steps:
                    done = True
                    reason = "Timeout"
                else:
                    reason = "Ongoing"
                
                # Console output
                if step % 10 == 0:
                    print(f"\r  Step {step:3d} | {ACTION_NAMES.get(action, '?'):12s} | "
                          f"Dist: {dist:.1f}m | Alt: {-pos[2]:.1f}m",
                          end="", flush=True)
                
                # Pygame render
                if self.use_pygame:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            return results
                        if ev.type == pygame.KEYDOWN:
                            if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                                try:
                                    from exit_screen import show_exit_screen
                                    if show_exit_screen(self.screen):
                                        return results
                                except ImportError:
                                    return results
                            if ev.key == pygame.K_r:
                                done = True
                                reason = "Manual Reset"
                    
                    img = self._get_camera_frame()
                    if img is not None:
                        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                        self.screen.blit(surf, (0, 0))
                    else:
                        self.screen.fill((20, 20, 30))
                    
                    self._draw_hud(ep, step, action, dist, found)
                    pygame.display.flip()
                    self.clock.tick(15)
            
            emoji = "🎯" if found else "💥" if "Crash" in reason or "Coll" in reason else "⏱"
            print(f"\n  {emoji} Episode {ep}: {reason} | Steps: {step} | Found: {found}")
            results.append({
                "episode": ep, "steps": step,
                "reason": reason, "found": found
            })
        
        # Summary
        print(f"\n{'='*60}")
        print(f"  RESULTS — Phase {'1' if self.phase == 1 else '1.5'}")
        print(f"{'='*60}")
        for r in results:
            e = "🎯" if r["found"] else "💥" if "Crash" in r["reason"] else "⏱"
            print(f"  {e} Ep {r['episode']}: {r['reason']:18s} | Steps: {r['steps']:3d}")
        
        successes = sum(1 for r in results if r["found"])
        print(f"\n  Success: {successes}/{len(results)} "
              f"({100*successes/max(len(results),1):.0f}%)")
        
        return results

    def close(self):
        try:
            self.client.armDisarm(False, VEHICLE)
            self.client.enableApiControl(False, VEHICLE)
        except:
            pass
        if self.use_pygame:
            pygame.quit()


# ═══════════════════════════════════════════════════════════════
# MAIN — Interactive menu
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  UAVision — Phase Model Runner")
    print("=" * 60)
    print()
    print("  Which phase model do you want to run?")
    print("    1 = Phase 1   (start → goal, NO obstacles)")
    print("    2 = Phase 1.5 (start → goal, WITH obstacles)")
    print()
    
    choice = input("  Enter 1 or 2: ").strip()
    
    if choice == "1":
        phase = 1
    elif choice == "2":
        phase = 2  # internally represents 1.5
    else:
        print("Invalid choice. Use 1 or 2.")
        sys.exit(1)
    
    # Ask for model path
    default_dirs = {
        1: os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))), "phase_1"),
        2: os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))), "phase_1_5"),
    }
    
    print(f"\n  Looking for .zip files in: {default_dirs.get(phase, '?')}")
    
    # Auto-find .zip files
    search_dir = default_dirs.get(phase, ".")
    zips = []
    if os.path.isdir(search_dir):
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".zip"):
                    zips.append(os.path.join(root, f))
    
    if zips:
        print(f"\n  Found {len(zips)} model(s):")
        for i, z in enumerate(zips):
            print(f"    [{i+1}] {z}")
        print(f"    [0] Enter custom path")
        pick = input(f"\n  Select model (1-{len(zips)}, or 0): ").strip()
        if pick == "0":
            model_path = input("  Enter full path to .zip: ").strip()
        else:
            try:
                model_path = zips[int(pick) - 1]
            except:
                print("Invalid selection.")
                sys.exit(1)
    else:
        print(f"\n  No .zip files found in {search_dir}")
        model_path = input("  Enter full path to .zip model: ").strip()
    
    if not os.path.exists(model_path):
        print(f"ERROR: File not found: {model_path}")
        sys.exit(1)
    
    eps = input("\n  How many episodes? (default 10): ").strip()
    episodes = int(eps) if eps else 10
    
    player = AirSimPhaseBridge(
        phase=phase,
        model_path=model_path,
        episodes=episodes,
    )
    
    try:
        player.play()
    finally:
        player.close()
    
    print("\n[✓] Done.")
