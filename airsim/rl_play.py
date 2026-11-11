"""
rl_play.py - Run Trained RL Model with MINIMAP (Inference / Demo) v3
==================================================================
Loads a trained DQN model and runs it in the AirSim environment.
Updated to sync perfectly with the v3 5-action space and VecTransposeImage.

CONTROLS:
  ESC / Q  = Stop
  R        = Reset episode
  M        = Toggle minimap ON/OFF
"""
import setup_path
import airsim
import numpy as np
import time
import sys
import os
import argparse
import math

try:
    from stable_baselines3 import DQN
except ImportError:
    print("ERROR: pip install stable-baselines3")
    sys.exit(1)

try:
    import pygame
except ImportError:
    pygame = None

try:
    import cv2
except ImportError:
    cv2 = None

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
VEHICLE            = "Drone1"
CAMERA             = "0"
# V3 ACTION SPACE: 0=Forward, 1=Up, 2=Down, 3=Yaw Left, 4=Yaw Right
ACTION_NAMES       = ["Forward", "Up", "Down", "Yaw Left", "Yaw Right"]
WIN_W, WIN_H       = 800, 600
MINIMAP_SIZE       = 220
MINIMAP_SCALE      = 1.5
MINIMAP_PADDING    = 10


# ═══════════════════════════════════════════════════════════════
# MINIMAP: Occupancy Grid Mapper
# ═══════════════════════════════════════════════════════════════
class OccupancyMinimap:
    def __init__(self, size=MINIMAP_SIZE, scale=MINIMAP_SCALE):
        self.size = size
        self.scale = scale
        self.grid = np.zeros((size, size, 3), dtype=np.uint8)
        self.grid[:] = [15, 15, 20]
        self.trail = []
        self.target_pos = None
        self.visible = True
        self.center_offset = np.array([0.0, 0.0])
    
    def world_to_grid(self, wx, wy):
        gx = int(self.size // 2 + (wx - self.center_offset[0]) / self.scale)
        gy = int(self.size // 2 + (wy - self.center_offset[1]) / self.scale)
        return gx, gy
    
    def _in_bounds(self, gx, gy):
        return 0 <= gx < self.size and 0 <= gy < self.size
    
    def update(self, drone_pos, drone_yaw, lidar_sectors, detected_target_pos=None):
        dx, dy = drone_pos[0], drone_pos[1]
        self.center_offset[0] = dx
        self.center_offset[1] = dy
        
        self.grid[:] = [15, 15, 20]
        self.trail.append((dx, dy))
        if len(self.trail) > 2000:
            self.trail = self.trail[-2000:]
        
        prev_px = None
        for tx, ty in self.trail:
            gx, gy = self.world_to_grid(tx, ty)
            if self._in_bounds(gx, gy):
                if prev_px and self._in_bounds(*prev_px):
                    self._draw_line(prev_px[0], prev_px[1], gx, gy, [0, 180, 200])
                prev_px = (gx, gy)
        
        for i, dist in enumerate(lidar_sectors):
            beam_angle_deg = i * 45 + drone_yaw
            beam_angle = math.radians(beam_angle_deg)
            steps = int(min(dist, 30.0) / self.scale)
            for s in range(steps):
                bx = dx + s * self.scale * math.cos(beam_angle)
                by = dy + s * self.scale * math.sin(beam_angle)
                gx, gy = self.world_to_grid(bx, by)
                if self._in_bounds(gx, gy):
                    curr = self.grid[gy, gx]
                    self.grid[gy, gx] = np.clip(curr + [5, 5, 3], 0, [40, 40, 35]).astype(np.uint8)
            
            if dist < 29.0:
                ex = dx + dist * math.cos(beam_angle)
                ey = dy + dist * math.sin(beam_angle)
                gx, gy = self.world_to_grid(ex, ey)
                if self._in_bounds(gx, gy):
                    for ox in range(-1, 2):
                        for oy in range(-1, 2):
                            if self._in_bounds(gx+ox, gy+oy):
                                self.grid[gy+oy, gx+ox] = [60, 20, 200]
        
        if detected_target_pos is not None:
            self.target_pos = detected_target_pos
        
        if self.target_pos is not None:
            gx, gy = self.world_to_grid(self.target_pos[0], self.target_pos[1])
            if self._in_bounds(gx, gy):
                pulse = int(180 + 75 * math.sin(time.time() * 4))
                for ox in range(-3, 4):
                    for oy in range(-3, 4):
                        if self._in_bounds(gx+ox, gy+oy):
                            self.grid[gy+oy, gx+ox] = [0, pulse, 0]
        
        cx, cy = self.size // 2, self.size // 2
        for ox in range(-2, 3):
            for oy in range(-2, 3):
                if self._in_bounds(cx+ox, cy+oy):
                    self.grid[cy+oy, cx+ox] = [255, 255, 0]
        
        yaw_rad = math.radians(drone_yaw)
        for d in range(3, 12):
            ax = int(cx + d * math.cos(yaw_rad))
            ay = int(cy + d * math.sin(yaw_rad))
            if self._in_bounds(ax, ay):
                self.grid[ay, ax] = [255, 200, 0]
    
    def _draw_line(self, x0, y0, x1, y1, color):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        steps = 0
        while steps < 500:
            if self._in_bounds(x0, y0):
                self.grid[y0, x0] = np.clip(self.grid[y0, x0].astype(int) + [0, 30, 40], 0, 255).astype(np.uint8)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            steps += 1
    
    def get_surface(self):
        return self.grid
    
    def reset(self):
        self.grid[:] = [15, 15, 20]
        self.trail = []
        self.target_pos = None


# ═══════════════════════════════════════════════════════════════
# DEMO PLAYER
# ═══════════════════════════════════════════════════════════════
class RLDemoPlayer:
    def __init__(self, model_path, render=True, episodes=5):
        print("=" * 60)
        print("  RL DEMO — Autonomous Target Search v3")
        print("=" * 60)
        
        self.episodes = episodes
        self.render = render and pygame is not None
        
        print(f"[1/3] Loading model: {model_path}")
        self.model = DQN.load(model_path)
        print("[✓] Model loaded!")
        
        print("[2/3] Creating environment...")
        from rl_drone_env import AirSimDroneSearchEnv, TARGET_LABEL
        self.env = AirSimDroneSearchEnv()
        self.target_label = TARGET_LABEL
        self.client = self.env.drone
        
        self.minimap = OccupancyMinimap()
        
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption(f"RL Demo — Finding {self.target_label}")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 14)
            self.font_mini = pygame.font.SysFont("consolas", 11)
        
        print(f"[3/3] Ready! Target: {self.target_label}")

    def _get_camera_frame(self):
        try:
            resp = self.client.simGetImage(CAMERA, airsim.ImageType.Scene, VEHICLE)
            if resp is None: return None
            img = np.frombuffer(resp, dtype=np.uint8)
            if cv2 is not None:
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (WIN_W, WIN_H))
                return img
            return None
        except:
            return None

    def _draw_hud(self, episode, step, action, reward, total_reward, info, obs):
        if not self.render: return
        cx, cy = WIN_W // 2, WIN_H // 2
        
        c = (0, 255, 0)
        pygame.draw.line(self.screen, c, (cx-15, cy), (cx-5, cy), 2)
        pygame.draw.line(self.screen, c, (cx+5, cy), (cx+15, cy), 2)
        pygame.draw.line(self.screen, c, (cx, cy-15), (cx, cy-5), 2)
        pygame.draw.line(self.screen, c, (cx, cy+5), (cx, cy+15), 2)
        
        lines = [
            f"RL DEMO — Episode {episode}/{self.episodes}",
            f"Step: {step}  Action: {ACTION_NAMES[action]}",
            f"Reward: {reward:+.1f}  Total: {total_reward:+.1f}",
            f"Target: {self.target_label}",
            f"LiDAR min: {np.min(obs['lidar']):.1f}m",
        ]
        
        if "reason" in info:
            lines.append(f"DONE: {info['reason'].upper()}")
        
        bg = pygame.Surface((250, len(lines) * 18 + 10))
        bg.set_alpha(150)
        bg.fill((0, 0, 0))
        self.screen.blit(bg, (5, 5))
        
        y = 10
        for i, line in enumerate(lines):
            clr = (0, 255, 0) if i == 0 else (255, 255, 0) if "DONE" in line else (200, 200, 200)
            self.screen.blit(self.font.render(line, True, clr), (10, y))
            y += 18
        
        rx, ry, rr = WIN_W - 70, 70, 50
        radar_bg = pygame.Surface((rr*2 + 10, rr*2 + 10), pygame.SRCALPHA)
        pygame.draw.circle(radar_bg, (0, 0, 0, 150), (rr+5, rr+5), rr+5)
        self.screen.blit(radar_bg, (rx - rr - 5, ry - rr - 5))
        pygame.draw.circle(self.screen, (0, 80, 0), (rx, ry), rr, 1)
        pygame.draw.circle(self.screen, (0, 255, 0), (rx, ry), 2)
        
        for i, d in enumerate(obs['lidar']):
            angle = np.radians(i * 45 - 90)
            r = min(d / 30.0, 1.0) * (rr - 5)
            px = int(rx + np.cos(angle) * r)
            py = int(ry + np.sin(angle) * r)
            clr = (255, 0, 0) if d < 5 else (255, 255, 0) if d < 15 else (0, 200, 0)
            pygame.draw.circle(self.screen, clr, (px, py), 3)
        
        td = obs['target_dir']
        ax = int(cx + td[1] * 30)
        ay = int(cy - td[0] * 30)
        pygame.draw.line(self.screen, (255, 0, 255), (cx, cy), (ax, ay), 2)
        pygame.draw.circle(self.screen, (255, 0, 255), (ax, ay), 4)

    def _draw_minimap(self):
        if not self.minimap.visible or not self.render: return
        mx, my = MINIMAP_PADDING, WIN_H - MINIMAP_SIZE - MINIMAP_PADDING
        pygame.draw.rect(self.screen, (0, 0, 0), (mx-2, my-2, MINIMAP_SIZE+4, MINIMAP_SIZE+4))
        pygame.draw.rect(self.screen, (0, 200, 200), (mx-2, my-2, MINIMAP_SIZE+4, MINIMAP_SIZE+4), 1)
        surf = pygame.surfarray.make_surface(self.minimap.get_surface().swapaxes(0, 1))
        self.screen.blit(surf, (mx, my))

    def play(self):
        results = []
        for ep in range(1, self.episodes + 1):
            print(f"\n{'='*40}\n  Episode {ep}/{self.episodes}\n{'='*40}")
            obs, info = self.env.reset()
            self.minimap.reset()
            done = False
            truncated = False
            total_reward = 0
            step = 0
            
            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
                
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                step += 1
                
                drone_pos = self.env._get_position()
                drone_yaw = self.env.drone_yaw
                lidar = obs['lidar']
                
                detected_pos = None
                try:
                    dets = self.client.simGetDetections(CAMERA, airsim.ImageType.Scene, VEHICLE)
                    if dets and hasattr(self.env, 'target_pos'):
                        detected_pos = self.env.target_pos
                except: pass
                
                self.minimap.update(drone_pos, drone_yaw, lidar, detected_pos)
                
                print(f"\r  Step {step:3d} | {ACTION_NAMES[action]:10s} | R: {reward:+6.1f} | Total: {total_reward:+7.1f} | LiDAR min: {np.min(obs['lidar']):5.1f}m", end="", flush=True)
                
                if self.render:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                            return results
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r: done = True
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_m: self.minimap.visible = not self.minimap.visible
                    
                    img = self._get_camera_frame()
                    if img is not None:
                        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                        self.screen.blit(surf, (0, 0))
                    else:
                        self.screen.fill((20, 20, 30))
                    
                    self._draw_hud(ep, step, action, reward, total_reward, info, obs)
                    self._draw_minimap()
                    pygame.display.flip()
                    self.clock.tick(10)
            
            reason = info.get("reason", "unknown")
            print(f"\n  → Episode {ep} ended: {reason} | Steps: {step} | Reward: {total_reward:+.1f}")
            results.append({"episode": ep, "steps": step, "reward": total_reward, "reason": reason})
        return results

    def close(self):
        self.env.close()
        if self.render: pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    
    base = os.path.dirname(__file__)
    candidates = [
        args.model if args.model else "",
        os.path.join(base, "rl_best", "best_model.zip"),
        os.path.join(base, "rl_models", "dqn_drone_final.zip")
    ]
    
    model_path = next((p for p in candidates if p and os.path.exists(p)), None)
    
    if model_path is None:
        print("ERROR: No trained model found!")
        sys.exit(1)
        
    player = RLDemoPlayer(model_path=model_path, render=not args.no_render, episodes=args.episodes)
    try:
        player.play()
    finally:
        player.close()