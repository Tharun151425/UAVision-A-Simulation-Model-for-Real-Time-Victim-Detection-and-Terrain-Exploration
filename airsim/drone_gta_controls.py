"""
drone_gta_controls.py - GTA 5 Style Drone Flight Controller
============================================================
Fly the drone with GTA/Roblox-style controls:
  Mouse + Right-Click Hold → Camera look (yaw + pitch)
  WASD                     → Move relative to camera direction
  Q/E                      → Yaw left/right (rotate drone)
  Space                    → Go up
  LShift                   → Go down
  V                        → Toggle FPV / TPV (third person)
  ESC                      → Quit

REQUIRES: pip install pygame opencv-python numpy

SETTINGS REQUIRED: 
  Copy settings.json to C:\\Users\\<you>\\Documents\\AirSim\\settings.json

HOW TO RUN:
  1. Unreal Editor → Open Blocks.uproject → Press Play
  2. python drone_gta_controls.py
  3. Click the pygame window to give it focus
  4. Hold right-mouse-button and move mouse to look around
  5. WASD to fly, Space/Shift for up/down
"""
import setup_path
import airsim
import numpy as np
import math
import time
import sys
import os

try:
    import pygame
except ImportError:
    print("ERROR: pygame required. Run: pip install pygame")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python required. Run: pip install opencv-python")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
VEHICLE_NAME    = "Drone1"
CAMERA_NAME     = "0"

MOVE_SPEED      = 5.0       # m/s movement speed
YAW_SPEED       = 45.0      # degrees/sec for Q/E
MOUSE_SENS_X    = 0.15      # mouse horizontal sensitivity
MOUSE_SENS_Y    = 0.10      # mouse vertical sensitivity
VERT_SPEED      = 3.0       # m/s up/down speed

# TPV camera offset (behind and above drone, in drone body frame)
TPV_OFFSET_BACK = -4.0      # meters behind (negative X = behind)
TPV_OFFSET_UP   = -2.0      # meters above (negative Z = up in NED)

WINDOW_W        = 800
WINDOW_H        = 600


class GTADroneController:
    def __init__(self):
        print("="*60)
        print("  GTA-STYLE DRONE CONTROLLER")
        print("="*60)
        
        # AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, VEHICLE_NAME)
        self.client.armDisarm(True, VEHICLE_NAME)
        
        # State
        self.yaw = 0.0          # current yaw in degrees
        self.cam_pitch = 0.0    # camera pitch offset (for mouse look)
        self.is_fpv = True      # True=FPV, False=TPV
        self.running = True
        
        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("GTA Drone Controller - Hold RMB to look")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_big = pygame.font.SysFont("consolas", 22, bold=True)
        
        print("[✓] Connected to AirSim")
        print("[✓] Pygame window created")
        print("\nControls:")
        print("  RMB + Mouse = Look around")
        print("  WASD        = Fly (camera-relative)")
        print("  Q/E         = Yaw left/right")
        print("  Space/Shift = Up/Down")
        print("  V           = Toggle FPV/TPV")
        print("  ESC         = Quit\n")

    def _get_camera_image(self):
        """Get scene image from the drone camera."""
        try:
            response = self.client.simGetImage(CAMERA_NAME, airsim.ImageType.Scene, VEHICLE_NAME)
            if response is None:
                return None
            img_1d = np.frombuffer(response, dtype=np.uint8)
            img = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (WINDOW_W, WINDOW_H))
            return img
        except Exception:
            return None

    def _update_tpv_camera(self):
        """Position camera behind the drone for third-person view."""
        # Camera offset in body frame
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.cam_pitch)
        
        # Position camera behind + above, looking at drone
        cam_pose = airsim.Pose(
            airsim.Vector3r(TPV_OFFSET_BACK, 0, TPV_OFFSET_UP),
            airsim.to_quaternion(pitch_rad, 0, 0)  # pitch down to look at drone
        )
        self.client.simSetCameraPose(CAMERA_NAME, cam_pose, vehicle_name=VEHICLE_NAME)

    def _reset_fpv_camera(self):
        """Reset camera to forward-facing FPV."""
        pitch_rad = math.radians(self.cam_pitch)
        cam_pose = airsim.Pose(
            airsim.Vector3r(0, 0, 0),
            airsim.to_quaternion(pitch_rad, 0, 0)
        )
        self.client.simSetCameraPose(CAMERA_NAME, cam_pose, vehicle_name=VEHICLE_NAME)

    def _draw_hud(self, surface):
        """Draw HUD overlay on pygame surface."""
        # ── Crosshair ──
        cx, cy = WINDOW_W // 2, WINDOW_H // 2
        color = (0, 255, 0)
        pygame.draw.line(surface, color, (cx-15, cy), (cx-5, cy), 2)
        pygame.draw.line(surface, color, (cx+5, cy), (cx+15, cy), 2)
        pygame.draw.line(surface, color, (cx, cy-15), (cx, cy-5), 2)
        pygame.draw.line(surface, color, (cx, cy+5), (cx, cy+15), 2)
        pygame.draw.circle(surface, color, (cx, cy), 2)
        
        # ── Top-left info ──
        mode = "FPV" if self.is_fpv else "TPV"
        state = self.client.getMultirotorState(vehicle_name=VEHICLE_NAME)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        speed = (vel.x_val**2 + vel.y_val**2 + vel.z_val**2)**0.5
        
        lines = [
            f"GTA DRONE [{mode}]",
            f"Pos: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})",
            f"Yaw: {self.yaw:.0f}°  Pitch: {self.cam_pitch:.0f}°",
            f"Speed: {speed:.1f} m/s",
            f"Alt: {-pos.z_val:.1f}m",
        ]
        
        y = 10
        for i, line in enumerate(lines):
            color = (0, 255, 0) if i == 0 else (200, 200, 200)
            text_surface = self.font.render(line, True, color)
            surface.blit(text_surface, (10, y))
            y += 20
        
        # ── Bottom controls help ──
        help_text = "RMB+Mouse=Look | WASD=Fly | Q/E=Yaw | Space/Shift=Up/Down | V=FPV/TPV | ESC=Quit"
        text_surface = self.font.render(help_text, True, (150, 150, 150))
        surface.blit(text_surface, (10, WINDOW_H - 25))

    def run(self):
        """Main game loop."""
        print("[⬆] Taking off...")
        self.client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
        self.client.moveToPositionAsync(0, 0, -5, 3, vehicle_name=VEHICLE_NAME).join()
        print("[✓] Airborne!\n")
        
        while self.running:
            dt = self.clock.tick(30) / 1000.0  # seconds per frame
            
            # ── Events ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_v:
                        self.is_fpv = not self.is_fpv
                        print(f"[📷] Camera: {'FPV' if self.is_fpv else 'TPV'}")
                        if self.is_fpv:
                            self._reset_fpv_camera()
                        else:
                            self._update_tpv_camera()
            
            # ── Mouse look (right button held) ──
            mouse_buttons = pygame.mouse.get_pressed()
            if mouse_buttons[2]:  # Right mouse button
                rel_x, rel_y = pygame.mouse.get_rel()
                self.yaw += rel_x * MOUSE_SENS_X
                self.cam_pitch += rel_y * MOUSE_SENS_Y
                self.cam_pitch = max(-60, min(60, self.cam_pitch))
                # Normalize yaw
                self.yaw = self.yaw % 360
            else:
                pygame.mouse.get_rel()  # consume delta to prevent jumps
            
            # ── Keyboard movement ──
            keys = pygame.key.get_pressed()
            
            # Movement in camera-relative frame
            vx, vy, vz = 0.0, 0.0, 0.0
            
            if keys[pygame.K_w]:
                vx += MOVE_SPEED
            if keys[pygame.K_s]:
                vx -= MOVE_SPEED
            if keys[pygame.K_a]:
                vy -= MOVE_SPEED
            if keys[pygame.K_d]:
                vy += MOVE_SPEED
            if keys[pygame.K_SPACE]:
                vz -= VERT_SPEED   # NED: negative Z = up
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                vz += VERT_SPEED   # NED: positive Z = down
            
            # Q/E for yaw
            yaw_rate = 0.0
            if keys[pygame.K_q]:
                yaw_rate = -YAW_SPEED
            if keys[pygame.K_e]:
                yaw_rate = YAW_SPEED
            
            # Apply yaw from Q/E
            self.yaw += yaw_rate * dt
            self.yaw = self.yaw % 360
            
            # ── Send movement command ──
            # Body frame velocity: vx=forward, vy=right, vz=down
            if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vz) > 0.01 or abs(yaw_rate) > 0.01:
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=self.yaw)
                self.client.moveByVelocityBodyFrameAsync(
                    vx, vy, vz, 0.1,
                    yaw_mode=yaw_mode,
                    vehicle_name=VEHICLE_NAME
                )
            
            # ── Update camera ──
            if self.is_fpv:
                self._reset_fpv_camera()
            else:
                self._update_tpv_camera()
            
            # ── Render ──
            img = self._get_camera_image()
            if img is not None:
                surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                self.screen.blit(surf, (0, 0))
            else:
                self.screen.fill((30, 30, 30))
            
            self._draw_hud(self.screen)
            pygame.display.flip()
        
        # ── Cleanup ──
        print("\n[⬇] Landing...")
        self.client.landAsync(vehicle_name=VEHICLE_NAME).join()
        self.client.armDisarm(False, VEHICLE_NAME)
        self.client.enableApiControl(False, VEHICLE_NAME)
        pygame.quit()
        print("[✓] GTA Controller stopped.")


if __name__ == "__main__":
    controller = GTADroneController()
    controller.run()
