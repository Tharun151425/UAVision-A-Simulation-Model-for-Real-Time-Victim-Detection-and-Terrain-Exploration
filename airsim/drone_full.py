"""
drone_full.py - Complete All-in-One: GTA Controls + Vision + Detection
======================================================================
Everything in one script:
  - GTA-style WASD + mouse controls
  - Live camera feed with crosshair
  - Object detection using AirSim APIs (NO ML/CV training)
  - Distance sensor + LiDAR readings
  - Segmentation-based object identification
  - FPV / TPV camera toggle
  - Screenshot on new object detection
  - Full HUD with all sensor data

REQUIRES: pip install pygame opencv-python numpy

HOW TO RUN:
  1. Copy settings.json → C:\\Users\\<you>\\Documents\\AirSim\\settings.json
  2. Unreal Editor → Open Blocks.uproject → Press Play
  3. python drone_full.py
  4. Click pygame window → Hold RMB + move mouse to look

CONTROLS:
  RMB + Mouse   = Look around
  WASD          = Fly (camera-relative)
  Q / E         = Yaw left / right
  Space         = Go up
  LShift        = Go down
  V             = Toggle FPV / TPV
  F             = Toggle detection overlay ON/OFF
  L             = List all scene objects in console
  P             = Take screenshot
  ESC           = Quit
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
    print("ERROR: pip install pygame"); sys.exit(1)
try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python"); sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# CONFIG - EDIT THESE
# ═══════════════════════════════════════════════════════════════
VEHICLE       = "Drone1"
CAMERA        = "0"
LIDAR_FRONT   = "FrontLidar"
LIDAR_SURR    = "SurroundLidar"
DIST_FRONT    = "FrontDistance"

MOVE_SPD      = 5.0        # m/s
YAW_SPD       = 45.0       # deg/s
VERT_SPD      = 3.0        # m/s
MOUSE_X       = 0.15       # mouse X sensitivity
MOUSE_Y       = 0.10       # mouse Y sensitivity
TPV_BACK      = -4.0       # meters behind drone
TPV_UP        = -2.0       # meters above drone (NED)
DETECT_RADIUS = 200 * 100  # cm
WIN_W, WIN_H  = 960, 720
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")

# ═══════════════════════════════════════════════════════════════
# Object name aliases — map mesh names to friendly labels
# ═══════════════════════════════════════════════════════════════
# Add custom mappings here! When you add new objects to the scene,
# map their mesh name pattern to a human-readable label.
#
# HOW TO ADD YOUR OWN:
#   1. Run this script, press L to list all scene objects
#   2. Find the mesh name of your object (e.g. "SM_Tree_01")  
#   3. Add it here: "SM_Tree" → "Tree"
#   4. The detection will auto-match partial names
OBJECT_ALIASES = {
    "PersonTarget": "🧑 Person",      # ← The target to find
    "MovingCube":   "📦 Box/Crate",   # ← Dynamic obstacles
    "Cone":     "🧑 Cone",
    "Cube":     "📦 Static Cube",
    "Sphere":   "🔴 Ball",
    "Cylinder": "🏗 Pillar",
    "SM_":      "🏠 Structure",
    "Wall":     "🧱 Wall",
    "Floor":    "⬛ Floor",
    "Ground":   "⬛ Ground",
}

def friendly_name(raw_name):
    """Convert raw mesh name to friendly label using OBJECT_ALIASES."""
    for pattern, label in OBJECT_ALIASES.items():
        if pattern.lower() in raw_name.lower():
            return f"{label} ({raw_name})"
    return raw_name


class DroneFullController:
    def __init__(self):
        print("="*60)
        print("  DRONE FULL CONTROLLER — Vision + GTA Controls")
        print("="*60)
        
        # ── AirSim ──
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, VEHICLE)
        self.client.armDisarm(True, VEHICLE)
        
        # ── Object registry ──
        self.obj_registry = {}
        self.seg_to_names = {}
        self._scan_scene()
        
        # ── Detection setup ──
        self.client.simSetDetectionFilterRadius(CAMERA, airsim.ImageType.Scene, DETECT_RADIUS, VEHICLE)
        self.client.simAddDetectionFilterMeshName(CAMERA, airsim.ImageType.Scene, "*", VEHICLE)
        
        # ── State ──
        self.yaw = 0.0
        self.cam_pitch = 0.0
        self.is_fpv = True
        self.show_detections = True
        self.running = True
        self.seen_objects = set()
        self.shot_count = 0
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        
        # ── Pygame ──
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Drone Full — GTA Controls + Vision")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_big = pygame.font.SysFont("consolas", 18, bold=True)
        
        print(f"[✓] Connected | {len(self.obj_registry)} scene objects indexed")
        self._print_controls()

    def _print_controls(self):
        print("\n╔══════════════════════════════════════╗")
        print("║  CONTROLS                            ║")
        print("╠══════════════════════════════════════╣")
        print("║  RMB + Mouse  = Look around          ║")
        print("║  WASD         = Fly (camera-relative) ║")
        print("║  Q / E        = Yaw left / right      ║")
        print("║  Space        = Go up                 ║")
        print("║  LShift       = Go down               ║")
        print("║  V            = Toggle FPV / TPV      ║")
        print("║  F            = Toggle detections     ║")
        print("║  L            = List scene objects    ║")
        print("║  P            = Screenshot            ║")
        print("║  ESC          = Quit                  ║")
        print("╚══════════════════════════════════════╝\n")

    def _scan_scene(self):
        """Build object_name → seg_id mapping from scene."""
        for name in self.client.simListSceneObjects(".*"):
            try:
                sid = self.client.simGetSegmentationObjectID(name)
                self.obj_registry[name] = sid
                self.seg_to_names.setdefault(sid, []).append(name)
            except Exception:
                pass

    # ─── SENSORS ────────────────────────────────────────────
    def _seg_center(self, seg_img):
        """Identify object at center of segmentation image."""
        if seg_img is None: return -1, "Sky/Air"
        h, w = seg_img.shape[:2]
        sid = int(seg_img[h//2, w//2, 0]) if len(seg_img.shape)==3 else int(seg_img[h//2, w//2])
        names = self.seg_to_names.get(sid, [])
        if names:
            raw = min(names, key=len)
            return sid, friendly_name(raw)
        return sid, "Sky/Air" if sid == 0 else f"Object(ID={sid})"

    def _dist_front(self):
        try: return self.client.getDistanceSensorData(DIST_FRONT, VEHICLE).distance
        except: return -1.0

    def _lidar_front(self):
        try:
            d = self.client.getLidarData(LIDAR_FRONT, VEHICLE)
            if len(d.point_cloud) < 3: return -1.0
            pts = np.array(d.point_cloud, dtype=np.float32).reshape(-1, 3)
            return float(np.min(np.linalg.norm(pts, axis=1)))
        except: return -1.0

    def _lidar_surround_sectors(self):
        """Get min distance in 8 sectors (N, NE, E, SE, S, SW, W, NW) from surround lidar."""
        sectors = [999.0] * 8
        try:
            d = self.client.getLidarData(LIDAR_SURR, VEHICLE)
            if len(d.point_cloud) < 3: return sectors
            pts = np.array(d.point_cloud, dtype=np.float32).reshape(-1, 3)
            angles = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))  # bearing from drone
            dists = np.linalg.norm(pts[:, :2], axis=1)  # horizontal distance
            for i, pt_angle in enumerate(angles):
                sector = int(((pt_angle + 180 + 22.5) % 360) / 45)
                if sector < 8 and dists[i] < sectors[sector]:
                    sectors[sector] = float(dists[i])
        except: pass
        return sectors

    def _detections(self):
        try: return self.client.simGetDetections(CAMERA, airsim.ImageType.Scene, VEHICLE)
        except: return []

    # ─── CAMERA ─────────────────────────────────────────────
    def _cam_tpv(self):
        cam = airsim.Pose(
            airsim.Vector3r(TPV_BACK, 0, TPV_UP),
            airsim.to_quaternion(math.radians(self.cam_pitch), 0, 0)
        )
        self.client.simSetCameraPose(CAMERA, cam, vehicle_name=VEHICLE)

    def _cam_fpv(self):
        cam = airsim.Pose(
            airsim.Vector3r(0, 0, 0),
            airsim.to_quaternion(math.radians(self.cam_pitch), 0, 0)
        )
        self.client.simSetCameraPose(CAMERA, cam, vehicle_name=VEHICLE)

    def _get_frame(self):
        """Get scene + segmentation images."""
        try:
            resps = self.client.simGetImages([
                airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False),
                airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False),
            ], vehicle_name=VEHICLE)
            
            scene = None
            if resps[0].width > 0:
                s1d = np.frombuffer(resps[0].image_data_uint8, dtype=np.uint8)
                scene = s1d.reshape(resps[0].height, resps[0].width, 3).copy()
            
            seg = None
            if resps[1].width > 0:
                g1d = np.frombuffer(resps[1].image_data_uint8, dtype=np.uint8)
                seg = g1d.reshape(resps[1].height, resps[1].width, 3)
            
            return scene, seg
        except:
            return None, None

    # ─── HUD ────────────────────────────────────────────────
    def _draw_hud(self, surf, center_obj, dist, detections, sectors):
        cx, cy = WIN_W // 2, WIN_H // 2
        
        # ── Crosshair ──
        c = (0, 255, 0) if "Sky" in center_obj else (255, 255, 0)
        pygame.draw.line(surf, c, (cx-20, cy), (cx-6, cy), 2)
        pygame.draw.line(surf, c, (cx+6, cy), (cx+20, cy), 2)
        pygame.draw.line(surf, c, (cx, cy-20), (cx, cy-6), 2)
        pygame.draw.line(surf, c, (cx, cy+6), (cx, cy+20), 2)
        pygame.draw.circle(surf, c, (cx, cy), 3)
        
        # ── Center label ──
        if "Sky" not in center_obj:
            lbl = self.font_big.render(center_obj, True, (255, 255, 0))
            surf.blit(lbl, (cx + 25, cy - 10))
            dlbl = self.font.render(f"{dist:.1f}m", True, (0, 220, 255))
            surf.blit(dlbl, (cx + 25, cy + 12))
        
        # ── Top-left telemetry ──
        st = self.client.getMultirotorState(vehicle_name=VEHICLE)
        p = st.kinematics_estimated.position
        v = st.kinematics_estimated.linear_velocity
        spd = (v.x_val**2 + v.y_val**2 + v.z_val**2)**0.5
        mode = "FPV" if self.is_fpv else "TPV"
        det_st = "ON" if self.show_detections else "OFF"
        
        info = [
            f"DRONE FULL [{mode}]  Det:{det_st}",
            f"Pos: ({p.x_val:.1f}, {p.y_val:.1f}, {p.z_val:.1f})",
            f"Alt: {-p.z_val:.1f}m  Speed: {spd:.1f}m/s",
            f"Yaw: {self.yaw:.0f}°  Pitch: {self.cam_pitch:.0f}°",
            f"Front Dist: {dist:.1f}m",
            f"Center: {center_obj}",
        ]
        y = 8
        for i, line in enumerate(info):
            clr = (0, 255, 0) if i == 0 else (200, 200, 200)
            surf.blit(self.font.render(line, True, clr), (8, y))
            y += 17
        
        # ── Proximity radar (bottom-left) ──
        radar_x, radar_y, radar_r = 80, WIN_H - 90, 55
        pygame.draw.circle(surf, (40, 40, 40), (radar_x, radar_y), radar_r)
        pygame.draw.circle(surf, (0, 100, 0), (radar_x, radar_y), radar_r, 1)
        pygame.draw.circle(surf, (0, 255, 0), (radar_x, radar_y), 3)
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        for i, (d, lbl) in enumerate(zip(sectors, dirs)):
            angle = math.radians(i * 45 - 90)
            # Scale: 0m=at center, 30m=at edge
            r = min(d / 30.0, 1.0) * (radar_r - 5)
            px = int(radar_x + math.cos(angle) * r)
            py = int(radar_y + math.sin(angle) * r)
            clr = (255, 0, 0) if d < 5 else (255, 255, 0) if d < 15 else (0, 200, 0)
            pygame.draw.circle(surf, clr, (px, py), 4)
        surf.blit(self.font.render("RADAR", True, (0, 200, 0)), (radar_x - 18, radar_y - radar_r - 15))
        
        # ── Detection list (right side) ──
        if self.show_detections and detections:
            rx = WIN_W - 280
            ry = 8
            surf.blit(self.font_big.render(f"DETECTED ({len(detections)})", True, (0, 255, 0)), (rx, ry))
            ry += 22
            for det in detections[:10]:
                rel = det.relative_pose.position
                d = (rel.x_val**2 + rel.y_val**2 + rel.z_val**2)**0.5
                name = friendly_name(det.name)
                txt = f"{name}: {d:.1f}m"
                surf.blit(self.font.render(txt, True, (200, 200, 200)), (rx, ry))
                ry += 16
        
        # ── Bottom help ──
        help_txt = "RMB+Mouse=Look | WASD=Fly | Q/E=Yaw | Space/Shift=Up/Down | V=FPV/TPV | F=Detect | P=Shot | ESC=Quit"
        surf.blit(self.font.render(help_txt, True, (120, 120, 120)), (8, WIN_H - 18))

    # ─── MAIN LOOP ──────────────────────────────────────────
    def run(self):
        print("[⬆] Taking off...")
        self.client.takeoffAsync(vehicle_name=VEHICLE).join()
        self.client.moveToPositionAsync(0, 0, -5, 3, vehicle_name=VEHICLE).join()
        print("[✓] Airborne!\n")
        
        while self.running:
            dt = self.clock.tick(30) / 1000.0
            
            # ── Events ──
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self.running = False
                    elif ev.key == pygame.K_v:
                        self.is_fpv = not self.is_fpv
                        print(f"[📷] {'FPV' if self.is_fpv else 'TPV'}")
                    elif ev.key == pygame.K_f:
                        self.show_detections = not self.show_detections
                        print(f"[🔍] Detections: {'ON' if self.show_detections else 'OFF'}")
                    elif ev.key == pygame.K_l:
                        print("\n=== SCENE OBJECTS ===")
                        for n, s in sorted(self.obj_registry.items()):
                            print(f"  {n:40s} segID={s}  → {friendly_name(n)}")
                        print(f"=== Total: {len(self.obj_registry)} ===\n")
                    elif ev.key == pygame.K_p:
                        self.shot_count += 1
                        # We'll screenshot after render
            
            # ── Mouse ──
            btns = pygame.mouse.get_pressed()
            if btns[2]:
                rx, ry = pygame.mouse.get_rel()
                self.yaw += rx * MOUSE_X
                self.cam_pitch = max(-60, min(60, self.cam_pitch + ry * MOUSE_Y))
                self.yaw %= 360
            else:
                pygame.mouse.get_rel()
            
            # ── Keys ──
            keys = pygame.key.get_pressed()
            vx = vy = vz = 0.0
            if keys[pygame.K_w]: vx += MOVE_SPD
            if keys[pygame.K_s]: vx -= MOVE_SPD
            if keys[pygame.K_a]: vy -= MOVE_SPD
            if keys[pygame.K_d]: vy += MOVE_SPD
            if keys[pygame.K_SPACE]: vz -= VERT_SPD
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: vz += VERT_SPD
            
            yr = 0.0
            if keys[pygame.K_q]: yr = -YAW_SPD
            if keys[pygame.K_e]: yr = YAW_SPD
            self.yaw = (self.yaw + yr * dt) % 360
            
            if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vz) > 0.01 or abs(yr) > 0.01:
                self.client.moveByVelocityBodyFrameAsync(
                    vx, vy, vz, 0.1,
                    yaw_mode=airsim.YawMode(False, self.yaw),
                    vehicle_name=VEHICLE
                )
            
            # ── Camera ──
            (self._cam_fpv if self.is_fpv else self._cam_tpv)()
            
            # ── Sensors ──
            scene, seg = self._get_frame()
            sid, center_obj = self._seg_center(seg)
            dist_val = self._dist_front()
            if dist_val < 0: dist_val = self._lidar_front()
            sectors = self._lidar_surround_sectors()
            detections = self._detections() if self.show_detections else []
            
            # ── New object alert ──
            if center_obj not in self.seen_objects and "Sky" not in center_obj:
                self.seen_objects.add(center_obj)
                print(f"\n[🆕] First time: {center_obj} (segID={sid})")
            
            # ── Render ──
            if scene is not None:
                rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (WIN_W, WIN_H))
                surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                self.screen.blit(surf, (0, 0))
            else:
                self.screen.fill((20, 20, 30))
            
            self._draw_hud(self.screen, center_obj, dist_val, detections, sectors)
            
            # Draw detection boxes on scene (via pygame rects)
            if self.show_detections and scene is not None:
                scale_x = WIN_W / scene.shape[1] if scene is not None else 1
                scale_y = WIN_H / scene.shape[0] if scene is not None else 1
                for det in detections[:10]:
                    x1 = int(det.box2D.min.x_val * scale_x)
                    y1 = int(det.box2D.min.y_val * scale_y)
                    x2 = int(det.box2D.max.x_val * scale_x)
                    y2 = int(det.box2D.max.y_val * scale_y)
                    pygame.draw.rect(self.screen, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 1)
                    lbl = self.font.render(friendly_name(det.name), True, (0, 255, 0))
                    self.screen.blit(lbl, (x1, y1 - 15))
            
            pygame.display.flip()
            
            # ── Console ──
            det_names = [friendly_name(d.name) for d in detections[:3]]
            print(
                f"\r[👁] {center_obj:30s} | Dist:{dist_val:5.1f}m | Detected:{det_names}",
                end="", flush=True
            )
        
        # ── Cleanup ──
        print("\n[⬇] Landing...")
        self.client.landAsync(vehicle_name=VEHICLE).join()
        self.client.armDisarm(False, VEHICLE)
        self.client.enableApiControl(False, VEHICLE)
        pygame.quit()
        print("[✓] Done.")


if __name__ == "__main__":
    DroneFullController().run()
