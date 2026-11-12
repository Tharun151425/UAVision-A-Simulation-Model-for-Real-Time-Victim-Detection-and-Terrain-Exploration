"""
drone_vision.py - Crosshair Object Detection & Distance System
==============================================================
Shows a live camera feed with a crosshair. Detects what object is 
in front of the drone and how far away it is.

Uses AirSim's built-in APIs (NO OpenCV ML, NO neural networks):
  - simListSceneObjects()     → list all objects in scene
  - simGetDetections()        → detect objects by mesh name
  - getDistanceSensorData()   → forward distance sensor
  - getLidarData()            → front-facing lidar
  - simGetSegmentationObjectID() → get seg ID for color mapping
  - Segmentation image        → identify center-pixel object

SETTINGS REQUIRED (settings.json):
  Copy the settings.json that came with this project to:
    C:\\Users\\<you>\\Documents\\AirSim\\settings.json

HOW TO RUN:
  1. Open Blocks.uproject in Unreal Editor → Press Play
  2. python drone_vision.py
  3. Press 'q' to quit

CONTROLS:
  q - Quit
  s - Take screenshot of current view
  l - List all scene objects in console
"""
import setup_path
import airsim
import numpy as np
import time
import os
import sys

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python required. Run: pip install opencv-python")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# CONFIG - Change these to customize
# ═══════════════════════════════════════════════════════════════
VEHICLE_NAME   = "Drone1"
CAMERA_NAME    = "0"
LIDAR_NAME     = "FrontLidar"
DISTANCE_NAME  = "FrontDistance"
DETECT_RADIUS  = 200 * 100            # detection radius in cm (200m)
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")

# ═══════════════════════════════════════════════════════════════
# Segmentation color → ID mapping (AirSim default palette)
# See: https://microsoft.github.io/AirSim/seg_rgbs.txt
# ═══════════════════════════════════════════════════════════════
SEG_COLORS = {}  # populated at runtime from scene objects


class DroneVision:
    def __init__(self):
        print("="*60)
        print("  DRONE VISION - Crosshair Object Detection")
        print("="*60)
        
        # Connect
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, VEHICLE_NAME)
        self.client.armDisarm(True, VEHICLE_NAME)
        
        # Build object registry: name → segmentation ID
        self.object_registry = {}
        self.seg_id_to_names = {}
        self._build_object_registry()
        
        # Setup detection for ALL objects (wildcard)
        self.client.simSetDetectionFilterRadius(
            CAMERA_NAME, airsim.ImageType.Scene, DETECT_RADIUS, VEHICLE_NAME
        )
        self.client.simAddDetectionFilterMeshName(
            CAMERA_NAME, airsim.ImageType.Scene, "*", VEHICLE_NAME
        )
        
        # Screenshot dir
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        self.seen_objects = set()
        self.screenshot_count = 0
        
        print("\n[✓] Connected & armed")
        print("[✓] Object registry built: %d objects" % len(self.object_registry))
        print("[✓] Detection filter set for ALL meshes")
        print("\nControls: q=quit, s=screenshot, l=list objects\n")

    def _build_object_registry(self):
        """Scan all scene objects and map name → segmentation ID."""
        objects = self.client.simListSceneObjects(".*")
        for obj_name in objects:
            try:
                seg_id = self.client.simGetSegmentationObjectID(obj_name)
                self.object_registry[obj_name] = seg_id
                if seg_id not in self.seg_id_to_names:
                    self.seg_id_to_names[seg_id] = []
                self.seg_id_to_names[seg_id].append(obj_name)
            except Exception:
                pass

    def _get_center_object_from_segmentation(self, seg_image):
        """Get the segmentation ID at the center pixel of the seg image."""
        if seg_image is None or seg_image.size == 0:
            return -1, "Unknown"
        h, w = seg_image.shape[:2]
        cx, cy = w // 2, h // 2
        # Segmentation image: each pixel's R channel = segmentation ID
        if len(seg_image.shape) == 3:
            seg_id = int(seg_image[cy, cx, 0])
        else:
            seg_id = int(seg_image[cy, cx])
        
        # Look up name
        names = self.seg_id_to_names.get(seg_id, [])
        if names:
            # Return the most specific (shortest) name
            name = min(names, key=len)
        elif seg_id == 0:
            name = "Sky/Air"
        else:
            name = f"Object(ID={seg_id})"
        return seg_id, name

    def _get_front_distance(self):
        """Get distance from front-facing distance sensor."""
        try:
            data = self.client.getDistanceSensorData(DISTANCE_NAME, VEHICLE_NAME)
            return data.distance
        except Exception:
            return -1.0

    def _get_front_lidar_distance(self):
        """Get closest point from front-facing lidar."""
        try:
            data = self.client.getLidarData(LIDAR_NAME, VEHICLE_NAME)
            if len(data.point_cloud) < 3:
                return -1.0, None
            points = np.array(data.point_cloud, dtype=np.float32)
            points = points.reshape(-1, 3)
            distances = np.linalg.norm(points, axis=1)
            min_idx = np.argmin(distances)
            return float(distances[min_idx]), points[min_idx]
        except Exception:
            return -1.0, None

    def _get_detections(self):
        """Get all detected objects from the camera."""
        try:
            return self.client.simGetDetections(
                CAMERA_NAME, airsim.ImageType.Scene, VEHICLE_NAME
            )
        except Exception:
            return []

    def _get_depth_at_center(self, depth_image):
        """Get depth value at center pixel."""
        if depth_image is None or depth_image.size == 0:
            return -1.0
        h, w = depth_image.shape[:2]
        return float(depth_image[h // 2, w // 2])

    def _draw_crosshair(self, img, obj_name, distance, detections):
        """Draw crosshair + HUD on image."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        
        # ── Crosshair lines ──
        color = (0, 255, 0) if obj_name == "Sky/Air" else (0, 255, 255)
        cv2.line(img, (cx - 20, cy), (cx - 8, cy), color, 2)
        cv2.line(img, (cx + 8, cy), (cx + 20, cy), color, 2)
        cv2.line(img, (cx, cy - 20), (cx, cy - 8), color, 2)
        cv2.line(img, (cx, cy + 8), (cx, cy + 20), color, 2)
        cv2.circle(img, (cx, cy), 3, color, -1)
        
        # ── Center object label ──
        if obj_name != "Sky/Air":
            label = f"{obj_name}"
            dist_label = f"{distance:.1f}m" if distance > 0 else "N/A"
            # Background box
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (cx+15, cy-th-5), (cx+20+tw, cy+5), (0,0,0), -1)
            cv2.putText(img, label, (cx+18, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            # Distance below
            cv2.putText(img, dist_label, (cx+18, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
        
        # ── HUD: Top-left info ──
        y = 25
        cv2.putText(img, "DRONE VISION", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y += 25
        cv2.putText(img, f"Front Dist: {distance:.1f}m", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        y += 20
        cv2.putText(img, f"Center: {obj_name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        # ── HUD: Detected objects list (right side) ──
        if detections:
            rx = w - 250
            ry = 25
            cv2.putText(img, f"DETECTED ({len(detections)}):", (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            for det in detections[:8]:
                ry += 18
                rel = det.relative_pose.position
                d = (rel.x_val**2 + rel.y_val**2 + rel.z_val**2)**0.5
                text = f"{det.name}: {d:.1f}m"
                cv2.putText(img, text, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
                # Draw bounding box
                cv2.rectangle(
                    img,
                    (int(det.box2D.min.x_val), int(det.box2D.min.y_val)),
                    (int(det.box2D.max.x_val), int(det.box2D.max.y_val)),
                    (0, 255, 0), 1
                )
                # Label on box
                cv2.putText(
                    img, f"{det.name} {d:.1f}m",
                    (int(det.box2D.min.x_val), int(det.box2D.min.y_val) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1
                )
        
        # ── HUD: Bottom controls ──
        cv2.putText(img, "q=Quit  s=Screenshot  l=List Objects", (10, h-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
        return img

    def _take_screenshot(self, img, obj_name):
        """Save screenshot."""
        self.screenshot_count += 1
        fname = os.path.join(SCREENSHOT_DIR, f"shot_{self.screenshot_count:04d}_{obj_name}.png")
        cv2.imwrite(fname, img)
        print(f"[📸] Screenshot saved: {fname}")

    def run(self):
        """Main loop - takeoff and start vision."""
        print("[⬆] Taking off...")
        self.client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
        self.client.moveToPositionAsync(0, 0, -5, 3, vehicle_name=VEHICLE_NAME).join()
        print("[✓] Airborne! Starting vision loop...\n")
        
        try:
            while True:
                # ── Get images ──
                responses = self.client.simGetImages([
                    airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Segmentation, False, False),
                    airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPerspective, True, False),
                ], vehicle_name=VEHICLE_NAME)
                
                # Scene image
                if responses[0].width == 0:
                    continue
                scene_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                scene_img = scene_1d.reshape(responses[0].height, responses[0].width, 3)
                scene_img = scene_img.copy()
                
                # Segmentation image
                seg_img = None
                if responses[1].width > 0:
                    seg_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
                    seg_img = seg_1d.reshape(responses[1].height, responses[1].width, 3)
                
                # Depth image
                depth_img = None
                if responses[2].width > 0:
                    depth_1d = np.array(responses[2].image_data_float, dtype=np.float32)
                    depth_img = depth_1d.reshape(responses[2].height, responses[2].width)
                
                # ── Object at crosshair center ──
                seg_id, center_obj = self._get_center_object_from_segmentation(seg_img)
                
                # ── Distance measurements ──
                front_dist = self._get_front_distance()
                lidar_dist, lidar_pt = self._get_front_lidar_distance()
                depth_dist = self._get_depth_at_center(depth_img)
                
                # Use best available distance
                best_dist = front_dist
                if best_dist < 0 and lidar_dist > 0:
                    best_dist = lidar_dist
                if best_dist < 0 and depth_dist > 0:
                    best_dist = depth_dist
                
                # ── Detections ──
                detections = self._get_detections()
                
                # ── Auto-screenshot new objects ──
                if center_obj not in self.seen_objects and center_obj != "Sky/Air":
                    self.seen_objects.add(center_obj)
                    print(f"[NEW] First time seeing: {center_obj} (segID={seg_id})")
                
                # ── Console output ──
                det_names = [d.name for d in detections[:5]]
                print(
                    f"\r[👁] Center: {center_obj:20s} | "
                    f"Dist: {best_dist:6.1f}m | "
                    f"LiDAR: {lidar_dist:6.1f}m | "
                    f"Depth: {depth_dist:6.1f}m | "
                    f"Detected: {det_names}",
                    end="", flush=True
                )
                
                # ── Draw HUD ──
                display = self._draw_crosshair(scene_img, center_obj, best_dist, detections)
                cv2.imshow("Drone Vision", display)
                
                # ── Keyboard ──
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._take_screenshot(display, center_obj)
                elif key == ord('l'):
                    print("\n\n=== SCENE OBJECTS ===")
                    for name, sid in sorted(self.object_registry.items()):
                        print(f"  {name:40s} segID={sid}")
                    print("=" * 50 + "\n")
                
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            print("\n[!] Interrupted")
        finally:
            cv2.destroyAllWindows()
            self.client.armDisarm(False, VEHICLE_NAME)
            self.client.enableApiControl(False, VEHICLE_NAME)
            print("\n[✓] Drone Vision stopped.")


if __name__ == "__main__":
    vision = DroneVision()
    vision.run()
