"""
AirSim Drone Controller v4 — GTA Style
========================================

CAMERA  (handled by C++ CameraDirector — mouse orbit on spring arm)
  Mouse move   → orbit camera around drone (GTA-style, always active)
  4            → FPV (first-person view through drone camera)
  5            → TPV (third-person, spring arm chase — default)

MOVEMENT  (camera-relative — W always moves toward where camera looks)
  W / S        → forward / backward (in camera-facing direction)
  A / D        → strafe left / right
  Space        → ascend
  Ctrl         → descend
  Q / E        → yaw rotate drone left / right

CAMERA FINE CONTROL
  Arrow keys   → fine pitch/yaw adjustment of FPV camera
  Z / X        → zoom spring arm in/out (not available from Python)

EXIT
  Esc          → land and quit

DETECTION
  • Identifies objects by name: tree, rock, cylinder, survivor, etc.
  • Shows stencil ID for unknown objects (so you can label them in Unreal)
  • 4m debug ray that stops at the object, color-coded by distance
  • Auto-screenshots new objects (FPV image saved to detected_objects/)
  • Filters sky (id-6), air (id-0), ground (id-1) — clean output only

CAMERA DIRECTION
  The C++ CameraDirector writes orbit yaw/pitch to:
      ~/Documents/AirSim/camera_state.txt
  This script reads that file to know which direction the camera faces,
  so WASD moves the drone in the camera-relative direction.

REQUIRES
  pip install airsim keyboard numpy
  (pynput NOT needed — mouse is handled by C++ now)
"""

import airsim
import keyboard
import time
import math
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# RL utilities (occupancy grid + object registry)
try:
    from rl_utils import OccupancyGrid, ObjectRegistry
    _HAS_RL = True
except ImportError:
    _HAS_RL = False
    print("[INFO] rl_utils.py not found — grid/registry disabled")

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SPEED          = 10.0     # m/s drone speed
YAW_RATE       = 70.0     # deg/s while Q/E held
PRINT_HZ       = 0.30     # seconds between terminal prints
CONE_DEG       = 20.0     # ±deg lidar "front" cone
MIN_HIT        = 0.20     # ignore lidar hits closer than this (drone body)
RAY_LEN        = 4.0      # debug ray length metres (shorter = cleaner)
ARROW_CAM_RATE = 1.5      # deg per frame for arrow key camera fine control

LIDAR_NAME     = "Lidar1"
VEHICLE_NAME   = "Drone1"
CAM_MAIN       = "0"         # main vehicle camera (FPV + segmentation)
SCREENSHOT_DIR = "detected_objects"
SURVIVOR_GLOB  = "Cube.*"

# Path where C++ CameraDirector writes orbit yaw/pitch
CAMERA_STATE_FILE = os.path.join(
    os.path.expanduser("~"), "Documents", "AirSim", "camera_state.txt"
)

# Stencil ID → human-readable label
# NOTE: Set stencil values in Unreal: select mesh → Details →
#       Render CustomDepth Pass = ON → Custom Depth Stencil Value = <id>
STENCIL_MAP = {
    0:   "air",
    1:   "ground",
    6:   "sky",           # BP_Sky_Sphere — always filtered
    10:  "tree",
    11:  "rock",
    15:  "cylinder",
    20:  "survivor",
}

# These stencil IDs are background — filtered from detection output
IGNORE_IDS = {0, 1, 6}

# ──────────────────────────────────────────────────────────────────────────────
#  CAMERA STATE READER  (reads orbit yaw from C++ CameraDirector)
# ──────────────────────────────────────────────────────────────────────────────
_last_cam_yaw   = 0.0
_last_cam_pitch = -20.0

def read_camera_state():
    """
    Read camera orbit yaw/pitch from the file written by CameraDirector C++.
    Returns (yaw_deg, pitch_deg). Falls back to last known values on error.
    """
    global _last_cam_yaw, _last_cam_pitch
    try:
        with open(CAMERA_STATE_FILE, "r") as f:
            parts = f.read().strip().split()
            if len(parts) >= 2:
                _last_cam_yaw   = float(parts[0])
                _last_cam_pitch = float(parts[1])
    except (FileNotFoundError, ValueError, IOError):
        pass  # use last known values
    return _last_cam_yaw, _last_cam_pitch


# ──────────────────────────────────────────────────────────────────────────────
#  SCREENSHOT FOLDER
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
_seen_ids = set()   # track which stencil IDs we've already screenshotted


# ──────────────────────────────────────────────────────────────────────────────
#  CONNECT
# ──────────────────────────────────────────────────────────────────────────────
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


# ──────────────────────────────────────────────────────────────────────────────
#  SCENE OBJECTS + SEGMENTATION SETUP
# ──────────────────────────────────────────────────────────────────────────────
cubes = client.simListSceneObjects(SURVIVOR_GLOB)
goal  = cubes[0] if cubes else None
print(f"[INIT] Cubes in scene : {cubes}")
print(f"[INIT] Tracking       : {goal}")


def _seg_setup():
    """Assigns specific Stencil IDs by looking at the object names in Unreal."""
    try:
        # 1. Reset everything to 0 (Air)
        client.simSetSegmentationObjectID(r"[\w]*", 0, True)

        # 2. Force the exact IDs based on the names in your Unreal World Outliner!
        # If your trees are named "SM_PineTree_1", "SM_PineTree_2", the regex ".*Tree.*" catches them.
        client.simSetSegmentationObjectID(r".*Cube.*", 20, True)   # Survivors = 20
        client.simSetSegmentationObjectID(r".*Tree.*", 10, True)   # Trees = 10
        client.simSetSegmentationObjectID(r".*Rock.*", 11, True)   # Rocks = 11
        client.simSetSegmentationObjectID(r".*Cylinder.*", 15, True) # Cylinders = 15
        
        # Filter out the sky and ground
        client.simSetSegmentationObjectID(r".*Sky.*", 6, True)
        client.simSetSegmentationObjectID(r".*Landscape.*", 1, True)
        client.simSetSegmentationObjectID(r".*Floor.*", 1, True)

        print("[INIT] Forced Segmentation IDs assigned successfully.")
    except Exception as e:
        print(f"[WARN] Seg setup failed: {e}")


_seg_setup()


# ──────────────────────────────────────────────────────────────────────────────
#  TAKEOFF
# ──────────────────────────────────────────────────────────────────────────────
print("[INIT] Taking off ...")
client.takeoffAsync().join()
time.sleep(0.6)

# Initialize RL systems
if _HAS_RL:
    occ_grid = OccupancyGrid(size_m=100.0, cell_m=0.5)
    obj_registry = ObjectRegistry(os.path.join(SCREENSHOT_DIR, "object_log.json"))
    print(f"[INIT] Occupancy grid: {occ_grid.n_cells}x{occ_grid.n_cells} cells")
    print(f"[INIT] Object registry: {obj_registry.path}")
else:
    occ_grid = None
    obj_registry = None


# ──────────────────────────────────────────────────────────────────────────────
#  STATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_state():
    """Returns (position, yaw_degrees, pitch_degrees)."""
    s  = client.getMultirotorState()
    p  = s.kinematics_estimated.position
    q  = s.kinematics_estimated.orientation

    # Yaw from quaternion
    sy = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cy = 1.0 - 2.0 * (q.y_val**2 + q.z_val**2)
    yaw = math.degrees(math.atan2(sy, cy))

    # Pitch from quaternion
    sp = 2.0 * (q.w_val * q.y_val - q.z_val * q.x_val)
    sp = max(-1.0, min(1.0, sp))
    pitch = math.degrees(math.asin(sp))

    return p, yaw, pitch


def front_lidar():
    """
    Returns (dist_m or None, label).
    Filters: drone body noise, pure vertical (ground) returns, behind drone.
    """
    try:
        pts = client.getLidarData(LIDAR_NAME, VEHICLE_NAME).point_cloud
        if len(pts) < 3:
            return None, "clear"
        best = float("inf")
        for i in range(0, len(pts) - 2, 3):
            x, y, z = pts[i], pts[i+1], pts[i+2]
            horiz = math.sqrt(x*x + y*y)
            dist  = math.sqrt(x*x + y*y + z*z)
            # Skip: too close (body), nearly vertical (ground), or behind
            if dist < MIN_HIT or horiz < 0.12 or x < 0:
                continue
            ang = math.degrees(math.atan2(y, x))
            if abs(ang) <= CONE_DEG:
                best = min(best, dist)
        return (best, "obstacle") if best < float("inf") else (None, "clear")
    except Exception:
        return None, "clear"


def seg_center_info():
    """
    Returns (stencil_id, label_string) from segmentation camera center pixel.
    Scans a 5×5 grid around center for the dominant non-background object.
    Returns (None, None) if nothing meaningful in frame center.
    """
    try:
        r = client.simGetImages([
            airsim.ImageRequest(CAM_MAIN, airsim.ImageType.Segmentation, False, False)
        ])
        if not r or r[0].width == 0:
            return None, None

        img = np.frombuffer(r[0].image_data_uint8, dtype=np.uint8)
        img = img.reshape(r[0].height, r[0].width, 3)

        cy, cx = r[0].height // 2, r[0].width // 2

        # Scan 5×5 grid around center for dominant non-background stencil
        stencil_counts = {}
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                py = max(0, min(cy + dy * 8, r[0].height - 1))
                px = max(0, min(cx + dx * 8, r[0].width - 1))
                sid = int(img[py, px, 0])
                if sid not in IGNORE_IDS:
                    stencil_counts[sid] = stencil_counts.get(sid, 0) + 1

        if not stencil_counts:
            return None, None

        # Pick the most common non-background stencil
        best_sid = max(stencil_counts, key=stencil_counts.get)
        label = STENCIL_MAP.get(best_sid, f"unknown-id{best_sid}")
        return best_sid, label

    except Exception:
        return None, None


def take_screenshot(sid, label, dist_m):
    """Saves an image safely without crashing OpenCV."""
    if sid in _seen_ids:
        return
    _seen_ids.add(sid)
    try:
        resp = client.simGetImages([
            airsim.ImageRequest(CAM_MAIN, airsim.ImageType.Scene, False, False)
        ])
        if not resp or resp[0].width == 0:
            return
            
        ts = datetime.now().strftime("%H%M%S")
        dist = f"{dist_m:.1f}m" if dist_m else "unkdist"
        fname = os.path.join(SCREENSHOT_DIR, f"id{sid}_{label}_{dist}_{ts}.png")
        
        # Safely write the raw image bytes directly to a file
        airsim.write_file(fname, resp[0].image_data_uint8)
        print(f"  📸 NEW OBJECT! Screenshot saved to → {fname}")
        
    except Exception as e:
        print(f"  [WARN] Screenshot failed: {e}")


def draw_ray(pos, yaw_deg, cam_yaw_offset, dist_m, label, sid):
    """
    Draw a debug ray from drone in camera-facing direction.
    • 4m max length, stops at hit object
    • Color: red (<1.5m), amber (<3.5m), green (clear)
    • Label shows: object_name distance [stencil:N]
    """
    try:
        total_yaw = math.radians(yaw_deg + cam_yaw_offset)
        fx = math.cos(total_yaw)
        fy = math.sin(total_yaw)
        tip = min(dist_m, RAY_LEN) if dist_m else RAY_LEN

        s = airsim.Vector3r(pos.x_val, pos.y_val, pos.z_val)
        e = airsim.Vector3r(pos.x_val + fx * tip,
                            pos.y_val + fy * tip,
                            pos.z_val)

        # Color by distance
        if   dist_m and dist_m <= 1.5: rgba = [1.0, 0.1, 0.1, 1.0]   # red
        elif dist_m and dist_m <= 3.5: rgba = [1.0, 0.7, 0.0, 1.0]   # amber
        else:                          rgba = [0.1, 1.0, 0.3, 1.0]   # green

        dur = PRINT_HZ + 0.12

        # Arrow
        client.simPlotArrows([s], [e], arrow_size=3.0,
                             color_rgba=rgba, duration=dur, is_persistent=False)

        # Label above tip
        txt_pos = airsim.Vector3r(e.x_val, e.y_val, e.z_val - 0.35)
        if dist_m and label not in ("clear", "obstacle"):
            txt = f"{label} {dist_m:.1f}m [stencil:{sid}]"
        elif dist_m:
            txt = f"{label} {dist_m:.1f}m"
        else:
            txt = "clear"
        client.simPlotStrings([txt], [txt_pos], scale=5,
                              color_rgba=rgba, duration=dur)

    except Exception:
        pass


def set_fpv_camera(pitch, yaw_off):
    """Tilt/pan the FPV camera (vehicle camera 0) for look-around in FPV mode."""
    try:
        pose = airsim.Pose(
            airsim.Vector3r(0, 0, 0),
            airsim.to_quaternion(math.radians(pitch), 0, math.radians(yaw_off))
        )
        client.simSetCameraPose(CAM_MAIN, pose, vehicle_name=VEHICLE_NAME)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — import into RL environment
# ──────────────────────────────────────────────────────────────────────────────
def getObject(name: str) -> dict:
    """
    Returns position + distance for any named Unreal actor.

    In RL env:
        from drone_control4 import client, getObject
        info = getObject("Cube_1")
        # → {"name": "Cube_1", "pos": (-1.9, -18.7, -1.6), "dist": 9.4}
    """
    try:
        g   = client.simGetObjectPose(name).position
        pos = client.getMultirotorState().kinematics_estimated.position
        d   = math.sqrt((g.x_val - pos.x_val)**2 +
                        (g.y_val - pos.y_val)**2 +
                        (g.z_val - pos.z_val)**2)
        return {
            "name": name,
            "pos":  (round(g.x_val, 2), round(g.y_val, 2), round(g.z_val, 2)),
            "dist": round(d, 2),
        }
    except Exception as ex:
        return {"name": name, "pos": None, "dist": None, "error": str(ex)}


def get_detected_objects():
    """
    Returns a dict of all currently detected object types + their screenshot status.
    Useful for RL reward shaping.
    """
    return {
        "seen_stencil_ids": list(_seen_ids),
        "labels": {sid: STENCIL_MAP.get(sid, f"unknown-id{sid}") for sid in _seen_ids},
        "screenshot_dir": os.path.abspath(SCREENSHOT_DIR),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

# FPV camera state (used when in FPV mode for arrow key fine control)
fpv_pitch = 0.0
fpv_yaw   = 0.0

# Per-loop state
cur_yaw   = 0.0
yaw_cd    = 0.0
last_print = 0.0
DT        = 0.04   # 25 Hz

print("\n" + "=" * 60)
print("  GTA STYLE DRONE CONTROLLER v4")
print("=" * 60)
print("  MOVEMENT (camera-relative)")
print("    W/S/A/D     = forward/back/strafe (toward camera)")
print("    Space/Ctrl  = up / down")
print("    Q / E       = yaw left / right")
print("")
print("  CAMERA")
print("    Mouse       = orbit around drone (GTA style)")
print("    Arrow keys  = fine pitch/yaw (FPV mode)")
print("    4           = FPV (first person)")
print("    5           = TPV (third person, default)")
print("")
print("  OTHER")
print("    Esc         = land + quit")
print("=" * 60)
print("")

# Check if camera state file exists (C++ should create it)
if not os.path.exists(CAMERA_STATE_FILE):
    print(f"[INFO] Waiting for camera state file: {CAMERA_STATE_FILE}")
    print("[INFO] (This file is created by C++ CameraDirector)")
    print("[INFO] If you haven't rebuilt AirSim, WASD will use drone yaw instead")
    print("")

try:
    while True:
        now = time.time()

        # ── READ CAMERA DIRECTION ────────────────────────────────────────────
        cam_yaw, cam_pitch = read_camera_state()

        # ── READ DRONE STATE ─────────────────────────────────────────────────
        pos, drone_yaw, drone_pitch = get_state()

        # ── FPV CAMERA FINE CONTROL (arrow keys) ─────────────────────────────
        if keyboard.is_pressed('up'):    fpv_pitch -= ARROW_CAM_RATE
        if keyboard.is_pressed('down'):  fpv_pitch += ARROW_CAM_RATE
        if keyboard.is_pressed('left'):  fpv_yaw   -= ARROW_CAM_RATE
        if keyboard.is_pressed('right'): fpv_yaw   += ARROW_CAM_RATE
        fpv_pitch = max(-85.0, min(85.0, fpv_pitch))
        fpv_yaw   = max(-180.0, min(180.0, fpv_yaw))
        set_fpv_camera(fpv_pitch, fpv_yaw)

        # ── MOVEMENT (GTA STYLE — camera-relative) ───────────────────────────
        # cam_yaw is the world-space yaw the camera faces (from C++ spring arm)
        # When camera state file isn't available, fall back to drone_yaw
        if os.path.exists(CAMERA_STATE_FILE):
            move_yaw_rad = math.radians(cam_yaw)
        else:
            move_yaw_rad = math.radians(drone_yaw)

        vx_world = 0.0
        vy_world = 0.0
        vz       = 0.0

        # WASD in camera-facing world direction
        if keyboard.is_pressed('w'):
            vx_world +=  SPEED * math.cos(move_yaw_rad)
            vy_world +=  SPEED * math.sin(move_yaw_rad)
        if keyboard.is_pressed('s'):
            vx_world += -SPEED * math.cos(move_yaw_rad)
            vy_world += -SPEED * math.sin(move_yaw_rad)
        if keyboard.is_pressed('a'):
            # strafe left = 90° left of camera direction
            vx_world +=  SPEED * math.cos(move_yaw_rad - math.pi/2)
            vy_world +=  SPEED * math.sin(move_yaw_rad - math.pi/2)
        if keyboard.is_pressed('d'):
            # strafe right = 90° right of camera direction
            vx_world +=  SPEED * math.cos(move_yaw_rad + math.pi/2)
            vy_world +=  SPEED * math.sin(move_yaw_rad + math.pi/2)

        if keyboard.is_pressed('space'): vz = -3.5   # up (AirSim Z is down)
        if keyboard.is_pressed('ctrl'):  vz =  3.5   # down

        # Send as world-frame velocity
        client.moveByVelocityAsync(
            vx_world, vy_world, vz, DT * 1.5,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=drone_yaw)
        )

        # ── YAW (Q/E smooth) ─────────────────────────────────────────────────
        if now > yaw_cd:
            if keyboard.is_pressed('q'):
                client.rotateByYawRateAsync(-YAW_RATE, DT).join()
                yaw_cd = now + DT
            elif keyboard.is_pressed('e'):
                client.rotateByYawRateAsync(YAW_RATE, DT).join()
                yaw_cd = now + DT

        # ── DETECTION + OUTPUT ────────────────────────────────────────────────
        if now - last_print > PRINT_HZ:
            lidar_d, _    = front_lidar()
            sid, seg_lbl  = seg_center_info()

            # Resolve best label:
            #   1. Segmentation wins if it's a known named object
            #   2. Unknown stencil → show "unknown-idNN" (go set it in Unreal)
            #   3. Lidar hit without identity → "obstacle"
            #   4. Nothing → "clear"
            if seg_lbl and seg_lbl not in ("air", "ground", "sky"):
                label = seg_lbl
            elif lidar_d:
                label = "obstacle"
            else:
                label = "clear"

            # Auto-screenshot any new object type
            if sid is not None and sid not in IGNORE_IDS:
                take_screenshot(sid, label, lidar_d)

            # Log to object registry
            if obj_registry and sid is not None and sid not in IGNORE_IDS:
                drone_pos = (round(pos.x_val, 2), round(pos.y_val, 2), round(pos.z_val, 2))
                obj_registry.log_object(sid, label, lidar_d, drone_pos)

            # Update occupancy grid from lidar
            if occ_grid:
                occ_grid.update_from_lidar(client, LIDAR_NAME, VEHICLE_NAME)

            # Draw debug ray in camera-facing direction
            draw_ray(pos, drone_yaw, cam_yaw - drone_yaw if os.path.exists(CAMERA_STATE_FILE) else 0,
                     lidar_d, label, sid)

            # Clean terminal output
            fd   = f"{lidar_d:.1f}m" if lidar_d else ">4m"
            warn = " ⚠ CLOSE!" if lidar_d and lidar_d < 1.5 else ""
            hit  = " ★ SURVIVOR!" if label == "survivor" and lidar_d and lidar_d < 2.0 else ""
            sid_str = f" stencil:{sid}" if sid is not None else ""
            expl = f"  map:{occ_grid.exploration_percent():.1f}%" if occ_grid else ""

            print(f"Front: {fd} [{label}]{sid_str}{warn}{hit}{expl}")

            last_print = now

        # ── EXIT ──────────────────────────────────────────────────────────────
        if keyboard.is_pressed('esc'):
            break

        time.sleep(DT)

except KeyboardInterrupt:
    pass
finally:
    print("\n[EXIT] Landing ...")
    try:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
    except Exception:
        pass

    # Save RL data
    if occ_grid:
        occ_grid.save_grid(os.path.join(SCREENSHOT_DIR, "occupancy_grid.npy"))
        try:
            occ_grid.save_as_image(os.path.join(SCREENSHOT_DIR, "occupancy_map.png"))
        except Exception:
            pass
        stats = occ_grid.get_stats()
        print(f"[EXIT] Grid saved — {stats['explored_pct']:.1f}% explored")
        print(f"       Free: {stats['free_cells']}  Occupied: {stats['occupied_cells']}")

    if obj_registry:
        obj_registry.save()
        print(f"[EXIT] Object registry saved → {obj_registry.path}")
        summary = obj_registry.get_summary()
        if summary != "No objects detected yet":
            print(f"[EXIT] Objects found:\n{summary}")

    print("[EXIT] Done.")