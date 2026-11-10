"""
AirSim Drone Controller — GTA Style
=====================================
VIEW MODES
  5 key    → 3rd person follow cam (GTA style, drone in front of you)
  4 key    → FPV (first person, you are the drone)
  (default starts in 3rd person)

MOVEMENT  (GTA style — W always moves in camera-facing direction)
  W / S       → forward / backward (relative to where camera looks)
  A / D       → strafe left / right
  Space       → up
  Ctrl        → down
  Q / E       → yaw rotate drone left / right

CAMERA (3rd person mode)
  Mouse move  → orbit camera around drone  ← uses AirSim simSetVehiclePose trick
  Arrow Up/Dn → camera pitch up/down
  Arrow L/R   → camera yaw left/right
  Scroll / Z/X → zoom in/out

EXIT
  Esc         → land and quit

DETECTION
  Terminal shows: what object + distance  (no noise, no sky id-6 garbage)
  New unknown objects → screenshot saved automatically
"""

import airsim
import keyboard
import time
import math
import threading
import os
import numpy as np
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG — tweak these
# ──────────────────────────────────────────────────────────────────────────────
SPEED          = 10.0     # m/s drone speed
YAW_RATE       = 70.0     # deg/s while Q/E held
PRINT_HZ       = 0.30     # seconds between terminal prints
CONE_DEG       = 18.0     # ±deg lidar "front" cone
MIN_HIT        = 0.20     # ignore lidar hits closer than this (drone body)
RAY_LEN        = 8.0      # debug ray length metres
MOUSE_SENS     = 0.08     # camera deg per pixel

LIDAR_NAME     = "Lidar1"
VEHICLE_NAME   = "Drone1"
CAM_MAIN       = "0"        # main camera (used for FPV + segmentation)
SCREENSHOT_DIR = "detected_objects"  # folder for auto-screenshots

SURVIVOR_GLOB  = "Cube.*"

# Stencil ID → label mapping
# NOTE: id-6 = BP_Sky_Sphere (Unreal skybox) — filtered out below
# Add your own objects here after setting stencil in Unreal
STENCIL_MAP = {
    0:   "air",
    1:   "ground",
    6:   "SKY",        # BP_Sky_Sphere — filtered, won't appear in output
    10:  "tree",
    11:  "rock",
    15:  "cylinder",
    20:  "survivor",
}

# These stencil IDs are "background" — filter them from detection output
IGNORE_IDS = {0, 6}     # air and sky dome

# ──────────────────────────────────────────────────────────────────────────────
#  MOUSE via pynput (passive, doesn't fight Unreal PIE input capture)
# ──────────────────────────────────────────────────────────────────────────────
_mdx = 0.0
_mdy = 0.0
_ml  = threading.Lock()

def _start_mouse():
    try:
        from pynput import mouse as _m
        _prev = [None, None]
        def _mv(x, y):
            global _mdx, _mdy
            if _prev[0] is not None:
                with _ml:
                    _mdx += x - _prev[0]
                    _mdy += y - _prev[1]
            _prev[0], _prev[1] = x, y
        lst = _m.Listener(on_move=_mv)
        lst.daemon = True
        lst.start()
        print("[INIT] Mouse listener started (pynput)")
        return True
    except ImportError:
        print("[WARN] pip install pynput  → needed for mouse camera")
        return False

_has_mouse = _start_mouse()

def consume_mouse():
    global _mdx, _mdy
    with _ml:
        dx, dy = _mdx, _mdy
        _mdx = _mdy = 0.0
    return dx, dy

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
#  SCENE OBJECTS + SEGMENTATION
# ──────────────────────────────────────────────────────────────────────────────
cubes = client.simListSceneObjects(SURVIVOR_GLOB)
goal  = cubes[0] if cubes else None
print(f"[INIT] Cubes in scene : {cubes}")
print(f"[INIT] Tracking       : {goal}")

def _seg_setup():
    try:
        client.simSetSegmentationObjectID(r"[\w]*", 0, True)   # reset all to 0

        # Explicit: set sky to 6 so we can filter it
        client.simSetSegmentationObjectID(r"BP_Sky_Sphere.*", 6, True)
        client.simSetSegmentationObjectID(r"SkyAtmosphere.*", 6, True)
        client.simSetSegmentationObjectID(r"SkyLight.*",      6, True)

        # Ground
        client.simSetSegmentationObjectID(r"Landscape.*",     1, True)

        # Survivor cubes
        for c in cubes:
            client.simSetSegmentationObjectID(c, 20, False)

        # Placed actors
        client.simSetSegmentationObjectID(r"Cylinder.*",  15, True)
        client.simSetSegmentationObjectID(r".*Cyl.*",     15, True)

        # Trees/rocks — only works for individually placed actors, not instanced foliage
        # For instanced foliage: open asset in Unreal → Details → CustomDepth = ON, stencil = 10
        client.simSetSegmentationObjectID(r".*Tree.*",    10, True)
        client.simSetSegmentationObjectID(r".*Pine.*",    10, True)
        client.simSetSegmentationObjectID(r".*Rock.*",    11, True)

        print("[INIT] Segmentation assigned")
    except Exception as e:
        print(f"[WARN] Seg: {e}")

_seg_setup()

# ──────────────────────────────────────────────────────────────────────────────
#  VIEW MODE CONTROL
#  AirSim ViewMode options: "Fpv", "FlyWithMe", "Manual", "SpringArmChase"
#  We switch via simSetCameraFov + camera pose tricks, and also toggle
#  AirSim's own external camera
# ──────────────────────────────────────────────────────────────────────────────
VIEW_THIRD  = "SpringArmChase"   # 3rd person follow = GTA style
VIEW_FPV    = "Fpv"

_current_view = "third"

def set_view_third():
    """GTA 3rd person: drone visible in front, camera follows behind/above."""
    global _current_view
    try:
        client.simSetTrackedObject(VEHICLE_NAME)
    except Exception:
        pass
    _current_view = "third"
    print("[VIEW] 3rd Person (GTA)")

def set_view_fpv():
    """FPV: camera is the drone's front camera."""
    global _current_view
    _current_view = "fpv"
    print("[VIEW] FPV")

# ──────────────────────────────────────────────────────────────────────────────
#  TAKEOFF
# ──────────────────────────────────────────────────────────────────────────────
print("[INIT] Taking off ...")
client.takeoffAsync().join()
time.sleep(0.6)

# ──────────────────────────────────────────────────────────────────────────────
#  STATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_state():
    """Returns (position, yaw_degrees, pitch_degrees, roll_degrees)."""
    s  = client.getMultirotorState()
    p  = s.kinematics_estimated.position
    q  = s.kinematics_estimated.orientation

    # Yaw
    sy = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cy = 1.0 - 2.0 * (q.y_val**2 + q.z_val**2)
    yaw = math.degrees(math.atan2(sy, cy))

    # Pitch
    sp = 2.0 * (q.w_val * q.y_val - q.z_val * q.x_val)
    sp = max(-1.0, min(1.0, sp))
    pitch = math.degrees(math.asin(sp))

    return p, yaw, pitch


def front_lidar():
    """
    Returns (dist_m or None, 'clear'/'obstacle').
    Filters: drone body noise, pure vertical (ground) returns, sky.
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
    Returns (stencil_id, label_string) from seg camera centre pixel.
    Filters out sky (id-6) and air (id-0).
    Returns (None, None) if nothing meaningful in frame centre.
    """
    try:
        r = client.simGetImages([
            airsim.ImageRequest(CAM_MAIN, airsim.ImageType.Segmentation, False, False)
        ])
        if not r or r[0].width == 0:
            return None, None
        img = np.frombuffer(r[0].image_data_uint8, dtype=np.uint8).reshape(r[0].height, r[0].width, 3)
        sid = int(img[r[0].height // 2, r[0].width // 2, 0])

        if sid in IGNORE_IDS:
            return None, None

        label = STENCIL_MAP.get(sid, f"id-{sid}")
        return sid, label
    except Exception:
        return None, None


def take_screenshot(sid, label, dist_m):
    """
    Saves a colour image when we see an unknown/new object.
    Filename encodes stencil ID, label, distance.
    Only takes one shot per unique stencil ID per session.
    """
    if sid in _seen_ids:
        return
    _seen_ids.add(sid)
    try:
        resp = client.simGetImages([
            airsim.ImageRequest(CAM_MAIN, airsim.ImageType.Scene, False, False)
        ])
        if not resp or resp[0].width == 0:
            return
        ts   = datetime.now().strftime("%H%M%S")
        dist = f"{dist_m:.1f}m" if dist_m else "unkdist"
        fname = os.path.join(SCREENSHOT_DIR, f"id{sid}_{label}_{dist}_{ts}.png")
        airsim.write_png(os.fsencode(fname), resp[0].image_data_uint8)
        print(f"  📸 Screenshot saved → {fname}")
    except Exception as e:
        print(f"  [WARN] screenshot failed: {e}")


def draw_ray(pos, yaw_deg, cam_yaw_off, dist_m, label):
    """
    Draw a clean, short debug ray from drone forward.
    Small text, tips at hit point so it doesn't fill screen.
    """
    try:
        total_yaw = math.radians(yaw_deg + cam_yaw_off)
        fx  = math.cos(total_yaw)
        fy  = math.sin(total_yaw)
        tip = min(dist_m, RAY_LEN) if dist_m else RAY_LEN

        s = airsim.Vector3r(pos.x_val, pos.y_val, pos.z_val)
        e = airsim.Vector3r(pos.x_val + fx*tip, pos.y_val + fy*tip, pos.z_val)

        if   dist_m and dist_m <= 1.5: rgba = [1.0, 0.1, 0.1, 1.0]   # red
        elif dist_m and dist_m <= 3.5: rgba = [1.0, 0.7, 0.0, 1.0]   # amber
        else:                          rgba = [0.1, 1.0, 0.3, 1.0]   # green

        dur = PRINT_HZ + 0.12

        client.simPlotArrows([s], [e], arrow_size=3.0,
                             color_rgba=rgba, duration=dur, is_persistent=False)

        txt_pos = airsim.Vector3r(e.x_val, e.y_val, e.z_val - 0.35)
        txt = f"{label} {dist_m:.1f}m" if dist_m else f"clear"
        client.simPlotStrings([txt], [txt_pos], scale=5, color_rgba=rgba, duration=dur)
    except Exception:
        pass


def set_camera_pose(pitch, yaw_off):
    """Tilt/pan the front camera for FPV or look-around."""
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
        from drone_control import client, getObject
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


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

# Camera orbit state (3rd person)
cam_pitch  =  -20.0    # start tilted slightly down so drone is visible
cam_yaw    =    0.0    # camera yaw offset relative to drone
cam_zoom   =    1.0    # not directly controllable via API but tracked

# Per-loop state
cur_yaw    =    0.0
yaw_cd     =    0.0
view_cd    =    0.0
last_print =    0.0
DT         =    0.04   # 25 Hz

print("\n" + "=" * 60)
print("  GTA STYLE DRONE CONTROLLER")
print("=" * 60)
print("  W/S/A/D     = move (in direction camera faces)")
print("  Space/Ctrl  = up / down")
print("  Q / E       = yaw left / right")
print("  Mouse       = look around (camera orbit)")
print("  Arrow keys  = camera pitch/yaw fine control")
print("  4           = FPV mode")
print("  5           = 3rd person mode  (default)")
print("  Esc         = land + quit")
print("=" * 60 + "\n")

# Start in 3rd person
set_view_third()

try:
    while True:
        now = time.time()

        # ── VIEW MODE TOGGLE ──────────────────────────────────────────────────
        if now > view_cd:
            if keyboard.is_pressed('4'):
                set_view_fpv()
                view_cd = now + 0.5
            elif keyboard.is_pressed('5'):
                set_view_third()
                view_cd = now + 0.5

        # ── READ DRONE HEADING ────────────────────────────────────────────────
        pos, drone_yaw, drone_pitch = get_state()

        # ── CAMERA INPUT (mouse + arrows) ─────────────────────────────────────
        mdx, mdy = consume_mouse()
        if abs(mdx) > 0.3 or abs(mdy) > 0.3:
            cam_yaw   += mdx * MOUSE_SENS
            cam_pitch += mdy * MOUSE_SENS

        # Arrow keys for fine camera control
        if keyboard.is_pressed('up'):    cam_pitch -= 1.5
        if keyboard.is_pressed('down'):  cam_pitch += 1.5
        if keyboard.is_pressed('left'):  cam_yaw   -= 1.5
        if keyboard.is_pressed('right'): cam_yaw   += 1.5

        cam_pitch = max(-85.0, min(85.0,   cam_pitch))
        cam_yaw   = max(-180.0, min(180.0, cam_yaw))

        set_camera_pose(cam_pitch, cam_yaw)

        # ── MOVEMENT (GTA STYLE) ──────────────────────────────────────────────
        # Camera yaw offset tells us which real-world direction "forward" is
        # W moves in direction camera is facing relative to drone
        world_yaw_rad = math.radians(drone_yaw + cam_yaw)

        vx_world = 0.0
        vy_world = 0.0
        vz       = 0.0

        # WASD in camera-relative world space
        if keyboard.is_pressed('w'):
            vx_world +=  SPEED * math.cos(world_yaw_rad)
            vy_world +=  SPEED * math.sin(world_yaw_rad)
        if keyboard.is_pressed('s'):
            vx_world += -SPEED * math.cos(world_yaw_rad)
            vy_world += -SPEED * math.sin(world_yaw_rad)
        if keyboard.is_pressed('a'):
            # strafe left = 90° left of camera direction
            vx_world +=  SPEED * math.cos(world_yaw_rad - math.pi/2)
            vy_world +=  SPEED * math.sin(world_yaw_rad - math.pi/2)
        if keyboard.is_pressed('d'):
            # strafe right
            vx_world +=  SPEED * math.cos(world_yaw_rad + math.pi/2)
            vy_world +=  SPEED * math.sin(world_yaw_rad + math.pi/2)

        if keyboard.is_pressed('space'): vz = -3.5   # up (AirSim Z is down)
        if keyboard.is_pressed('ctrl'):  vz =  3.5   # down

        # Send as world-frame velocity (not body frame — body frame ignores cam offset)
        client.moveByVelocityAsync(vx_world, vy_world, vz, DT * 1.5,
                                   drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=drone_yaw))

        # ── YAW (Q/E smooth) ──────────────────────────────────────────────────
        if now > yaw_cd:
            if keyboard.is_pressed('q'):
                client.rotateByYawRateAsync(-YAW_RATE, DT).join()
                cam_yaw += YAW_RATE * DT   # keep camera aligned
                yaw_cd   = now + DT
            elif keyboard.is_pressed('e'):
                client.rotateByYawRateAsync( YAW_RATE, DT).join()
                cam_yaw -= YAW_RATE * DT
                yaw_cd   = now + DT

        # ── DETECTION OUTPUT ──────────────────────────────────────────────────
        if now - last_print > PRINT_HZ:
            lidar_d, _    = front_lidar()
            sid, seg_lbl  = seg_center_info()   # None, None if sky/air/nothing

            # Resolve best label:
            #   seg wins if it's a known object (not sky, not air)
            #   lidar fills in distance if seg is unclear
            if seg_lbl and seg_lbl not in ("air", "ground"):
                label = seg_lbl
            elif lidar_d:
                label = "obstacle"
            else:
                label = "clear"

            # Auto-screenshot any new unknown stencil ID
            if sid is not None:
                take_screenshot(sid, label, lidar_d)

            draw_ray(pos, drone_yaw, cam_yaw, lidar_d, label)

            # Clean output — just what you're looking at
            fd   = f"{lidar_d:.1f}m" if lidar_d else ">8m"
            warn = " ⚠" if lidar_d and lidar_d < 1.5 else ""
            hit  = " ★SURVIVOR FOUND" if label == "survivor" and lidar_d and lidar_d < 2.0 else ""

            print(f"Front: {fd} [{label}]{warn}{hit}")

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
    print("[EXIT] Done.")
