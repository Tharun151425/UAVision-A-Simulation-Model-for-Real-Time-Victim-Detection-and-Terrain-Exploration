"""
AirSim Drone Controller v3
==========================
Movement   : Arrow keys        = forward / back / strafe left / right
             Space / Ctrl      = up / down
Yaw        : Q / E             = rotate left / right (smooth)
Camera     : Mouse             = look (pitch + yaw, no Unreal conflict)
Exit       : Esc

OUTPUT     : single clean line  →  Front: 3.2m [tree]  |  Goal: 9.1m
"""

import airsim
import keyboard
import time
import math
import threading

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SPEED         = 8.0      # m/s translation speed
YAW_RATE      = 60.0     # deg/s while Q/E held
PRINT_HZ      = 0.25     # seconds between output lines
CONE_DEG      = 22.0     # ±deg for "front" lidar cone
MIN_HIT       = 0.18     # ignore lidar hits closer (drone body noise)
RAY_LEN       = 6.0      # debug ray length in metres
MOUSE_SENS    = 0.10     # camera degrees per pixel of mouse movement

LIDAR_NAME    = "Lidar1"
VEHICLE_NAME  = "Drone1"
CAM_NAME      = "0"
SURVIVOR_GLOB = "Cube.*"

# Stencil ID → label
# Set these in Unreal: select mesh → Details → Render CustomDepth Pass = ON
#                                             → Custom Depth Stencil Value = <id>
STENCIL_MAP = {
    0:  "air",
    1:  "ground",
    10: "tree",
    11: "rock",
    15: "cylinder",
    20: "survivor",
}

# ──────────────────────────────────────────────────────────────────────────────
#  MOUSE via pynput  (doesn't conflict with Unreal's input capture)
# ──────────────────────────────────────────────────────────────────────────────
_mdx = 0.0
_mdy = 0.0
_ml  = threading.Lock()

def _start_mouse():
    try:
        from pynput import mouse as _m
        _prev = [None, None]

        def _on_move(x, y):
            global _mdx, _mdy
            if _prev[0] is not None:
                with _ml:
                    _mdx += x - _prev[0]
                    _mdy += y - _prev[1]
            _prev[0], _prev[1] = x, y

        lst = _m.Listener(on_move=_on_move)
        lst.daemon = True
        lst.start()
        return True
    except ImportError:
        print("[WARN] pip install pynput  to enable mouse camera")
        return False

_has_mouse = _start_mouse()

def consume_mouse():
    global _mdx, _mdy
    with _ml:
        dx, dy = _mdx, _mdy
        _mdx = _mdy = 0.0
    return dx, dy

# ──────────────────────────────────────────────────────────────────────────────
#  CONNECT
# ──────────────────────────────────────────────────────────────────────────────
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# ──────────────────────────────────────────────────────────────────────────────
#  SCENE + SEGMENTATION
# ──────────────────────────────────────────────────────────────────────────────
cubes = client.simListSceneObjects(SURVIVOR_GLOB)
goal  = cubes[0] if cubes else None
print(f"[INIT] Cubes in scene : {cubes}")
print(f"[INIT] Tracking       : {goal}")

def _seg_setup():
    try:
        client.simSetSegmentationObjectID(r"[\w]*", 0, True)   # reset all
        for c in cubes:
            client.simSetSegmentationObjectID(c, 20, False)    # exact match
        client.simSetSegmentationObjectID(r"Landscape.*", 1, True)
        client.simSetSegmentationObjectID(r"Cylinder.*",  15, True)
        client.simSetSegmentationObjectID(r".*Cyl.*",     15, True)
        # Trees only work for individually placed actors, not instanced foliage
        client.simSetSegmentationObjectID(r".*Tree.*",    10, True)
        client.simSetSegmentationObjectID(r".*Pine.*",    10, True)
        client.simSetSegmentationObjectID(r".*Rock.*",    11, True)
        print("[INIT] Segmentation assigned")
    except Exception as e:
        print(f"[WARN] Seg: {e}")

_seg_setup()

# ──────────────────────────────────────────────────────────────────────────────
#  TAKEOFF
# ──────────────────────────────────────────────────────────────────────────────
print("[INIT] Taking off ...")
client.takeoffAsync().join()
time.sleep(0.5)

# ──────────────────────────────────────────────────────────────────────────────
#  SENSOR HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_state():
    """Returns (position, yaw_degrees)."""
    s  = client.getMultirotorState()
    p  = s.kinematics_estimated.position
    q  = s.kinematics_estimated.orientation
    sy = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cy = 1.0 - 2.0 * (q.y_val**2 + q.z_val**2)
    return p, math.degrees(math.atan2(sy, cy))


def front_lidar():
    """
    Returns (dist_m, 'clear'|'obstacle').
    Filters ground hits: requires point to have meaningful horizontal distance
    so vertically-downward returns (ground bounce) are ignored.
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
            if dist < MIN_HIT or horiz < 0.1:   # skip body noise + ground
                continue
            ang = math.degrees(math.atan2(y, x))
            if abs(ang) <= CONE_DEG:
                best = min(best, dist)
        return (best, "obstacle") if best < float("inf") else (None, "clear")
    except Exception:
        return None, "clear"


def seg_center():
    """
    Returns label string from segmentation camera centre pixel.
    If stencil is unknown → returns 'id-NN' so you know to set it in Unreal.
    """
    try:
        import numpy as np
        r = client.simGetImages([
            airsim.ImageRequest("seg_front", airsim.ImageType.Segmentation, False, False)
        ])
        if not r or r[0].width == 0:
            return None
        img = np.frombuffer(r[0].image_data_uint8, dtype=np.uint8).reshape(r[0].height, r[0].width, 3)
        sid = int(img[r[0].height // 2, r[0].width // 2, 0])
        if sid == 0:
            return None
        return STENCIL_MAP.get(sid, f"id-{sid}")
    except Exception:
        return None


def get_goal_dist(pos):
    """Returns (goal_position, distance_m) or (None, None)."""
    if not goal:
        return None, None
    try:
        g  = client.simGetObjectPose(goal).position
        d  = math.sqrt((g.x_val - pos.x_val)**2 +
                       (g.y_val - pos.y_val)**2 +
                       (g.z_val - pos.z_val)**2)
        return g, d
    except Exception:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC getObject() — use this in your RL env
# ──────────────────────────────────────────────────────────────────────────────
def getObject(name: str) -> dict:
    """
    Returns position + drone distance for any named Unreal actor.

    Example (in RL env):
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
#  DEBUG RAY
# ──────────────────────────────────────────────────────────────────────────────
def draw_ray(pos, yaw_deg, cam_yaw_off, dist_m, label):
    """
    Draw a clean arrow in Unreal viewport.
    Shorter text (scale=6), arrow_size=4 — much less cluttered.
    End-point raised by 0.3m so text doesn't sit on the arrow tip.
    """
    try:
        total_yaw = math.radians(yaw_deg + cam_yaw_off)
        fx = math.cos(total_yaw)
        fy = math.sin(total_yaw)
        tip_d = min(dist_m, RAY_LEN) if dist_m else RAY_LEN

        s = airsim.Vector3r(pos.x_val,           pos.y_val,           pos.z_val)
        e = airsim.Vector3r(pos.x_val + fx*tip_d, pos.y_val + fy*tip_d, pos.z_val)

        if   dist_m is None or dist_m > 3.5: rgba = [0.1, 1.0, 0.2, 1.0]
        elif dist_m > 1.5:                   rgba = [1.0, 0.7, 0.0, 1.0]
        else:                                rgba = [1.0, 0.1, 0.1, 1.0]

        dur = PRINT_HZ + 0.12

        client.simPlotArrows([s], [e], arrow_size=4.0,
                             color_rgba=rgba, duration=dur, is_persistent=False)

        # Text raised above tip
        tp = airsim.Vector3r(e.x_val, e.y_val, e.z_val - 0.4)
        txt = f"{label} {dist_m:.1f}m" if dist_m else f"{label} >6m"
        client.simPlotStrings([txt], [tp], scale=6, color_rgba=rgba, duration=dur)
    except Exception:
        pass


def set_camera(pitch, yaw_off):
    try:
        pose = airsim.Pose(
            airsim.Vector3r(0, 0, 0),
            airsim.to_quaternion(math.radians(pitch), 0, math.radians(yaw_off))
        )
        client.simSetCameraPose(CAM_NAME, pose, vehicle_name=VEHICLE_NAME)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────
cam_pitch  = 0.0
cam_yaw    = 0.0
cur_yaw    = 0.0
yaw_cd     = 0.0
last_print = 0.0
DT         = 0.04   # ~25 Hz loop

print("\n[READY]  Arrows=move  Space/Ctrl=up/dn  Q/E=yaw  Mouse=camera  Esc=quit")
print("─" * 60)

try:
    while True:
        now = time.time()

        # ── MOVEMENT ──────────────────────────────────────────────────────────
        vx = vy = vz = 0.0
        if keyboard.is_pressed('up'):    vx =  SPEED
        if keyboard.is_pressed('down'):  vx = -SPEED
        if keyboard.is_pressed('left'):  vy = -SPEED
        if keyboard.is_pressed('right'): vy =  SPEED
        if keyboard.is_pressed('space'): vz = -3.5
        if keyboard.is_pressed('ctrl'):  vz =  3.5
        client.moveByVelocityBodyFrameAsync(vx, vy, vz, DT * 1.5)

        # ── YAW (smooth while held) ───────────────────────────────────────────
        if now > yaw_cd:
            if keyboard.is_pressed('q'):
                client.rotateByYawRateAsync(-YAW_RATE, DT).join()
                yaw_cd = now + DT
            elif keyboard.is_pressed('e'):
                client.rotateByYawRateAsync( YAW_RATE, DT).join()
                yaw_cd = now + DT

        # ── CAMERA (mouse) ────────────────────────────────────────────────────
        mdx, mdy = consume_mouse()
        if abs(mdx) > 0.5 or abs(mdy) > 0.5:
            cam_yaw   += mdx * MOUSE_SENS
            cam_pitch += mdy * MOUSE_SENS
            cam_pitch  = max(-80.0, min(80.0,  cam_pitch))
            cam_yaw    = max(-160.0, min(160.0, cam_yaw))
            set_camera(cam_pitch, cam_yaw)

        # ── TELEMETRY ─────────────────────────────────────────────────────────
        if now - last_print > PRINT_HZ:
            pos, yaw   = get_state()
            lidar_d, _ = front_lidar()
            seg        = seg_center()
            _, gdist   = get_goal_dist(pos)

            # Resolve label priority:
            #   1. seg label if it's a known named thing  (tree, survivor, cylinder...)
            #   2. "id-NN" printed as "?id-NN" → tells you to set stencil in Unreal
            #   3. lidar hit without identity → "obstacle"
            #   4. nothing → "clear"
            if seg and not seg.startswith("id-") and seg not in ("air", "ground"):
                label = seg
            elif seg and seg.startswith("id-"):
                label = f"?{seg}"    # prompt: go set this stencil in Unreal
            elif lidar_d:
                label = "obstacle"
            else:
                label = "clear"

            draw_ray(pos, yaw, cam_yaw, lidar_d, label)

            fd   = f"{lidar_d:.1f}m" if lidar_d else ">6m"
            gd   = f"{gdist:.1f}m"   if gdist   else "N/A"
            warn = " ⚠"              if lidar_d and lidar_d < 1.5 else ""
            hit  = " ★FOUND"         if gdist   and gdist  < 1.5  else ""

            print(f"Front: {fd} [{label}]{warn}  |  Goal: {gd}{hit}")
            last_print = now

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