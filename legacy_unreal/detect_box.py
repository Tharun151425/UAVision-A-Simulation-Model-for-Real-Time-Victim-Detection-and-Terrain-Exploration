"""
AirSim Drone Controller
=======================
Controls  : W/S = forward/back | A/D = left/right
            Space = up          | Shift = down
            Q/E = yaw left/right (15 deg snap)
            Esc = land & exit

Features  : Front object detection with label
            Unreal debug ray (5 m forward, visible in viewport)
            Live dashboard: what object, at what distance
"""

import airsim
import keyboard
import time
import math
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
SPEED           = 5          # m/s translation speed
YAW_STEP        = 15         # degrees per Q/E press
FRONT_CONE_DEG  = 20         # ± degrees for "front" cone from LiDAR
DEBUG_RAY_M     = 5.0        # length of Unreal viewport debug ray
PRINT_HZ        = 0.4        # seconds between dashboard prints
LIDAR_NAME      = "Lidar1"
VEHICLE_NAME    = "Drone1"

# Segmentation stencil → human-readable label
STENCIL_LABELS = {
    0:   "air",
    1:   "ground",
    10:  "tree",
    11:  "rock",
    20:  "survivor-box",
    255: "unknown",
}

SURVIVOR_PATTERN = "Cube.*"

# ── CONNECT ───────────────────────────────────────────────────────────────────
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# ── FIND SURVIVOR OBJECTS ─────────────────────────────────────────────────────
scene_objects = client.simListSceneObjects(SURVIVOR_PATTERN)
if not scene_objects:
    print("[WARN] No objects matching '{}' found.".format(SURVIVOR_PATTERN))
    goal_name = None
else:
    goal_name = scene_objects[0]
    print(f"[INFO] Tracking target: {goal_name}")
    print(f"[INFO] All Cube actors in scene: {scene_objects}")

# ── SEGMENTATION SETUP ────────────────────────────────────────────────────────
# FIX 1: use raw string r"[\w]*" to avoid SyntaxWarning on \w
client.simSetSegmentationObjectID(r"[\w]*", 0, True)   # reset all to 0

for cube_name in scene_objects:
    client.simSetSegmentationObjectID(cube_name, 20, False)

client.simSetSegmentationObjectID(r"SM_Tree.*",  10, True)
client.simSetSegmentationObjectID(r"SM_Rock.*",  11, True)
client.simSetSegmentationObjectID(r"Landscape*",  1, True)

print("[INFO] Segmentation IDs assigned.")

# ── TAKEOFF ───────────────────────────────────────────────────────────────────
print("Taking off...")
client.takeoffAsync().join()
time.sleep(1)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_drone_pos_yaw():
    state = client.getMultirotorState()
    pos   = state.kinematics_estimated.position
    ori   = state.kinematics_estimated.orientation
    q     = ori
    siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw_deg   = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return pos, yaw_deg


def get_front_lidar(yaw_deg):
    lidar = client.getLidarData(LIDAR_NAME, VEHICLE_NAME)
    pts   = lidar.point_cloud
    front_dist = float("inf")

    for i in range(0, len(pts) - 2, 3):
        x, y, z = pts[i], pts[i+1], pts[i+2]
        dist = math.sqrt(x*x + y*y + z*z)
        if dist < 0.1:
            continue
        ang = math.degrees(math.atan2(y, x))
        if -FRONT_CONE_DEG <= ang <= FRONT_CONE_DEG:
            if dist < front_dist:
                front_dist = dist

    if front_dist == float("inf"):
        return None, "nothing"
    return front_dist, "obstacle"


def get_front_seg_label():
    try:
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
        ])
        if not responses or responses[0].width == 0:
            return "unknown"

        resp    = responses[0]
        img     = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img     = img.reshape(resp.height, resp.width, 3)
        cy, cx  = resp.height // 2, resp.width // 2
        stencil = int(img[cy, cx, 2])
        return STENCIL_LABELS.get(stencil, f"mesh-id-{stencil}")
    except Exception:
        return "unknown"


def draw_debug_ray(drone_pos, yaw_deg, dist_m, label):
    """
    Draw a colored arrow + text label in the Unreal viewport.
    FIX 2: simPlotStrings() does NOT accept is_persistent kwarg — removed it.
    """
    yaw_rad = math.radians(yaw_deg)
    fwd_x   = math.cos(yaw_rad)
    fwd_y   = math.sin(yaw_rad)

    start = airsim.Vector3r(drone_pos.x_val, drone_pos.y_val, drone_pos.z_val)
    end   = airsim.Vector3r(
        drone_pos.x_val + fwd_x * DEBUG_RAY_M,
        drone_pos.y_val + fwd_y * DEBUG_RAY_M,
        drone_pos.z_val
    )

    if dist_m is None or dist_m > 3.0:
        rgba = [0.0, 1.0, 0.0, 1.0]    # green — clear
    elif dist_m > 1.5:
        rgba = [1.0, 0.8, 0.0, 1.0]    # yellow — caution
    else:
        rgba = [1.0, 0.1, 0.1, 1.0]    # red — danger

    duration = PRINT_HZ + 0.1

    # Arrow — this one supports is_persistent, but we just use duration
    client.simPlotArrows(
        [start], [end],
        arrow_size=8.0,
        color_rgba=rgba,
        duration=duration,
        is_persistent=False
    )

    # FIX 2: simPlotStrings signature is (strings, positions, scale, color_rgba, duration)
    # NO is_persistent parameter — removed
    dist_str = f"{dist_m:.1f}m" if dist_m else f">{DEBUG_RAY_M:.0f}m"
    client.simPlotStrings(
        [f"{label.upper()} {dist_str}"],
        [end],
        scale=12,
        color_rgba=rgba,
        duration=duration
    )


def get_goal_info(drone_pos):
    if not goal_name:
        return None, None
    try:
        pose = client.simGetObjectPose(goal_name)
        g    = pose.position
        dx   = g.x_val - drone_pos.x_val
        dy   = g.y_val - drone_pos.y_val
        dz   = g.z_val - drone_pos.z_val
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        return g, dist
    except Exception:
        return None, None


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
current_yaw  = 0.0
yaw_cooldown = 0.0
last_print   = 0.0

print("\n[READY] Controls: WASD=move | Space/Shift=up/down | Q/E=turn | Esc=land")
print("=" * 60)

try:
    while True:
        now = time.time()
        vx = vy = vz = 0.0

        if keyboard.is_pressed('w'):     vx =  SPEED
        if keyboard.is_pressed('s'):     vx = -SPEED
        if keyboard.is_pressed('a'):     vy = -SPEED
        if keyboard.is_pressed('d'):     vy =  SPEED
        if keyboard.is_pressed('space'): vz = -2.5
        if keyboard.is_pressed('shift'): vz =  2.5

        # Q/E yaw with debounce
        if now > yaw_cooldown:
            if keyboard.is_pressed('q'):
                current_yaw -= YAW_STEP
                yaw_cooldown = now + 0.25
                client.rotateToYawAsync(current_yaw, timeout_sec=1, margin=2).join()
            elif keyboard.is_pressed('e'):
                current_yaw += YAW_STEP
                yaw_cooldown = now + 0.25
                client.rotateToYawAsync(current_yaw, timeout_sec=1, margin=2).join()

        client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.1)

        if now - last_print > PRINT_HZ:
            drone_pos, drone_yaw = get_drone_pos_yaw()
            lidar_dist, _        = get_front_lidar(drone_yaw)
            seg_label            = get_front_seg_label()
            goal_pos, goal_dist  = get_goal_info(drone_pos)

            if lidar_dist is not None and seg_label not in ("air", "ground", "nothing", "unknown"):
                front_label = seg_label
            elif lidar_dist is not None:
                front_label = "obstacle"
            else:
                front_label = "clear"

            draw_debug_ray(drone_pos, drone_yaw, lidar_dist, front_label)

            print("=" * 60)
            print(f"  DRONE      : ({drone_pos.x_val:+.1f}, {drone_pos.y_val:+.1f}, {drone_pos.z_val:+.1f})  YAW: {drone_yaw:.0f}°")

            if goal_pos:
                print(f"  GOAL BOX   : ({goal_pos.x_val:+.1f}, {goal_pos.y_val:+.1f}, {goal_pos.z_val:+.1f})")
                print(f"  GOAL DIST  : {goal_dist:.2f} m  [{goal_name}]")
            else:
                print("  GOAL BOX   : not found")

            if lidar_dist:
                fill     = max(0, int(20 * (1 - lidar_dist / DEBUG_RAY_M)))
                bar      = "█" * fill + "░" * (20 - fill)
                warn     = "  ⚠ CLOSE!" if lidar_dist < 1.5 else ""
                print(f"  FRONT OBJ  : [{bar}] {lidar_dist:.2f} m  → {front_label.upper()}{warn}")
            else:
                print(f"  FRONT OBJ  : [░░░░░░░░░░░░░░░░░░░░]  >{DEBUG_RAY_M:.0f}m  → CLEAR")

            if goal_dist is not None and goal_dist < 1.5:
                print("\n  *** SURVIVOR REACHED! ***")

            print("=" * 60)
            last_print = now

        if keyboard.is_pressed('esc'):
            print("\n[INFO] Landing...")
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n[INFO] Interrupted.")
    client.enableApiControl(False)