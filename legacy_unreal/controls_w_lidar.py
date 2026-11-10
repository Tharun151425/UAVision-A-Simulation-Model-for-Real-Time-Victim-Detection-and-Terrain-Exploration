import airsim
import keyboard
import time
import math

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

speed = 3
last_print = 0

while True:
    vx = vy = vz = 0

    # Controls
    if keyboard.is_pressed('w'):
        vx = speed
    if keyboard.is_pressed('s'):
        vx = -speed
    if keyboard.is_pressed('a'):
        vy = -speed
    if keyboard.is_pressed('d'):
        vy = speed
    if keyboard.is_pressed('space'):
        vz = -2      # up
    if keyboard.is_pressed('shift'):
        vz = 2       # down

    client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.1)

    # Print every 0.5 sec
    now = time.time()
    if now - last_print > 0.5:

        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position

        # Lidar
        lidar = client.getLidarData("Lidar1", "Drone1")
        pts = lidar.point_cloud

        front_dist = float("inf")

        for i in range(0, len(pts), 3):
            x = pts[i]
            y = pts[i+1]
            z = pts[i+2]

            # distance
            d = math.sqrt(x*x + y*y + z*z)

            # horizontal angle of point
            ang = math.degrees(math.atan2(y, x))

            # keep only front cone ±20 degrees
            if -20 <= ang <= 20:
                front_dist = min(front_dist, d)

        fd = "INF" if front_dist == float("inf") else f"{round(front_dist,2)} m"

        print(
            f"X:{round(pos.x_val,1)} | "
            f"Y:{round(pos.y_val,1)} | "
            f"Z:{round(pos.z_val,1)} | "
            f"Front Obj:{fd}"
        )

        last_print = now

    if keyboard.is_pressed('esc'):
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        break

    time.sleep(0.25)