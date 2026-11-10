import airsim
import keyboard
import time
 # PX4 pro , ROS , python API
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

speed = 100

while True:
    vx = vy = vz = 0

    if keyboard.is_pressed('w'):
        vx = speed
    if keyboard.is_pressed('s'):
        vx = -speed
    if keyboard.is_pressed('a'):
        vy = -speed
    if keyboard.is_pressed('d'):
        vy = speed
    if keyboard.is_pressed('space'):
        vz = -2     # up
    if keyboard.is_pressed('shift'):
        vz = 2      # down

    client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.1)

    if keyboard.is_pressed('esc'):
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        break

    time.sleep(0.25)