import airsim
import time
import random
import math

client = airsim.MultirotorClient()
client.confirmConnection()
objtofind = "MovingCube_2"
target_name = None
objs = client.simListSceneObjects(".*")
for obj in objs:
    print(obj)
    if obj == objtofind:
        target_name = obj
        break

if not target_name:
    print(f"Could not find {objtofind} in the scene!")
else:
    print(f"Found {target_name}! Starting teleport test...")
    for i in range(10):
        # Random coordinates between -20 and 20
        rx = random.uniform(-20, 20)
        ry = random.uniform(-20, 20)
        
        print(f"Teleporting Dwight to X: {rx:.1f}, Y: {ry:.1f}")
        pose = airsim.Pose(airsim.Vector3r(rx, ry, 0), airsim.to_quaternion(0, 0, 0))
        client.simSetObjectPose(target_name, pose, True)
        
        time.sleep(3.0) # Wait 3 seconds so you can see it