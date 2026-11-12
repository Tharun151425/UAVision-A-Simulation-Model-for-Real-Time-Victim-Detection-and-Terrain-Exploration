"""
Phase 2C Raycast Sensor — identical to 2A/2B.
All obstacles map to code 2. Survivor maps to code 3.
"""

import pybullet as p
import math
import numpy as np


class RaycastSensor:
    def __init__(self, h_rays=32, v_rays=8, h_fov=120, v_fov=60, max_range=20.0):
        self.h_rays    = h_rays
        self.v_rays    = v_rays
        self.h_fov     = h_fov
        self.v_fov     = v_fov
        self.max_range = max_range

    def fire(self, x, y, z, yaw,
             ground_id=None, obstacle_ids=None, survivor_id=None,
             physics_client=0):
        grid      = np.zeros((self.v_rays, self.h_rays), dtype=np.int32)
        ray_hits  = []
        drone_pos = [x, y, z]
        h_angles  = np.linspace(-self.h_fov / 2, self.h_fov / 2, self.h_rays)
        v_angles  = np.linspace(-self.v_fov / 2, self.v_fov / 2, self.v_rays)

        ray_starts, ray_ends = [], []
        for v_ang in v_angles:
            for h_ang in h_angles:
                yaw_rad = math.radians(yaw)
                h_rad   = math.radians(h_ang)
                v_rad   = math.radians(v_ang)
                fx = math.cos(yaw_rad + h_rad) * math.cos(v_rad)
                fy = math.sin(yaw_rad + h_rad) * math.cos(v_rad)
                fz = math.sin(v_rad)
                ray_starts.append(drone_pos)
                ray_ends.append([drone_pos[0] + fx * self.max_range,
                                  drone_pos[1] + fy * self.max_range,
                                  drone_pos[2] + fz * self.max_range])

        hits         = p.rayTestBatch(ray_starts, ray_ends)
        obstacle_set = set(obstacle_ids) if obstacle_ids else set()

        idx = 0
        for vi in range(self.v_rays):
            for hi in range(self.h_rays):
                hit = hits[idx]; idx += 1
                hid = hit[0]; hpos = hit[3]
                if hid == -1:
                    grid[vi][hi] = 0
                elif hid == ground_id:
                    grid[vi][hi] = 1
                    ray_hits.append((hpos[0], hpos[1], 1))
                elif hid in obstacle_set:
                    grid[vi][hi] = 2
                    ray_hits.append((hpos[0], hpos[1], 2))
                elif hid == survivor_id:
                    grid[vi][hi] = 3
                    ray_hits.append((hpos[0], hpos[1], 3))
        return grid, ray_hits
