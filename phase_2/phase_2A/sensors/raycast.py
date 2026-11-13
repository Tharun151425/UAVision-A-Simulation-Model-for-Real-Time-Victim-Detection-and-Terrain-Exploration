"""
Phase 2A Raycast Sensor — self-contained copy of Phase 1.5 sensor.

32 horizontal × 8 vertical rays spread across a 120° × 60° cone.
Ray codes:
  0  — air (nothing in range)
  1  — ground
  2  — obstacle (cylinder, box, tree, rock, wall — ALL same code)
  3  — survivor

Tree trunks and canopies are both in the obstacle_ids set so they
get code 2 — identical to any other solid obstacle.
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
        """
        Fire all rays from (x, y, z) with current yaw.
        Returns (grid[v_rays, h_rays], ray_hits_list).
        Single p.rayTestBatch call — no physicsClientId keyword.
        """
        grid     = np.zeros((self.v_rays, self.h_rays), dtype=np.int32)
        ray_hits = []

        drone_pos = [x, y, z]
        h_angles  = np.linspace(-self.h_fov / 2, self.h_fov / 2, self.h_rays)
        v_angles  = np.linspace(-self.v_fov / 2, self.v_fov / 2, self.v_rays)

        ray_starts = []
        ray_ends   = []

        for v_ang in v_angles:
            for h_ang in h_angles:
                yaw_rad = math.radians(yaw)
                h_rad   = math.radians(h_ang)
                v_rad   = math.radians(v_ang)

                fx = math.cos(yaw_rad + h_rad) * math.cos(v_rad)
                fy = math.sin(yaw_rad + h_rad) * math.cos(v_rad)
                fz = math.sin(v_rad)

                ray_starts.append(drone_pos)
                ray_ends.append([
                    drone_pos[0] + fx * self.max_range,
                    drone_pos[1] + fy * self.max_range,
                    drone_pos[2] + fz * self.max_range,
                ])

        hits = p.rayTestBatch(ray_starts, ray_ends)

        obstacle_set = set(obstacle_ids) if obstacle_ids else set()

        idx = 0
        for vi in range(self.v_rays):
            for hi in range(self.h_rays):
                hit           = hits[idx]
                hit_object_id = hit[0]
                hit_pos       = hit[3]
                idx += 1

                if hit_object_id == -1:
                    grid[vi][hi] = 0                   # air
                elif hit_object_id == ground_id:
                    grid[vi][hi] = 1                   # ground
                    ray_hits.append((hit_pos[0], hit_pos[1], 1))
                elif hit_object_id in obstacle_set:
                    grid[vi][hi] = 2                   # obstacle (ALL shapes)
                    ray_hits.append((hit_pos[0], hit_pos[1], 2))
                elif hit_object_id == survivor_id:
                    grid[vi][hi] = 3                   # survivor
                    ray_hits.append((hit_pos[0], hit_pos[1], 3))

        return grid, ray_hits
