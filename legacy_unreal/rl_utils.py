"""
UAVision RL Utilities — Occupancy Grid + Object Registry
==========================================================
Import alongside drone_control4.py for RL training.

Usage:
    from rl_utils import OccupancyGrid, ObjectRegistry

    grid = OccupancyGrid(size_m=100, cell_m=0.5)
    registry = ObjectRegistry("object_log.json")

    # In your RL loop:
    grid.update_from_lidar(client, vehicle_name="Drone1")
    reward = grid.exploration_reward()

    registry.log_object(stencil_id=10, label="tree", distance=3.2,
                        drone_pos=(x, y, z))
"""

import json
import math
import os
import time
import numpy as np
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
#  OCCUPANCY GRID  (2D top-down map from LiDAR)
# ──────────────────────────────────────────────────────────────────────────────

class OccupancyGrid:
    """
    2D occupancy grid built from LiDAR point clouds.
    Used for:
      - Exploration reward (RL)
      - Path planning (A*)
      - World mapping visualization

    Cell values:
      0 = unknown (never seen)
      1 = free (lidar passed through)
      2 = occupied (lidar hit something)

    Coordinate system:
      Grid center = world origin (0, 0)
      X → right,  Y → forward (AirSim NED, but projected to 2D)
    """

    UNKNOWN  = 0
    FREE     = 1
    OCCUPIED = 2

    def __init__(self, size_m=100.0, cell_m=0.5):
        """
        Args:
            size_m: total grid size in metres (square)
            cell_m: cell resolution in metres
        """
        self.size_m   = size_m
        self.cell_m   = cell_m
        self.n_cells  = int(size_m / cell_m)
        self.half     = size_m / 2.0

        # Grid: (n_cells × n_cells), uint8
        self.grid = np.zeros((self.n_cells, self.n_cells), dtype=np.uint8)

        # Stats
        self._prev_explored = 0
        self._total_explored = 0

    def world_to_grid(self, wx, wy):
        """Convert world (x, y) to grid (row, col). Returns None if out of bounds."""
        col = int((wx + self.half) / self.cell_m)
        row = int((wy + self.half) / self.cell_m)
        if 0 <= row < self.n_cells and 0 <= col < self.n_cells:
            return row, col
        return None

    def update_from_lidar(self, client, lidar_name="Lidar1", vehicle_name="Drone1"):
        """
        Ingest a LiDAR sweep and update the grid.
        Marks free cells along each ray, and occupied at the hit point.
        Returns number of newly explored cells this update.
        """
        try:
            data = client.getLidarData(lidar_name, vehicle_name)
            pts = data.point_cloud
            if len(pts) < 3:
                return 0

            # Drone world position
            state = client.getMultirotorState()
            dx = state.kinematics_estimated.position.x_val
            dy = state.kinematics_estimated.position.y_val

            new_cells = 0

            for i in range(0, len(pts) - 2, 3):
                # Point in drone body frame → world frame
                px = dx + pts[i]
                py = dy + pts[i + 1]
                # pz = pts[i + 2]  # we ignore Z for 2D grid

                # Mark hit cell as occupied
                hit = self.world_to_grid(px, py)
                if hit:
                    r, c = hit
                    if self.grid[r, c] == self.UNKNOWN:
                        new_cells += 1
                    self.grid[r, c] = self.OCCUPIED

                # Bresenham-lite: mark cells along ray as free
                origin = self.world_to_grid(dx, dy)
                if origin and hit:
                    self._mark_ray_free(origin, hit)

            self._total_explored = int(np.count_nonzero(self.grid))
            delta = self._total_explored - self._prev_explored
            self._prev_explored = self._total_explored
            return delta

        except Exception:
            return 0

    def _mark_ray_free(self, start, end):
        """Mark cells along a ray as FREE (Bresenham's line algorithm)."""
        r0, c0 = start
        r1, c1 = end
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        while (r0, c0) != (r1, c1):
            if self.grid[r0, c0] == self.UNKNOWN:
                self.grid[r0, c0] = self.FREE
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dr
                c0 += sc

    def exploration_reward(self):
        """
        Returns exploration-based reward for RL:
          +1.0 per newly explored cell since last call
          Normalized by total possible cells
        """
        explored = int(np.count_nonzero(self.grid))
        delta = explored - self._prev_explored
        self._prev_explored = explored
        return delta / max(1, self.n_cells * self.n_cells) * 100.0

    def exploration_percent(self):
        """Returns percentage of grid that has been explored (free or occupied)."""
        explored = int(np.count_nonzero(self.grid))
        return explored / (self.n_cells * self.n_cells) * 100.0

    def get_stats(self):
        """Return grid statistics dict."""
        total = self.n_cells * self.n_cells
        free  = int(np.sum(self.grid == self.FREE))
        occ   = int(np.sum(self.grid == self.OCCUPIED))
        unk   = total - free - occ
        return {
            "total_cells":    total,
            "free_cells":     free,
            "occupied_cells": occ,
            "unknown_cells":  unk,
            "explored_pct":   round((free + occ) / total * 100, 2),
            "grid_size_m":    self.size_m,
            "cell_size_m":    self.cell_m,
        }

    def save_grid(self, path="occupancy_grid.npy"):
        """Save grid as numpy file for visualization / training."""
        np.save(path, self.grid)

    def save_as_image(self, path="occupancy_map.png"):
        """Save grid as a PNG image (black=unknown, white=free, red=occupied)."""
        try:
            from PIL import Image
            img = np.zeros((self.n_cells, self.n_cells, 3), dtype=np.uint8)
            img[self.grid == self.FREE]     = [255, 255, 255]  # white
            img[self.grid == self.OCCUPIED] = [255, 50, 50]    # red
            # unknown stays black
            Image.fromarray(img).save(path)
        except ImportError:
            # Fallback: save as numpy
            self.save_grid(path.replace(".png", ".npy"))


# ──────────────────────────────────────────────────────────────────────────────
#  OBJECT REGISTRY  (persistent log of all detected objects)
# ──────────────────────────────────────────────────────────────────────────────

class ObjectRegistry:
    """
    Persistent JSON registry of all detected objects.
    Tracks: stencil ID, label, first-seen time, position, distance, count.
    Auto-saves after each new object.

    Usage:
        registry = ObjectRegistry("detected_objects/object_log.json")
        registry.log_object(sid=10, label="tree", dist=3.2, pos=(1.0, 2.0, -1.0))
        registry.log_object(sid=10, label="tree", dist=5.1, pos=(4.0, 8.0, -1.0))
        # Second call increments count but doesn't duplicate entry
    """

    def __init__(self, path="detected_objects/object_log.json"):
        self.path = path
        self.objects = {}  # stencil_id → object info dict
        self._load()

    def _load(self):
        """Load existing registry from disk."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                    self.objects = {int(k): v for k, v in data.items()}
            except (json.JSONDecodeError, IOError):
                self.objects = {}

    def _save(self):
        """Save registry to disk."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.objects, f, indent=2, default=str)

    def log_object(self, sid, label, dist=None, pos=None):
        """
        Log a detected object. If already seen, increments sighting count
        and updates last_seen timestamp.

        Args:
            sid:   stencil ID (int)
            label: human-readable label (str)
            dist:  distance from drone (float, metres)
            pos:   drone position when detected (tuple of 3 floats)
        """
        now = datetime.now().isoformat()

        if sid not in self.objects:
            self.objects[sid] = {
                "stencil_id": sid,
                "label": label,
                "first_seen": now,
                "last_seen": now,
                "sighting_count": 1,
                "closest_distance_m": dist,
                "first_position": pos,
                "positions": [pos] if pos else [],
            }
            self._save()
            return True  # new object
        else:
            entry = self.objects[sid]
            entry["last_seen"] = now
            entry["sighting_count"] += 1
            if dist and (entry["closest_distance_m"] is None or dist < entry["closest_distance_m"]):
                entry["closest_distance_m"] = dist
            if pos and pos not in entry["positions"][-5:]:  # keep last 5 positions
                entry["positions"].append(pos)
                if len(entry["positions"]) > 20:
                    entry["positions"] = entry["positions"][-20:]
            # Auto-save every 10 sightings to reduce I/O
            if entry["sighting_count"] % 10 == 0:
                self._save()
            return False  # existing object

    def get_summary(self):
        """Return a summary string for terminal output."""
        if not self.objects:
            return "No objects detected yet"
        lines = [f"  {v['label']:12s}  stencil:{v['stencil_id']:3d}  "
                 f"seen:{v['sighting_count']:4d}x  closest:{v['closest_distance_m']:.1f}m"
                 for v in sorted(self.objects.values(), key=lambda x: x['stencil_id'])]
        return "\n".join(lines)

    def get_all(self):
        """Return all objects as a list of dicts (for RL state)."""
        return list(self.objects.values())

    def save(self):
        """Force save to disk."""
        self._save()


# ──────────────────────────────────────────────────────────────────────────────
#  RL STATE BUILDER  (combine everything into an RL observation)
# ──────────────────────────────────────────────────────────────────────────────

def build_rl_state(client, grid, registry, vehicle_name="Drone1"):
    """
    Build a complete RL observation dict.
    Call this each step in your RL environment.

    Returns:
        dict with keys:
            position:    (x, y, z)
            velocity:    (vx, vy, vz)
            yaw_deg:     float
            lidar_front: float or None (distance to nearest front obstacle)
            grid_stats:  dict from OccupancyGrid
            objects:     list from ObjectRegistry
            explored_pct: float
    """
    state = client.getMultirotorState()
    p = state.kinematics_estimated.position
    v = state.kinematics_estimated.linear_velocity
    q = state.kinematics_estimated.orientation

    sy = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cy = 1.0 - 2.0 * (q.y_val**2 + q.z_val**2)
    yaw = math.degrees(math.atan2(sy, cy))

    return {
        "position":     (round(p.x_val, 2), round(p.y_val, 2), round(p.z_val, 2)),
        "velocity":     (round(v.x_val, 2), round(v.y_val, 2), round(v.z_val, 2)),
        "yaw_deg":      round(yaw, 2),
        "grid_stats":   grid.get_stats() if grid else {},
        "objects":      registry.get_all() if registry else [],
        "explored_pct": grid.exploration_percent() if grid else 0.0,
    }
