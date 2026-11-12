"""
Shared Fog-of-War Map
======================
Used by ALL GRU phases (Phase 1 through Phase 2C).

Grid values:
  -1 = unknown  (not yet visited by drone)
   0 = visited

The patch extraction method returns a flat 256-element array (16×16)
centred on the drone.  Out-of-map cells return -1.0 (unknown).
This value is stable regardless of map size, so the obs vector is
always exactly 519 values across every phase.
"""

import numpy as np


class FogOfWarMap:

    def __init__(self, world_size: float = 50.0, cell_size: float = 2.0):
        self.world_size = world_size
        self.cell_size  = cell_size
        self.n          = int(world_size / cell_size)   # cells per side
        self.grid       = np.full((self.n, self.n), -1, dtype=np.int8)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        self.grid[:] = -1

    # ── Cell conversion ───────────────────────────────────────────────────────

    def world_to_cell(self, x: float, y: float):
        cx = int((x + self.world_size / 2) / self.cell_size)
        cy = int((y + self.world_size / 2) / self.cell_size)
        cx = int(np.clip(cx, 0, self.n - 1))
        cy = int(np.clip(cy, 0, self.n - 1))
        return cx, cy

    # ── Visit ─────────────────────────────────────────────────────────────────

    def visit(self, x: float, y: float) -> bool:
        """Mark current cell as visited. Returns True if it was unknown."""
        cx, cy = self.world_to_cell(x, y)
        was_unknown      = (self.grid[cy][cx] == -1)
        self.grid[cy][cx] = 0
        return was_unknown

    # ── Query ─────────────────────────────────────────────────────────────────

    def is_known(self, x: float, y: float) -> bool:
        cx, cy = self.world_to_cell(x, y)
        return self.grid[cy][cx] != -1

    def total_unknown(self) -> int:
        return int(np.sum(self.grid == -1))

    def total_visited(self) -> int:
        return int(np.sum(self.grid == 0))

    # ── Patch extraction ──────────────────────────────────────────────────────

    def get_patch(self, x: float, y: float, patch_size: int = 16) -> np.ndarray:
        """
        Return a flat float32 array of length patch_size² centred on (x, y).
        Values: -1.0 (unknown / outside map) or 0.0 (visited).
        patch_size must be even.  Default 16 → 256 elements.
        """
        cx, cy = self.world_to_cell(x, y)
        half   = patch_size // 2
        patch  = np.full(patch_size * patch_size, -1.0, dtype=np.float32)

        idx = 0
        for dy in range(-half, half):
            for dx in range(-half, half):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < self.n and 0 <= ny < self.n:
                    patch[idx] = float(self.grid[ny][nx])
                idx += 1

        return patch
