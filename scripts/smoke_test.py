"""Lightweight smoke test for the UAVision PyBullet stack.

Run from the repository root:

    python scripts/smoke_test.py

The test does not train anything. It verifies that the shared perception
utilities import cleanly and that the Phase-2 environments honour their
observation-space contract for one reset/step cycle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_shared_utilities() -> None:
    from sensors.raycast_gru import RaycastSensor
    from utils.fog_map import FogOfWarMap

    sensor = RaycastSensor()
    assert sensor.h_rays == 32
    assert sensor.v_rays == 8
    assert sensor.max_range == 20.0

    fog = FogOfWarMap(world_size=50.0, cell_size=2.0)
    assert fog.visit(0.0, 0.0) is True
    assert fog.visit(0.0, 0.0) is False
    patch = fog.get_patch(0.0, 0.0)
    assert patch.shape == (256,)
    print("[PASS] shared raycast sensor + fog-of-war map")


def check_phase2a() -> None:
    from phase_2.phase_2A.envs.phase2A_env import Phase2AEnv

    env = Phase2AEnv(render=False, curriculum_level=1)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (262,), obs.shape
        assert env.action_space.n == 6
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert obs.shape == (262,), obs.shape
            assert isinstance(float(reward), float)
            if terminated or truncated:
                break
        print("[PASS] Phase 2A environment reset/step (262-dim observation)")
    finally:
        env.close()


def check_phase2a_gru() -> None:
    from phase_2.phase_2A.envs.phase2A_env_gru import Phase2AEnvGRU

    env = Phase2AEnvGRU(render=False, curriculum_level=1)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (519,), obs.shape
        assert env.action_space.n == 6
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (519,), obs.shape
        print("[PASS] Phase 2A GRU environment reset/step (519-dim observation)")
    finally:
        env.close()


def main() -> None:
    print("UAVision smoke test")
    print("=" * 50)
    check_shared_utilities()
    check_phase2a()
    check_phase2a_gru()
    print("=" * 50)
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
