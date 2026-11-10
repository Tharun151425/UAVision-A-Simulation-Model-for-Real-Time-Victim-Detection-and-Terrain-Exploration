# Legacy branch README

> **Archived first-generation UAVision prototype.**
>
> This branch is preserved for provenance. It contains the original manual
> AirSim flight controller, segmentation-based object detection experiments,
> the Unreal `CameraDirector` camera patch, and sample mapping output.
>
> The maintained project lives on `main` (AirSim DQN + PyBullet SAR curriculum).
> See [`legacy_unreal/README.md`](legacy_unreal/README.md) for setup and file
> documentation.

## Quick context

The first version answered a simple question: *can a drone in Unreal Engine
identify what it is looking at and how far away it is?* The answer was yes,
using AirSim's segmentation buffer, CustomDepth stencil values, LiDAR, and a
Python HUD. The prototype was manually flown. It produced the object registry
and occupancy-grid utilities that later informed the autonomous search agent.

## What is in this branch

- Manual flight: `WASD`, mouse orbit, FPV/TPV toggle.
- Crosshair object identification with stencil labels.
- LiDAR range and 360° obstacle sensing.
- Automatic screenshots of newly seen objects.
- 2-D occupancy grid + object registry (`rl_utils.py`).
- The C++ camera patch used by `drone_control4.py`.
- The original experiment notes and sample output.

## Important limitation

This is not the final autonomous system. There is no learned policy on this
branch, and the Unreal `.uproject`/assets are intentionally not distributed.
Use `main` for the reinforcement-learning implementations.
