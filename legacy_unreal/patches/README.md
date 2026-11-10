# Unreal CameraDirector Patch

These two files are the project-specific modifications to Microsoft AirSim's
`ACameraDirector` actor. They add:

- a spring-arm chase camera with mouse orbit,
- mouse capture on `BeginPlay`,
- `4` / `5` bindings for FPV and third-person view,
- a small `Documents/AirSim/camera_state.txt` file containing orbit yaw/pitch,
  which `drone_control4.py` reads for camera-relative movement.

## Apply

Copy both files over the matching files in an AirSim source checkout:

```text
AirSim/Unreal/Plugins/AirSim/Source/CameraDirector.h
AirSim/Unreal/Plugins/AirSim/Source/CameraDirector.cpp
```

Then rebuild the Unreal project. The patch is tied to the AirSim plugin version
used for this prototype (1.8.x); if your checkout differs, merge the changes
rather than replacing the files.

The patch is **optional**. The Python scripts work without it; you simply lose
the GTA-style orbit camera and `camera_state.txt` feedback.
