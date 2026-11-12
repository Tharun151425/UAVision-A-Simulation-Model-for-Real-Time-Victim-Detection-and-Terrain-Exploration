# Unreal CameraDirector Patch (AirSim RL branch)

These files are the project-specific variants of Microsoft AirSim's
`ACameraDirector` actor. They are **optional** and only affect the human-facing
camera, not the RL environment.

The patch provides:

- a spring-arm third-person camera suitable for manual inspection,
- view-mode helpers used while debugging the Blocks scene,
- a consistent camera configuration between the manual-flight tools and the
  RL demo.

## Apply

Copy over the matching files in an AirSim source checkout:

```text
AirSim/Unreal/Plugins/AirSim/Source/CameraDirector.h
AirSim/Unreal/Plugins/AirSim/Source/CameraDirector.cpp
```

Then rebuild the Unreal project. The patch targets AirSim 1.8.x; merge the
changes if your checkout differs. The RL scripts do not require this patch —
they use AirSim's standard camera APIs.

For the more elaborate legacy GTA-style orbit camera, see the
`legacy/old-unreal` branch (`legacy_unreal/patches/`).
