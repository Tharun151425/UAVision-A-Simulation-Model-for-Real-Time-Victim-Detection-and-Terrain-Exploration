# Legacy Prototype — Manual AirSim Flight, Perception & Mapping

> **Archived first generation.** This branch preserves the original UAVision
> prototype: a human-piloted drone in Unreal Engine/AirSim with live
> segmentation-based object identification, LiDAR ranging, automatic object
> screenshots, and a 2-D occupancy map. It is intentionally kept separate from
> the reinforcement-learning stack on `main`.

The prototype was built on **Microsoft AirSim 1.8.1** and the Unreal Engine
**Blocks** environment. The original Unreal project and `.uproject` files are
not distributed here (see [Why no Unreal project?](#why-no-unreal-project)).
Everything that is project-specific Python or C++ is included.

---

## What this generation proved

- AirSim can be driven from Python with low-latency, camera-relative manual
  flight.
- Unreal's **CustomDepth stencil buffer** can label scene actors (tree, rock,
  cylinder, survivor) without training a detector.
- A forward-facing LiDAR cone provides a usable crosshair range reading, and a
  360° LiDAR supports a top-down occupancy grid.
- A GTA-style spring-arm camera can be implemented in the AirSim Unreal plugin
  and its orbit state can be shared with Python through a small text file.
- The resulting object log and occupancy maps are a useful baseline for the
  later autonomous search agent.

The limitation was equally clear: manual flight does not scale, and
segmentation labels are simulator ground truth rather than a learned perception
model. Those observations motivated the AirSim DQN work and then the
PyBullet curriculum on `main`.

---

## File map

| File | Purpose |
| --- | --- |
| `controls_test.py` | Minimal velocity-control smoke test (`WASD`, `Space`, `Shift`, `Esc`). |
| `controls_w_lidar.py` | Same manual control plus a forward LiDAR cone readout. |
| `detect_box.py` | First detection prototype: crosshair, segmentation ID, distance. |
| `detect_box2.py` | Improved controller: camera-relative movement, mouse look, stencil labels. |
| `drone_control3.py` | GTA-style iteration using `pynput`; FPV/TPV toggle, sky filter. |
| `drone_control4.py` | Final prototype: C++ orbit camera, camera-relative `WASD`, 5×5 segmentation scan, auto-screenshots. |
| `gemini_control4.py` | Experimental variant of v4 produced while exploring an LLM-assisted control loop. |
| `rl_utils.py` | `OccupancyGrid` and `ObjectRegistry` used by v4 for mapping and object logging. |
| `settings2.json` / `settings3.json` | LiDAR + camera configurations used by the early iterations. |
| `settings4.json` | Final configuration: 16-channel 360° LiDAR and RGB/segmentation camera. |
| `patches/CameraDirector.*` | Unreal C++ patch for spring-arm orbit, mouse capture, FPV/TPV keys. |
| `sample_outputs/` | Real output from a session: object log and occupancy arrays. |
| `docs/EXPERIMENT_NOTES.md` | Original experiment log comparing the iterations. |

---

## Setup

### 1. Python environment

```bash
cd legacy_unreal
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. AirSim settings

Copy the configuration that matches the script you want to run into the AirSim
settings folder:

```text
settings2.json -> %USERPROFILE%\Documents\AirSim\settings.json
settings3.json -> %USERPROFILE%\Documents\AirSim\settings.json
settings4.json -> %USERPROFILE%\Documents\AirSim\settings.json
```

`settings4.json` is the recommended starting point for `drone_control4.py`
because it enables the 16-channel surround LiDAR and both RGB and segmentation
captures.

### 3. Unreal environment

1. Install Unreal Engine and build Microsoft AirSim's Blocks environment
   following the official AirSim documentation.
2. Add the actors you want to detect (tree, rock, cylinder, survivor mesh).
3. For stencil-based labels, enable **Render CustomDepth Pass** on each actor
   and set **Custom Depth Stencil Value** to the IDs in the table below.
4. If you want the GTA-style orbit camera, apply
   `patches/CameraDirector.cpp` and `patches/CameraDirector.h` to
   `AirSim/Unreal/Plugins/AirSim/Source/` and rebuild the plugin.
5. Start the Blocks game / PIE session and press **Play**.

### Stencil IDs used by `drone_control4.py`

| ID | Label |
| ---: | --- |
| 0 | air |
| 1 | ground |
| 6 | sky (filtered) |
| 10 | tree |
| 11 | rock |
| 15 | cylinder |
| 20 | survivor |

Unknown IDs are printed with their numeric value so they can be labelled in
Unreal. This is how the prototype discovered new scene objects.

### 4. Run

```bash
python drone_control4.py
```

Controls:

| Input | Action |
| --- | --- |
| `W A S D` | Move relative to the camera direction |
| `Space` / `Ctrl` | Ascend / descend |
| `Q` / `E` | Yaw the drone |
| Mouse | Orbit the chase camera (patched C++) |
| `4` / `5` | FPV / third-person view |
| Arrow keys | Fine FPV camera pitch/yaw |
| `Esc` | Land and exit |

The script prints one clean line per interval, for example:

```text
Front: 3.2m [tree]  |  Goal: 9.1m [survivor]
```

New object IDs are screenshotted automatically into `detected_objects/`;
`rl_utils.py` writes `object_log.json` and the occupancy arrays.

---

## Why no Unreal project?

The Unreal environment contains built binaries, cooked assets, and
marketplace/licensed content. Redistributing it would be large, brittle, and in
some cases a licence violation. The repository therefore ships the **code,
configuration, patches, and sample outputs** only. To reproduce the original
scene:

1. Build the stock AirSim Blocks environment.
2. Apply the optional camera patch.
3. Add actors and stencil values as described above.
4. Use the included `settings*.json`.

This is also why the `.uproject` is ignored: the repository is the reusable
project layer, not a redistribution of Unreal Engine content.

---

## Known limitations

- Detection is **simulator-oracle segmentation**, not a trained vision model.
- Object naming depends on the stencil values configured in Unreal.
- The prototype is manually flown; there is no autonomous policy on this
  branch.
- The occupancy grid is 2-D and uses LiDAR returns; it does not perform loop
  closure or full SLAM.
- The C++ camera patch is optional and tied to the AirSim plugin version you
  build.

For the autonomous versions, see the `main` branch: `airsim/` contains the
DQN search agent, and `phase_2/` contains the PyBullet SAR curriculum with
recurrent policies and fog-of-war exploration.
