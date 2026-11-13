# Screenshot drop zone

Put project screenshots in this folder. The main README embeds the files below
by name. Files currently in this folder are **labelled placeholders**; replace
each one with a real capture and keep the filename unchanged.

## Required files

| File | What to capture | Suggested source |
| --- | --- | --- |
| `hero-airsim-flight.png` | Wide 16:9 hero frame: drone in the Unreal Blocks scene with the flight HUD visible. | AirSim manual demo |
| `manual-flight-hud.png` | The GTA-style camera-relative controller and live perception HUD. | `python airsim/drone_full.py` |
| `segmentation-detection.png` | Crosshair detection with the victim label, confidence, and distance readout. | `python airsim/drone_vision.py` |
| `airsim-training.png` | TensorBoard reward/loss curves, or a clean terminal showing a resumable DQN run. | `tensorboard --logdir airsim/tb_logs` |
| `airsim-demo-minimap.png` | The autonomous demo with LiDAR minimap, target marker, and reward overlay. | `python airsim/rl_play.py` |
| `pybullet-phase2a.png` | Phase 2A: survivor visible, obstacles, and ray fan. | `Phase2AEnv(render=True)` |
| `pybullet-phase2b.png` | Phase 2B: survivor partially occluded behind an obstacle. | `Phase2BEnv(render=True)` |
| `pybullet-phase2c.png` | Phase 2C: larger map, fog coverage, and a search trajectory. | `Phase2CEnvGRU(render=True)` |
| `evaluation-rewards.png` | Episode-reward chart from `airsim/rl_best/evaluations.npz` or a stronger run. | `airsim/eval.py` |

## Capture rules

- Use 16:9, ideally 1600x900 or 1920x1080. The README cards crop to 16:9.
- Keep text readable at GitHub's content width (about 880 px).
- Do not include personal information, local file paths, API keys, or email.
- Do not stage simulator build output, `.uproject`, `.uasset`, or `.umap` files.
- Compress PNGs; keep each file below roughly 2 MB.
- Real results only. Do not edit a screenshot to imply a better policy than the
  logs support. If a run is a baseline, label it as a baseline.

## After adding files

```bash
git add docs/screenshots/*.png
git commit -m "docs(media): add simulation and evaluation screenshots"
git push
```

The README image links are already wired to these paths, so replacing a
placeholder with the same filename is enough. If you rename a file, update the
corresponding `<img src="...">` in the root `README.md` and this table.
