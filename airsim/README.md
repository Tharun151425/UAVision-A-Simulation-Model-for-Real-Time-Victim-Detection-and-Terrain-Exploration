# AirSim RL Agent

This module is the project-specific AirSim layer of UAVision. It contains a
Gymnasium environment, a DQN training pipeline, an inference/demo player, a
live perception HUD, and the configuration used with Unreal Engine.

> **The Unreal project is not distributed.** Only the reusable project code,
> configuration, and one baseline checkpoint are included. You need a working
> Microsoft AirSim + Unreal Engine environment to run the scripts. See
> [`../docs/SETUP.md`](../docs/SETUP.md).

---

## Why this layer exists

The legacy prototype on `legacy/old-unreal` proved that AirSim can provide the
perception and flight primitives. The next question was whether an agent could
learn to search on its own. This module turns the simulator into a standard
Gymnasium environment so a DQN agent can learn a target-search policy from:

- an **84×84 depth image** (what is in front),
- **8 LiDAR sector distances** (what is around the drone),
- a **3-D target-direction vector** (where the target is relative to the drone).

The agent selects one of five FPV-realistic actions: forward, ascend, descend,
yaw left, or yaw right. It cannot strafe or fly backwards, which forces it to
turn and face its direction of travel.

---

## Files

| File | Purpose |
| --- | --- |
| `setup_path.py` | Makes the AirSim Python package importable. |
| `rl_drone_env.py` | Single-target Gymnasium environment: observation, action, reward, reset/step logic. |
| `rl_drone_env_multi.py` | Multi-target variant that requires all targets to be found. |
| `rl_train.py` | Resumable DQN training, evaluation callback, checkpointing, TensorBoard logs. |
| `rl_train_multi.py` | Multi-target training variant. |
| `rl_play.py` | Loads a checkpoint and runs a live demo with a LiDAR minimap. |
| `rl_play_phase.py` | Bridges an older PyBullet PPO policy into AirSim for visualisation. |
| `eval.py` | Runs evaluation episodes and produces reward/step charts. |
| `drone_full.py` | Manual GTA-style flight + live HUD + segmentation detection. |
| `drone_gta_controls.py` | Standalone camera-relative manual controller. |
| `drone_vision.py` | Crosshair object identification and distance display. |
| `exit_screen.py` | Optional end-of-training screen. |
| `moving_test.py` | Teleports a named target around the scene to test detection. |
| `config/settings.json` | AirSim sensor configuration used by the project. |
| `rl_best/best_model.zip` | Curated baseline DQN checkpoint. |
| `rl_best/evaluations.npz` | Evaluation history for that checkpoint. |
| `patches/` | Optional Unreal camera patch (see its README). |

Operator guides are in [`docs/`](docs/):

- [`docs/GUIDE.md`](docs/GUIDE.md) — full setup, training, minimap, and troubleshooting.
- [`docs/RL_GUIDE.md`](docs/RL_GUIDE.md) — v2 → v3 engineering changes.
- [`docs/RL_PROJECT_EXPLAINED.md`](docs/RL_PROJECT_EXPLAINED.md) — interview-oriented deep dive.

---

## Quick start

### 1. Install dependencies

```bash
cd airsim
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

AirSim's Python client is installed by `requirements.txt`. If you prefer the
client from an AirSim source checkout, run from that checkout or put its
`PythonClient` folder on `PYTHONPATH`.

### 2. Configure AirSim

```bash
# Windows
copy config\settings.json "%USERPROFILE%\Documents\AirSim\settings.json"
# Linux/macOS
mkdir -p ~/Documents/AirSim && cp config/settings.json ~/Documents/AirSim/settings.json
```

The configuration enables:

- a forward LiDAR (`FrontLidar`),
- a 16-channel surround LiDAR (`SurroundLidar`),
- a forward distance sensor (`FrontDistance`),
- 128×96 RGB, depth, and segmentation captures,
- `ClockSpeed: 3.0` for faster-than-real-time training.

### 3. Prepare the Unreal scene

1. Open the stock AirSim **Blocks** environment (or another AirSim-enabled map).
2. Add the target actor named `Dwight` (or change `TARGET_LABEL` in
   `rl_drone_env.py` to your target's name).
3. Add several `Cube` actors as obstacles; the environment randomises them.
4. Press **Play** and leave the simulator running.
5. Optionally minimise the Unreal window for a 2–3× training speed-up.

### 4. Train

```bash
python rl_train.py
```

Training is resumable: stop with `Ctrl+C` and run the same command again. The
script writes:

```text
tb_logs/       TensorBoard event files (ignored by git)
rl_models/     periodic checkpoints (ignored by git)
rl_best/       best evaluation checkpoint + evaluations.npz
```

Monitor with:

```bash
tensorboard --logdir tb_logs
```

### 5. Run the demo

```bash
python rl_play.py
python rl_play.py --episodes 10 --no-render
python rl_play.py --model rl_best/best_model.zip
```

Demo controls: `M` toggles the minimap, `R` resets an episode, `Q`/`ESC` quits.

---

## Observation, action, and reward contract

### Observation

```python
{
    "image":      Box(0, 255, (84, 84, 1), uint8),   # depth
    "lidar":      Box(0, 100, (8,), float32),        # min distance per sector
    "target_dir": Box(-1, 1, (3,), float32),         # relative unit vector
}
```

### Action (Discrete 5)

| ID | Action | Effect |
| ---: | --- | --- |
| 0 | Forward | 5 m along the current heading |
| 1 | Up | +2 m altitude |
| 2 | Down | −2 m altitude (floor at 2 m) |
| 3 | Yaw left | −45° |
| 4 | Yaw right | +45° |

### Reward (single-target v3)

| Event | Reward |
| --- | ---: |
| Target found | +200 |
| Ground contact | −200 |
| Obstacle collision | −100 |
| Out of bounds | −50 |
| Timeout | −20 |
| LiDAR proximity warning | −5 |
| Per-step cost | −0.3 |
| Progress/target-heading shaping | small positive term |

The full implementation and comments are in `rl_drone_env.py`. Reward shaping
is intentionally explicit so it can be ablated.

---

## Reproducibility notes

- Environment resets use domain randomisation: drone spawn, target position,
  obstacle positions, and yaw are sampled per episode.
- A short grace period after teleport prevents false collision flags.
- The altitude floor is enforced after every action.
- The training script is resumable and keeps the best evaluation checkpoint.
- Hyperparameters are module-level constants in `rl_train.py`.
- A baseline checkpoint is included so the demo can be run without retraining;
  it is a **research baseline**, not a claim of production performance. See
  [`rl_best/README.md`](rl_best/README.md) for the model card.

## Current status and limitations

- The included policy was trained for a single target in the Blocks scene; it
  is sensitive to scene layout and camera/LiDAR configuration.
- The evaluation history shows negative mean rewards, i.e. the agent often
  times out. It demonstrates the end-to-end pipeline and is a starting point
  for longer training, reward ablation, and better exploration.
- Detection uses simulator-ground-truth segmentation, not a learned detector.
- AirSim's Python client and Unreal rendering are the bottleneck; PyBullet is
  used on `main` for large-scale curriculum experiments.
- This code is for simulation research only. Do not use it to control real
  aircraft.

## License and attribution

Project code is MIT licensed. Microsoft AirSim is a separate MIT-licensed
project; its source and Unreal environment are not redistributed here. See the
repository [`LICENSE`](../LICENSE).
