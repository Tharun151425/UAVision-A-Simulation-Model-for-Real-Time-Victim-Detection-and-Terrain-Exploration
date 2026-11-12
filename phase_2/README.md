# Phase 2 - PyBullet Search-and-Rescue Curriculum

Phase 2 is the large-scale training layer of UAVision. It moves the project
from hand-flown perception demos to a reproducible reinforcement-learning
curriculum for autonomous victim search. The environments run in PyBullet, so
millions of steps can be collected without the rendering and RPC overhead of a
full Unreal Engine session.

The curriculum deliberately separates the three capabilities a search policy
needs:

1. **Recognise** what the sensor is looking at (Phase 2A).
2. **Persist** when the victim disappears behind an obstacle (Phase 2B).
3. **Explore** a larger map when the victim is hidden from the start
   (Phase 2C).

This repository contains the environment code, reward functions, training
entry points, shared perception utilities, and a smoke test. It does **not**
ship Phase-2 checkpoints or training logs. The only included policy is the
older AirSim DQN baseline in [`../airsim/rl_best/`](../airsim/rl_best/).

---

## Environment families

Each phase has two variants that share the same task but not the same network:

| Variant | Policy | Observation | Purpose |
| --- | --- | --- | --- |
| MLP | `PPO` / `MlpPolicy` | 262 | Fast feed-forward baseline. No explicit memory. |
| GRU | `RecurrentPPO` / `MlpLstmPolicy` | 519 | Adds a 16x16 fog-of-war patch so the recurrent policy can remember where it has been. |

The 262-value observation is:

```
256 raycast values (32 x 8, first-hit codes 0-3)
  5 ego values      (x, y, altitude, cos(yaw), sin(yaw))
  1 time value      (step / max_steps)
= 262
```

The 519-value observation adds the 256-cell fog patch and a terrain flag:

```
256 raycast values + 5 ego values + 256 fog values + 1 time + 1 terrain = 519
```

Ray codes are `0` air, `1` ground, `2` obstacle, and `3` survivor. A ray stops
at the first solid hit, so a survivor behind a wall is observed as an obstacle.

---

## Phase map

| Phase | Folder | Spawn visibility | Core difficulty | Default algorithm |
| --- | --- | --- | --- | --- |
| 2A | `phase_2A/` | Survivor always visible | Shape and obstacle recognition | PPO (MLP or GRU) |
| 2B | `phase_2B/` | Visible at spawn, may be occluded later | Re-acquisition and memory | PPO / RecurrentPPO |
| 2C | `phase_2C/` | May be fully hidden | Systematic exploration | RecurrentPPO |

All phases use the same six discrete actions:

| ID | Action | Effect |
| ---: | --- | --- |
| 0 | `TURN_LEFT` | Yaw left by 15 degrees |
| 1 | `TURN_RIGHT` | Yaw right by 15 degrees |
| 2 | `MOVE_UP` | Climb 0.5 m |
| 3 | `MOVE_DOWN` | Descend 0.5 m |
| 4 | `MOVE_FORWARD` | Move 1.0 m along the current heading |
| 5 | `SCAN` | Hold position; the sensor still fires |

There is no strafe and no reverse action. The sensor only covers the forward
cone, so forcing the agent to turn before translating removes a large class of
blind-side collisions.

---

## Curriculum gates

Every environment owns an internal three-level curriculum. Promotion is based
on a rolling 100-episode window; the final level is the evaluation level.

| Phase | Level 1 | Level 2 | Level 3 (evaluation) |
| --- | --- | --- | --- |
| 2A (MLP) | 5-10 obstacles, cylinders + boxes | 8-12, adds trees | 10-15, all shapes |
| 2A (GRU) | 5-10 obstacles, survivor visible | 8-12, adds trees | 10-15, all shapes |
| 2B (MLP) | 8-12 obstacles, ~20% mid-flight occlusion | 10-15, ~50% | 12-18, ~80% |
| 2B (GRU) | 8-12 obstacles, 30% hidden at spawn | 10-15, 50% | 12-18, 70% |
| 2C | 10-15 obstacles, 40% hidden | 12-18, 70% hidden | 15-22, 90% hidden |

Pass criteria at the final level are defined in each environment module. They
are engineering targets, not published benchmark results:

| Variant | Success | Crash | Timeout |
| --- | ---: | ---: | ---: |
| 2A MLP | >= 70% | <= 10% | <= 20% |
| 2A GRU | >= 72% | <= 10% | <= 18% |
| 2B MLP | >= 55% | <= 12% | <= 33% |
| 2B GRU | >= 65% | <= 12% | <= 23% |
| 2C MLP | >= 48% | <= 12% | <= 40% |
| 2C GRU | >= 55% | <= 12% | <= 33% |

The exact thresholds, obstacle counts, map sizes, and promotion rules live in
the `CURRICULUM_LEVELS` dictionaries inside each environment file. Treat the
code as the source of truth. See [`docs/PHASE_GUIDE.md`](docs/PHASE_GUIDE.md)
for the design rationale and evaluation protocol.

---

## Repository layout

```text
phase_2/
├── phase_2A/
│   ├── envs/       # 262-obs MLP env + 519-obs GRU env
│   ├── rewards/    # phase-specific reward shaping
│   ├── sensors/    # phase-local raycast sensor
│   └── training/   # PPO / RecurrentPPO entry points
├── phase_2B/
│   └── ...         # partial-occlusion environments
├── phase_2C/
│   └── ...         # full-occlusion environments
└── docs/
    └── PHASE_GUIDE.md

sensors/            # shared raycast sensor used by the GRU environments
utils/              # shared fog-of-war map
scripts/smoke_test.py
```

The phase-local `sensors/` folders keep each environment runnable in
isolation. The GRU environments import the shared `sensors.raycast_gru` and
`utils.fog_map` modules from the repository root.

---

## Quick start

### 1. Install dependencies

From the repository root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

PyBullet runs headless by default. Pass `render=True` when constructing an
environment to open the GUI.

### 2. Run the smoke test

The smoke test checks the shared utilities and one reset/step cycle of both
Phase 2A observation contracts. It does not train anything.

```bash
python scripts/smoke_test.py
```

Expected output:

```text
[PASS] shared raycast sensor + fog-of-war map
[PASS] Phase 2A environment reset/step (262-dim observation)
[PASS] Phase 2A GRU environment reset/step (519-dim observation)
All smoke checks passed.
```

### 3. Train a phase

Run training scripts from the repository root. The scripts prompt for a
checkpoint path; press Enter to start from scratch.

```bash
python phase_2/phase_2A/training/train_phase2A.py
python phase_2/phase_2B/training/train_phase2B.py
python phase_2/phase_2C/training/train_phase2C_gru.py
```

Checkpoints, TensorBoard event files, and human-readable episode logs are
written beside each training script under `checkpoints/` and `logs/`. These
directories are ignored by Git.

Training is long-running by design. The Phase 2C GRU script, for example,
targets four million environment steps on CPU. Start with a short run to
validate the setup before committing to a full schedule.

### 4. Watch an environment

```python
from phase_2.phase_2A.envs.phase2A_env import Phase2AEnv

env = Phase2AEnv(render=True, curriculum_level=3)
obs, info = env.reset(seed=0)
for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

---

## Reward design

The reward functions are intentionally explicit and phase-specific. The main
signals are:

- a one-time detection bonus when the survivor enters the sensor cone;
- distance shaping, applied only when the survivor is actually visible;
- a terminal reward for reaching the survivor;
- terminal penalties for collisions and altitude loss;
- a per-step time cost;
- anti-spin, proximity, coverage, and revisit terms where the phase needs them.

Phase 2A removes the coverage bonus after an earlier version learned to spin
next to obstacles for reward. Phase 2B adds a lost-target penalty and rewards
re-acquisition. Phase 2C adds coverage and revisit terms on top of the shared
fog map.

Read the docstrings in `phase_2/*/rewards/` before changing a coefficient.
Reward shaping is the most common source of unintended behaviour in this
project, and the comments record the failure modes that motivated each term.

---

## Status and limitations

- The environments and training entry points are complete and the smoke test
  passes on Python 3.12 with PyBullet.
- No Phase-2 policy checkpoint is included. Training results depend on the
  machine, step budget, and reward version, so the repository does not claim a
  converged success rate.
- The simulator uses kinematic position updates and abstract obstacle shapes,
  not full quadrotor dynamics. It is a curriculum testbed, not a flight
  controller.
- Detection is based on simulator object identity, not a learned detector.
- The PyBullet stack is a stepping stone to AirSim and Unreal Engine. The
  AirSim layer in [`../airsim/`](../airsim/) covers the higher-fidelity path.

---

## Citation

If this code is useful in academic work, cite the repository using
[`../CITATION.cff`](../CITATION.cff).
