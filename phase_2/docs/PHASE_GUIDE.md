# Phase 2 Design and Evaluation Guide

This guide explains why the Phase-2 curriculum is shaped the way it is, how to
extend it, and how to measure a policy without fooling yourself. The code is
the source of truth: thresholds and constants are defined in the
`CURRICULUM_LEVELS` and `PASS_CRITERIA` dictionaries inside each environment.

---

## 1. Why a separate PyBullet stack?

AirSim gives high-fidelity sensing inside Unreal Engine, but every step carries
rendering and RPC cost. A search policy needs hundreds of thousands to millions
of transitions; collecting them through a full game engine is slow and makes
ablation studies expensive.

PyBullet is used for the curriculum because it provides:

- fast headless stepping (`p.DIRECT`);
- deterministic, inspectable geometry;
- trivial parallelisation across processes;
- cheap raycasts for a 32x8 sensor grid;
- a small dependency footprint.

The trade-off is fidelity. The PyBullet drone is a kinematic body moved by
fixed position updates, not a quadrotor with thrust dynamics. The environments
test **decision-making and perception contracts**, not flight control. The
AirSim layer is the next fidelity step and the place to validate camera,
LiDAR, and control assumptions.

---

## 2. Observation contract

All environments use a flat `float32` vector with bounds `[-1, 10]`.

### MLP variant: 262 values

| Slice | Size | Contents |
| --- | ---: | --- |
| Rays | 256 | 8 rows x 32 columns, values 0-3 |
| Ego | 5 | `x / half_map`, `y / half_map`, `altitude / max_altitude`, `cos(yaw)`, `sin(yaw)` |
| Time | 1 | `step_count / max_steps` |

### GRU variant: 519 values

| Slice | Size | Contents |
| --- | ---: | --- |
| Rays | 256 | Same 32x8 first-hit grid |
| Ego | 5 | Same normalised pose |
| Fog | 256 | 16x16 local patch: `-1` unknown, `0` visited |
| Time | 1 | Normalised episode time |
| Terrain | 1 | Reserved terrain flag (currently `0`) |

### Ray semantics

Each ray is fired from the drone pose through the centre of one cell in a
120-degree horizontal by 60-degree vertical forward cone. The first hit decides
the code:

| Code | Meaning |
| ---: | --- |
| 0 | No solid hit within 20 m |
| 1 | Ground plane |
| 2 | Any obstacle |
| 3 | Survivor |

The first-hit rule is deliberate. It makes occlusion observable: if a survivor
is behind a wall, the sensor returns `2`, not `3`. The policy must either move
to a different viewpoint or rely on memory.

### Why the fog patch exists

A feed-forward policy can see what is in front of it but cannot tell whether a
region was already searched. The 16x16 fog patch supplies that context. It is
centred on the drone, clipped at map edges, and flattened row-major. Out-of-map
cells return `-1`, so the observation size is independent of map size.

---

## 3. Action contract

All phases share `Discrete(6)`:

| ID | Action | Step |
| ---: | --- | --- |
| 0 | Turn left | -15 degrees yaw |
| 1 | Turn right | +15 degrees yaw |
| 2 | Climb | +0.5 m, capped at 10 m |
| 3 | Descend | -0.5 m, floored at 1 m |
| 4 | Forward | 1.0 m along heading |
| 5 | Scan | No movement; sensors still fire |

There is no strafe and no reverse. The forward sensor cone means a sideways or
backward step would move into space the agent cannot currently observe. Turning
first is slower but safer, and it makes the learned behaviour easier to audit.

`SCAN` is the "look before you commit" action. It is especially important at
occlusion boundaries, where a single extra observation can disambiguate a
partially visible target.

---

## 4. Curriculum mechanics

Each environment starts at level 1. After every episode it appends the outcome
to three rolling deques of length 100. When the window is full, the environment
computes success, crash, and timeout rates. If all three gates pass, it promotes
one level, clears the window, and prints a promotion line. Level 3 never
promotes; it is the evaluation level.

This is a fixed-window gate, not a smoothed or confidence-based test. It is
simple and reproducible, but it can promote on a lucky window. For publishable
results, evaluate the final policy on a separate fixed seed set rather than
trusting the training-time promotion log.

### Gate summary

| Variant | L1 gate | L2 gate | Final target |
| --- | --- | --- | --- |
| 2A MLP | 80% success, 8% crash, 12% timeout | 75% / 10% / 15% | 70% / 10% / 20% |
| 2A GRU | 80% / 8% / 12% | 75% / 10% / 15% | 72% / 10% / 18% |
| 2B MLP | 65% / 25% / 20% | 60% / 20% / 30% | 55% / 12% / 33% |
| 2B GRU | 68% / 12% / 20% | 65% / 12% / 23% | 65% / 12% / 23% |
| 2C MLP | 52% / 20% / 36% | 48% / 20% / 40% | 48% / 12% / 40% |
| 2C GRU | 52% / 20% / 36% | 48% / 20% / 40% | 55% / 12% / 33% |

The 2B MLP gate is intentionally permissive because that variant has no memory:
it must solve re-acquisition using only the current frame. The GRU variants are
held to a higher standard because they have history.

---

## 5. Phase-by-phase design notes

### Phase 2A - recognition without occlusion

**Question:** Can the agent map ray codes to the correct behaviour when the
survivor is always visible at spawn?

**Design choices:**

- Survivor distance is 8-20 m, inside or near the 20 m sensor range.
- Obstacles are placed with a drone-safe radius and a survivor-safe radius.
- If the initial yaw does not see the survivor, the environment searches for a
  clear yaw before the episode starts. It does not rotate the survivor into
  view; it rotates the drone.
- Shapes are cosmetic. Cylinders, boxes, trees, rocks, and walls all return
  obstacle code `2`. The agent should not learn shape-specific semantics.

**Reward failure recorded in the code:** an early version paid a coverage bonus
for every ray that hit anything. With dense obstacles, spinning in place
generated more reward than finding the survivor. The current 2A reward removes
coverage entirely, gates distance shaping on actual visibility, increases the
time penalty, and adds an escalating anti-spin penalty.

**Transfer:** the 262-value MLP observation matches the earlier Phase-1 / 1.5
setup, so a checkpoint can be loaded without architecture surgery. The GRU
variant exists to test whether memory helps even when it is not required.

### Phase 2B - partial occlusion

**Question:** Can the agent re-acquire a survivor that was visible at spawn but
disappears behind an obstacle during the flight?

**Two variants:**

- `phase2B_env.py` uses an MLP and biases Level-1 obstacle placement away from
  the line of sight, then removes the bias at higher levels.
- `phase2B_env_gru.py` uses RecurrentPPO and samples a hidden-at-spawn
  probability (30%, 50%, 70%) in addition to mid-flight occlusion. It also
  carries the fog patch.

**Reward design:** a re-acquisition bonus is paid only after at least 15 steps
of genuine occlusion and at most twice per episode. This prevents the agent
from farming the bonus by blinking the target in and out of view. The v8
reward notes in `phase2B_rewards.py` document a proximity-zone bug that made
successful episodes net-negative and taught the policy to spin.

**Why memory matters:** with a feed-forward policy, the last observation before
occlusion is discarded. A recurrent policy can carry the last-seen bearing and
the recent obstacle pattern through the gap.

### Phase 2C - full occlusion and search

**Question:** Can the agent search a larger map when the survivor may be hidden
from the first frame?

**Design choices:**

- Map size grows to 50 m; survivor distance is 10-35 m.
- Obstacle counts reach 15-22 at the final level.
- Hidden-at-spawn probability reaches 90%.
- The fog map drives coverage and revisit terms, so the policy is rewarded for
  expanding the searched area rather than orbiting a safe region.
- RecurrentPPO uses `MlpLstmPolicy` with a 128-unit LSTM and a 256x128 MLP trunk.

**Evaluation is harder than 2A/2B.** Success depends on exploration strategy,
not just target tracking. Report coverage (visited cells) and path length
alongside success, crash, and timeout rates. A policy that finds the survivor
after 2,900 steps is not equivalent to one that finds it in 300 steps.

---

## 6. Training pipeline

### Entry points

| Script | Algorithm | Default budget |
| --- | --- | ---: |
| `phase_2A/training/train_phase2A.py` | PPO, MLP | 800,000 steps |
| `phase_2A/training/train_phase2A_gru.py` | RecurrentPPO | 1,200,000 steps |
| `phase_2B/training/train_phase2B.py` | PPO, MLP | 1,500,000 steps |
| `phase_2B/training/train_phase2B_gru.py` | RecurrentPPO | 2,000,000 steps |
| `phase_2C/training/train_phase2C.py` | RecurrentPPO | 4,000,000 steps |
| `phase_2C/training/train_phase2C_gru.py` | RecurrentPPO | 4,000,000 steps |

The scripts are interactive: they prompt for a checkpoint path. Press Enter to
start fresh. A path containing the current phase name is treated as a resume;
a path from the previous phase is treated as transfer learning.

### Outputs

Each run creates a timestamped directory:

```text
phase_2/<phase>/training/
├── checkpoints/<run_name>/   # periodic .zip checkpoints
└── logs/<run_name>/
    ├── events.out.tfevents.* # TensorBoard
    └── human_readable/       # one line per episode
```

These paths are ignored by Git. Keep large runs outside the repository or in
ignored directories.

### Resuming

All entry points use Stable-Baselines3 checkpointing. To resume:

1. Run the same script.
2. Paste the path to the latest checkpoint under the current phase.
3. Keep the same observation and action contract.

Recurrent policies additionally require the same `lstm_hidden_size`,
`n_lstm_layers`, and `enable_critic_lstm` settings. Changing them invalidates
the checkpoint.

### Hardware notes

- CPU training is the default and is often faster for these small MLPs/LSTMs
  because per-step Python and PyBullet overhead dominate.
- `n_envs = 1` is the safe default. Multiple environments require a vectorised
  setup and increase RAM.
- Headless mode (`render=False`) is strongly recommended for training.
- A full 2C run is a multi-hour to multi-day job depending on CPU. Use the
  smoke test and a 10k-step trial before launching a long run.

---

## 7. Evaluation protocol

Use the same protocol for every policy you compare.

1. **Freeze the environment.** Set `curriculum_level=3`. Do not let training
   promotion change the evaluation difficulty.
2. **Fix the seeds.** Evaluate on a fixed list of seeds that were not used to
   tune the policy. `env.reset(seed=s)` seeds NumPy inside the environment.
3. **Run enough episodes.** Two hundred episodes is the minimum for a
   success-rate estimate with a useful confidence interval; 500 is better.
4. **Use deterministic actions.** Call `model.predict(obs, deterministic=True)`.
5. **Handle recurrent state.** For RecurrentPPO, pass `state` and
   `episode_start` on every step, and reset them at episode boundaries.
6. **Report distributions, not one number.** Mean and standard deviation of
   episode reward, success rate, crash rate, timeout rate, episode length, and
   cells explored.
7. **Separate training from evaluation.** Do not report the best rolling window
   from training as a final result.

Minimal evaluation loop for an MLP policy:

```python
import numpy as np
from stable_baselines3 import PPO
from phase_2.phase_2A.envs.phase2A_env import Phase2AEnv

model = PPO.load("path/to/model.zip")
env = Phase2AEnv(render=False, curriculum_level=3)
outcomes = []

for seed in range(200):
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    outcomes.append(info)

env.close()
success = np.mean([o["survivor_found"] for o in outcomes])
print(f"success={success:.1%}")
```

For RecurrentPPO, keep `state=None` at reset and pass the returned state into
the next `predict` call. Set `episode_start` to a boolean array of length one.

---

## 8. Debugging playbook

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| Policy spins forever | Reward makes turning cheaper than moving | Time penalty, anti-spin term, proximity cost |
| Policy never finds survivor | Sensor range, spawn distance, or fog contract | Ray codes at spawn, survivor distance, observation shape |
| Success rises then collapses | Curriculum promotion too early or reward exploit | Rolling window, per-level reward logs |
| Training crashes after transfer | Observation or LSTM shape mismatch | `observation_space`, `lstm_hidden_size`, `net_arch` |
| GUI run is very slow | Rendering overhead | Use `render=False` for training |
| `pybullet` import fails | Wrong interpreter or missing build tools | Use the project venv; install `pybullet` from wheels |
| Different results across machines | Unseeded NumPy, thread nondeterminism, different PyBullet build | Fix seeds, record versions, evaluate multiple seeds |

When a policy learns something unexpected, inspect the reward first. The reward
docstrings in this repository are a record of real exploits found during
development; they are more useful than the final coefficients alone.

---

## 9. Extending the curriculum

To add a level:

1. Add an entry to `CURRICULUM_LEVELS` with the same keys used by the
   environment.
2. Update the level clamp if the environment hard-codes a maximum.
3. Add the promotion gates for the new level, or set them to `None` if it is
   terminal.
4. Update `PASS_CRITERIA` and this guide.
5. Run the smoke test and a short training trial.

To add a new observation term:

1. Write it into `_get_obs` in every variant that should share the contract.
2. Update `observation_space` and `OBS_SIZE`.
3. Update the smoke test assertion.
4. Retrain from scratch or rebuild the policy head; do not silently load an
   old checkpoint into a different input size.

---

## 10. From PyBullet to AirSim

The migration path is intentionally incremental:

| PyBullet | AirSim |
| --- | --- |
| Integer raycast grid | Depth image + LiDAR sectors |
| Kinematic pose updates | Multirotor velocity/pose commands |
| Abstract obstacle shapes | Unreal scene actors |
| Object-ID survivor detection | Segmentation or a learned detector |
| Single agent | Multi-agent swarm with shared map |

Keep the observation semantics stable across the transition where possible.
The AirSim DQN environment uses an 84x84 depth image, eight LiDAR sectors, and
a relative target vector; that is a different contract, so policies do not
transfer directly. The value of the PyBullet phase is the curriculum and reward
methodology, not a drop-in checkpoint.

---

## 11. Limitations

- Kinematic movement and abstract geometry are not a flight-dynamics model.
- Simulator object identity is ground truth, not perception.
- The reward functions are hand-designed and have known exploit history.
- Curriculum gates use a fixed 100-episode window and can be noisy.
- No Phase-2 checkpoint or benchmark table is included in this repository.
- The code targets research and education, not real search-and-rescue
  operations.
