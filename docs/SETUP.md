# Setup Guide

UAVision has two runnable stacks:

| Stack | What you need | What it proves |
| --- | --- | --- |
| PyBullet curriculum | Python, NumPy, Gymnasium, PyBullet, Stable-Baselines3 | Search-policy training and curriculum logic |
| AirSim perception | Python 3.10/3.11, AirSim client, Unreal Engine + AirSim plugin | High-fidelity sensing, manual flight, DQN demo |

You can run the PyBullet stack on its own. The AirSim stack requires a local
Unreal project; this repository does **not** distribute the Unreal project,
`.uproject`, assets, or build output.

---

## 1. Clone and choose a branch

```bash
git clone https://github.com/Tharun151425/UAVision-A-Simulation-Model-for-Real-Time-Victim-Detection-and-Terrain-Exploration.git
cd UAVision-A-Simulation-Model-for-Real-Time-Victim-Detection-and-Terrain-Exploration
```

`main` contains the integrated project. The history is also available as
focused branches:

```bash
git branch -a
# legacy/old-unreal
# feature/airsim-rl
# feature/sar-swarm
```

Use `main` for the full project. Check out a feature branch only when you want
to inspect or extend that stage in isolation.

---

## 2. PyBullet stack

### 2.1 Create an environment

Python 3.10-3.12 works for the PyBullet stack. The smoke test was run on
Python 3.12.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate
```

### 2.2 Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The root requirements include PyTorch through Stable-Baselines3. If you only
want to validate the environments, install the minimal set:

```bash
pip install numpy gymnasium pybullet
```

### 2.3 Run the smoke test

```bash
python scripts/smoke_test.py
```

Expected:

```text
[PASS] shared raycast sensor + fog-of-war map
[PASS] Phase 2A environment reset/step (262-dim observation)
[PASS] Phase 2A GRU environment reset/step (519-dim observation)
All smoke checks passed.
```

### 2.4 Open a GUI environment

```bash
python -c "from phase_2.phase_2A.envs.phase2A_env import Phase2AEnv; e=Phase2AEnv(render=True, curriculum_level=1); e.reset(seed=0); [e.step(e.action_space.sample()) for _ in range(200)]; e.close()"
```

Use `render=False` for training. PyBullet's GUI is useful for debugging sensor
geometry, not for collecting millions of steps.

### 2.5 Train

```bash
python phase_2/phase_2A/training/train_phase2A.py
```

The script asks for a checkpoint path. Press Enter to start from scratch. Logs
and checkpoints are written under `phase_2/<phase>/training/` and ignored by
Git. See [`phase_2/README.md`](../phase_2/README.md) and
[`phase_2/docs/PHASE_GUIDE.md`](../phase_2/docs/PHASE_GUIDE.md) for the full
curriculum and evaluation protocol.

---

## 3. AirSim stack

### 3.1 Prerequisites

- Windows 10/11 is the best-supported path for Unreal + AirSim.
- Unreal Engine with an AirSim plugin version that matches your engine.
- The AirSim **Blocks** environment or another AirSim-enabled map.
- A Python environment that matches the AirSim client requirements.

The project was developed against the AirSim 1.8.x Python client. The client
stack pins older `msgpack` and `tornado` versions, so use a **separate**
environment from the PyBullet stack if you hit dependency conflicts.

### 3.2 Install the client

```bash
cd airsim
python -m venv .venv-air
# Windows
.venv-air\Scripts\activate
# Linux / macOS
source .venv-air/bin/activate
pip install -r requirements.txt
```

`airsim==1.8.1` is installed from PyPI. If you already have an AirSim source
checkout and want to use its client instead, put that checkout's `PythonClient`
directory on `PYTHONPATH`; do not place the repository's own `airsim/` folder on
the path as a package.

### 3.3 Install the AirSim settings

```powershell
# Windows
copy config\settings.json "$env:USERPROFILE\Documents\AirSim\settings.json"
```

```bash
# Linux / macOS
mkdir -p ~/Documents/AirSim
cp config/settings.json ~/Documents/AirSim/settings.json
```

The settings enable a forward LiDAR, a surround LiDAR, a forward distance
sensor, RGB/depth/segmentation capture, and a 3x clock speed. Change them only
if you understand how the observation distribution will shift.

### 3.4 Prepare the scene

1. Open the AirSim Blocks project (or your own AirSim-enabled map) in Unreal
   Editor.
2. Add a target actor named `Dwight` (a person mesh, cone, or any actor the
   segmentation camera can see). If you use another name, update
   `TARGET_LABEL` in `rl_drone_env.py`.
3. Add several Cube actors as obstacles. The environment randomises their
   positions, so initial placement does not matter.
4. Press **Play** and leave the simulator running.
5. Optionally minimise the Unreal window for faster training.

The exact actor names matter. `simListSceneObjects`, `simSetObjectPose`, and the
segmentation lookup all use scene names.

### 3.5 Run a manual flight demo

Open a second terminal, activate the AirSim environment, and run from inside
`airsim/`:

```bash
cd airsim
python drone_full.py
```

`drone_full.py` provides camera-relative manual control, a live HUD, and
segmentation-based detection. Use it to verify that the scene, sensors, and
keyboard bindings work before training.

### 3.6 Run the baseline policy

```bash
cd airsim
python rl_play.py --model rl_best/best_model.zip --episodes 5
```

The included checkpoint is an **early DQN baseline**. Its recorded evaluation
rewards are negative and every episode in the log reached the 200-step limit.
It is useful for validating the pipeline, not as evidence of strong search
behaviour. Read [`airsim/rl_best/README.md`](../airsim/rl_best/README.md).

### 3.7 Train from scratch or resume

```bash
cd airsim
python rl_train.py
```

Training is resumable. Stop with `Ctrl+C` and run the same command again; the
script finds the best checkpoint first, then the final model, then numbered
checkpoints. Outputs:

```text
airsim/tb_logs/     TensorBoard events
airsim/rl_models/   periodic checkpoints
airsim/rl_best/     best evaluation checkpoint
```

Monitor with:

```bash
tensorboard --logdir airsim/tb_logs
```

---

## 4. Common problems

### `ModuleNotFoundError: No module named 'airsim'`

You are probably in the wrong environment or running from a directory where
the local `airsim/` folder shadows the installed package. Use the AirSim
environment and run from inside the project's `airsim/` folder. Do not add the
repository root's `airsim/` folder to `PYTHONPATH` as a package.

### `Connection refused` / `Cannot connect to AirSim`

- Is Unreal in **Play** mode?
- Is the AirSim plugin loaded?
- Is `settings.json` in the right Documents folder?
- Is a firewall or VPN blocking localhost?
- Did you restart the Play session after changing settings?

### PyBullet opens a window and is slow

Pass `render=False`. GUI rendering is for inspection only. Training scripts
already default to headless mode.

### `pybullet` install fails on Linux

Install the build prerequisites for your distribution, or use a Python version
with a prebuilt wheel. On headless Linux, PyBullet may also need EGL/OSMesa
libraries for GUI mode; `p.DIRECT` still works without a display.

### AirSim Python client dependency conflict

The client pins old `msgpack`/`tornado` versions. Keep it in a separate virtual
environment from the PyBullet stack. Do not upgrade those pins unless you have
tested the RPC layer end to end.

### Training is slower than expected

- Minimise the Unreal window.
- Use `render=False` for PyBullet.
- Keep `n_envs=1` unless you have measured a benefit.
- Check that TensorBoard is not writing to a network drive.
- Use a machine with a modern CPU; these small policies are often CPU-bound.

---

## 5. Reproducibility checklist

- [ ] Record the Python version and `pip freeze` output.
- [ ] Record the AirSim plugin and Unreal Engine versions.
- [ ] Keep `settings.json` under version control.
- [ ] Fix evaluation seeds and use deterministic actions.
- [ ] Evaluate on more than one seed set.
- [ ] Keep training logs and checkpoints out of Git.
- [ ] Report the evaluation protocol alongside any success rate.
- [ ] Do not compare runs with different reward versions.

---

## 6. Safety and scope

This is simulation research code. It is not certified flight software and has
not been validated on physical aircraft. Do not use it to control a real drone
near people, property, or airspace without the required training, permissions,
and safety systems.
