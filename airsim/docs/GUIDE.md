# 🛸 UAVision RL Drone Training — Complete Guide

> **Historical operator notes.** This guide records the working setup used
> during the AirSim experiments. For the current interface contract and the
> supported commands, start with [../README.md](../README.md). The action
> space evolved during development, so prefer the configuration values in the
> source code when the notes disagree.

## 📁 File Overview

| File | What It Does |
|------|-------------|
| `rl_drone_env.py` | **Base environment** — Single cone search with domain randomization. 6 FPV actions, 5m steps, random drone/cone/cube positions each episode. |
| `rl_drone_env_multi.py` | **Multi-cone environment** — Same as above but the drone must find ALL 3 cones (3/3). Each cone found gives +200 reward. |
| `rl_train.py` | **Train (single cone)** — Trains a DQN agent to find 1 cone. Resumable (Ctrl+C → re-run). ~3-6 hours. |
| `rl_train_multi.py` | **Train (multi cone)** — Trains to find all 3 cones. Resumable. ~6-12 hours. |
| `rl_play.py` | **Demo / Inference** — Loads a trained model and runs it with live camera + MINIMAP. Works for both single and multi-cone. |
| `drone_full.py` | **Manual controller** — GTA-style WASD+mouse controls. Not used for RL training, just manual flying. |
| `settings.json` | **AirSim config** — Sensors, camera resolution, physics speed. Goes in `Documents/AirSim/`. |

---

## 🎮 Action Space (What the Drone Can Do)

| Action | Name | What Happens |
|--------|------|-------------|
| 0 | **Forward** | Moves 5 meters in the direction the drone is currently facing |
| 1 | **Up** | Ascends 2 meters |
| 2 | **Down** | Descends 2 meters |
| 3 | **Hover** | Stays in place |
| 4 | **Yaw Left** | Rotates 45° counter-clockwise (no movement) |
| 5 | **Yaw Right** | Rotates 45° clockwise (no movement) |

> **FPV-realistic**: The drone has ONE front-facing camera. It CANNOT fly backward or strafe. To go right, it must: Yaw Right → Forward.

---

## ⚡ Setup Steps in Unreal Editor

### Step 1: Copy settings.json

Copy the updated `settings.json` to your AirSim settings folder:

```
C:\Users\<YOUR_USERNAME>\Documents\AirSim\settings.json
```

> **This is already done** — the code updated it automatically. But if you need to reset it, copy from the repo.

### Step 2: Open Your Unreal Project

Open your Blocks project (or any other AirSim project) in Unreal Editor.

### Step 3: Add Cones (= People)

The drone searches for **Cones**. You need at least **1 Cone** for single-cone training, **3 Cones** for multi-cone training.

#### How to add Cones:

1. In the **Content Browser** or **Place Actors** panel, search for **"Cone"**
2. Drag a **Cone** (Basic Shape) into the viewport
3. In the **World Outliner** (right panel), you should see it named `Cone`
4. For multi-cone: 
   - Drag **2 more Cones** into the scene
   - They will auto-name as `Cone2`, `Cone3`
   - **OR** manually rename them in the Outliner to `Cone2`, `Cone3`
5. **Position doesn't matter!** The Python code will teleport them every episode.

### Step 4: Add Cubes (= Obstacles) [Optional but Recommended]

Cubes act as random obstacles. The code scatters them randomly each episode.

1. Drag **5-10 Cube** shapes into the viewport
2. They will auto-name as `Cube`, `Cube2`, `Cube3`, etc.
3. **Scale them up** to make them actual obstacles:
   - Select each Cube → Details panel → Transform → Scale: `(2, 2, 3)` or similar
   - This makes them ~2m×2m×3m blocks
4. **Position doesn't matter** — Python randomizes their positions every episode.

> **Tip**: The cubes are individual objects, not walls. The code scatters them randomly — it doesn't try to form walls. They act as random obstacles the drone must avoid.

### Step 5: Press Play!

1. Click the **Play** button (▶) in Unreal Editor
2. You should see the drone (it will be controlled by Python)
3. **Keep the Unreal window open** — you can watch the drone fly
4. **Minimize for speed** — minimizing the Unreal window gives ~2-3x training speedup

---

## 🚀 How to Run

### Training (Single Cone) — Find 1 Person

```bash
cd AirSim/PythonClient/multirotor
python rl_train.py
```

- Trains for ~50,000 steps
- **Time**: ~3-6 hours (with ClockSpeed 3x and minimized Unreal)
- **Resumable**: Press **Ctrl+C** to stop. Run the same command again to resume!
- Models saved to `rl_models/` and `rl_best/`

### Training (Multi Cone) — Find ALL 3 People

```bash
python rl_train_multi.py
```

- Trains for ~80,000 steps
- **Time**: ~6-12 hours
- **Requires**: At least 3 Cones in Unreal scene (see Setup Step 3)
- Models saved to `rl_models_multi/` and `rl_best_multi/`

### Demo / Inference — Watch the Trained Drone

```bash
# Single cone demo (uses best model from rl_train.py)
python rl_play.py

# Multi-cone demo (uses best model from rl_train_multi.py)
python rl_play.py --multi

# Custom model
python rl_play.py --model path/to/model.zip

# More episodes
python rl_play.py --episodes 20

# No pygame window (headless)
python rl_play.py --no-render
```

### Demo Controls

| Key | Action |
|-----|--------|
| **M** | Toggle minimap ON/OFF |
| **R** | Reset current episode |
| **ESC** or **Q** | Quit |

### Monitoring with TensorBoard

While training is running, open another terminal:

```bash
tensorboard --logdir tb_logs
# or for multi-cone:
tensorboard --logdir tb_logs_multi
```

Then open http://localhost:6006 in your browser to see:
- Episode reward over time
- Episode length
- Loss curves

---

## 🗺️ Minimap (SLAM-like Map)

The demo (`rl_play.py`) has a **togglable minimap** (press **M**).

**How it works:**
- As the drone flies, it uses its **8 LiDAR sectors** to sense surroundings
- **Free space** (no obstacles) is marked as dark gray
- **Obstacles** (LiDAR hits something) are marked as **red dots**
- The **drone** is shown as a **cyan dot** with a heading arrow
- The drone's **trail** (path history) is shown as a **cyan line**
- When the **target is detected**, it appears as a **pulsing green dot**
- For multi-cone: **found cones** stay as **solid green dots**

> **Note**: The minimap is in the **Pygame window**, not inside Unreal. AirSim doesn't support custom Unreal UI from Python.

---

## 🌍 Using Different Environments (landscape-stf-pro-03, etc.)

**Yes, the same code works with any Unreal environment!** As long as:

1. **AirSim plugin is loaded** (same as Blocks)
2. **settings.json** is in `Documents/AirSim/`
3. **Cones and Cubes exist** in the scene (add them manually)

### For large environments:
If the environment is much bigger than Blocks, you may want to adjust the spawn bounds in `rl_drone_env.py`:

```python
# In rl_drone_env.py, change these:
SPAWN_MIN_XY   = -50.0     # increase for larger maps
SPAWN_MAX_XY   =  50.0
BOUNDS_MIN     = (-80, -80, -30)
BOUNDS_MAX     = (80, 80, 0)
```

Everything else (sensors, detection, training) works the same.

---

## ⏱️ Training Time Estimates

| Scenario | Steps | Estimated Time | Notes |
|----------|-------|---------------|-------|
| Single cone, Unreal minimized | 50K | **3-6 hours** | Best case with ClockSpeed 3x |
| Single cone, Unreal visible | 50K | **6-10 hours** | Slower due to rendering |
| Multi cone, Unreal minimized | 80K | **6-12 hours** | More complex task |
| Multi cone, Unreal visible | 80K | **12-20 hours** | Slower |

### Speed Tips:
1. **Minimize Unreal window** (click the `−` button) — 2-3x speedup
2. **ClockSpeed 3x** is already set in settings.json
3. **Small camera resolution** is already set (32×8 for depth)
4. **Close other programs** — AirSim uses GPU
5. **Resumable** — stop and restart anytime with Ctrl+C

---

## 🔄 Domain Randomization (How It Works)

Every time an episode resets (crash, found target, timeout), the code:

1. **Teleports drone** to random X/Y in [-30, +30] range, random yaw
2. **Teleports cone(s)** to random positions 10-25m from drone
3. **Scatters cubes** to random positions across the map
4. All positions are validated to avoid overlap

This means the drone sees a **different layout every episode**, which makes the trained model much more robust (it learns to navigate *any* layout, not just one specific one).

> **Python does all of this** — no Blueprint/Unreal changes needed. `simSetObjectPose()` teleports any actor by name while the game is running.

---

## 📂 Output Files

After training, you'll have:

```
multirotor/
├── rl_best/             # Best model from single-cone training
│   └── best_model.zip
├── rl_models/           # Checkpoints from single-cone training  
│   ├── dqn_drone_5000_steps.zip
│   ├── dqn_drone_10000_steps.zip
│   └── dqn_drone_final.zip
├── rl_best_multi/       # Best model from multi-cone training
│   └── best_model.zip
├── rl_models_multi/     # Checkpoints from multi-cone training
│   └── dqn_multi_final.zip
├── tb_logs/             # TensorBoard logs (single cone)
└── tb_logs_multi/       # TensorBoard logs (multi cone)
```

---

## 🐛 Troubleshooting

### "No trained model found!"
→ Run `python rl_train.py` first (or `rl_train_multi.py` for multi-cone)

### "Connection refused" or "Cannot connect to AirSim"
→ Make sure Unreal Editor is running and you clicked **Play**

### Drone doesn't move / stuck
→ Check that `settings.json` is in `C:\Users\<you>\Documents\AirSim\`
→ Try restarting Unreal (stop Play, re-Press Play)

### "Need 3 cones but only 1 exist!"
→ Add more Cone actors in Unreal Editor (see Setup Step 3)

### Training is too slow
→ Minimize Unreal window
→ Close other GPU-heavy programs
→ Reduce `TOTAL_TIMESTEPS` in `rl_train.py` for a quick test

### Cubes/Cones don't move between episodes
→ Check that their names match: `Cone`, `Cone2`, etc. and `Cube`, `Cube2`, etc.
→ The code uses `simListSceneObjects(".*")` and filters by name prefix

---

## 🎯 Quick Start Cheat Sheet

```bash
# 1. Setup (one time)
# - Copy settings.json to Documents/AirSim/
# - Add Cones + Cubes in Unreal Editor
# - Press Play in Unreal

# 2. Train (single cone, ~3-6 hours)
python rl_train.py

# 3. Demo (watch it find the cone!)
python rl_play.py

# 4. Train multi-cone (optional, ~6-12 hours)  
python rl_train_multi.py

# 5. Demo multi-cone
python rl_play.py --multi
```
