# RL Drone Training Guide - UAVision

> **Historical engineering notes.** This document explains the v2 → v3 fixes
> that made training stable (hover before teleport, collision grace period,
> altitude floor, revised rewards). It is kept because the failure analysis is
> useful. For the current commands and observation contract, see
> [`../README.md`](../README.md).

## Quick Start

```bash
# 1. Make sure Unreal Editor is running with Play pressed
# 2. Activate your venv
cd C:\Users\tharu\Music\Nuclear Launch Codes\UAVision
.\venv\Scripts\activate

# 3. Train the model
cd AirSim\PythonClient\multirotor
python rl_train.py

# 4. Stop anytime with Ctrl+C (auto-saves!)
# 5. Resume by running the same command again

# 6. Watch training progress
tensorboard --logdir ./tb_logs
# Open http://localhost:6006

# 7. Run the trained model
python rl_play.py

# 8. Test the exit screen standalone
python exit_screen.py
```

---

## Files Overview

| File | Purpose |
|---|---|
| `rl_drone_env.py` | Gymnasium environment — the drone brain. Handles observations, actions, rewards |
| `rl_train.py` | Training script — creates DQN model, runs training loop, saves checkpoints |
| `rl_play.py` | Inference — loads trained model and runs it with live camera + minimap |
| `exit_screen.py` | **Optional** — God of War style exit screen. Standalone, safe to delete |
| `drone_full.py` | Manual GTA-style controls (not RL, human-controlled) |

---

## What Changed (v2 → v3)

### Problem: Drone crashed instantly every episode

**Root cause**: `drone.reset()` killed all motors → drone fell from 5m → hit ground → collision penalty (-100) → episode ended in 1 step. This repeated forever.

### Fix: Proper hover sequence

```diff
 def _setup_flight(self):
     self.drone.reset()
     self.drone.enableApiControl(True, VEHICLE)
     self.drone.armDisarm(True, VEHICLE)
+    
+    # TAKEOFF FIRST to engage motors (THE FIX)
+    self.drone.takeoffAsync(vehicle_name=VEHICLE).join()
+    time.sleep(0.3)
     
     # Teleport to random position
     self.drone.simSetVehiclePose(start_pose, True, VEHICLE)
     
-    # This was too late — drone already falling!
-    self.drone.moveToPositionAsync(sx, sy, sz, 5, ...).join()
+    # Hold position IMMEDIATELY after teleport
+    self.drone.moveToPositionAsync(sx, sy, sz, 5, timeout_sec=3, ...).join()
+    
+    # Clear false collision flags from teleport
+    time.sleep(0.2)
+    self.drone.simGetCollisionInfo(VEHICLE)  # discard
+    self._spawn_time = time.time()  # grace period starts
```

### New rewards

| Reward | Value | Why |
|---|---|---|
| Ground contact | **-200** | Falling = death. Works for water/tsunami too |
| Building collision | -100 | Bad but not fatal |
| Target found | +200 | Success! |
| Facing target (A*) | +2.0 × cos(θ) | Guides drone TOWARD target |
| Good altitude (3-8m) | +1.0 | Encourages stable hover |
| Dangerously low (<1.5m) | -3.0 | Stay away from ground |
| New area explored | +0.5 | Systematic search, not spinning |
| Approaching target | +1.0 bonus | Move closer! |
| Per-step cost | -0.3 | Encourages efficiency |

### Actions (5 instead of 6)

Removed **Hover** action — it was useless (the drone just wasted a step doing nothing).

| Action | Effect |
|---|---|
| 0 = Forward | Move 5m in facing direction |
| 1 = Up | Ascend 2m |
| 2 = Down | Descend 2m (floor at 2m altitude) |
| 3 = Yaw Left | Rotate -45° |
| 4 = Yaw Right | Rotate +45° |

### Safety features

- **Altitude floor**: Drone can never go below 2m altitude (NED Z > -2.0)
- **Altitude safety net**: After every action, if too low, auto-recover to 5m
- **Grace period**: 1.5 seconds after spawn, collisions are ignored (prevents false positives from teleport)
- **Ground death zone**: Z > -0.5 = instant death (-200). Universal — works for water, lava, tsunami, anything

---

## Exit Screen

### Standalone test
```bash
python exit_screen.py
```

### Integration into any script (2 lines)

```python
# LINE 1: Add this import at the top of your script
from exit_screen import show_exit_screen

# LINE 2: Call this wherever you want the exit screen
if show_exit_screen():
    sys.exit(0)   # User chose QUIT
# else: user chose RESUME, continue
```

### To disable exit screen in rl_train.py

Comment out or delete these lines in `rl_train.py`:

```python
# Line ~47: Comment this out
# from exit_screen import show_exit_screen

# The script already handles this gracefully with try/except,
# so you can also just delete exit_screen.py and it won't break anything.
```

### Controls
- **ENTER** = Resume (go back)
- **ESC** or **Q** = Quit
- **Click buttons** = Resume / Quit

---

## Training Tips

1. **Minimize Unreal window** during training → 2-3x speed boost
2. **ClockSpeed**: In `settings.json`, set `"ClockSpeed": 3.0` for faster training
3. **Watch TensorBoard**: reward should trend upward after ~2000 steps
4. **Expected behavior**: First 500-1000 steps = random exploration. After that, drone should start moving toward target
5. **ep_len_mean** should be **> 1** now (was stuck at 1 before the fix)

## Settings.json

Your current `settings.json` is fine. The key settings:

```json
{
    "SimMode": "Multirotor",
    "ClockSpeed": 1.0,    // ← Change to 3.0 for faster training
    "Vehicles": {
        "Drone1": {
            "VehicleType": "SimpleFlight",  // ← This is correct for RL
            "EnableTrace": true,
            "Sensors": { ... }  // LiDAR + Distance sensor = good
        }
    }
}
```

> **Tip**: Set `"ClockSpeed": 3.0` for 3x faster training. Set back to `1.0` for manual flying.
