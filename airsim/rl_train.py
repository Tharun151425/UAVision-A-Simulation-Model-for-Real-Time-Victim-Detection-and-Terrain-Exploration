"""
rl_train.py - Train DQN Agent (Single Target) — RESUMABLE v3
=============================================================
Trains a DQN agent to navigate, avoid obstacles, and find a target.
Fully RESUMABLE: Ctrl+C to stop, re-run to continue from last checkpoint.

V3 FIXES:
  - Model now properly resumes from checkpoint
  - Eval env wrapped in VecTransposeImage (fixes warning)
  - Optimized hyperparams for longer episodes
  - Optional God of War exit screen on completion

REQUIRES:
  pip install stable-baselines3 gymnasium tensorboard numpy Pillow

HOW TO RUN:
  1. Unreal Editor → Press Play
  2. python rl_train.py
  3. (Ctrl+C to stop, re-run to resume!)

EXIT SCREEN (optional — remove these 2 lines if it causes issues):
  Line 1: from exit_screen import show_exit_screen
  Line 2: show_exit_screen()  (called at end of training)
"""
import setup_path
import time
import os
import sys
import glob

# ── Check dependencies ──
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
except ImportError:
    print("ERROR: Install stable-baselines3:")
    print("  pip install stable-baselines3[extra]")
    sys.exit(1)

try:
    import gymnasium as gym
except ImportError:
    print("ERROR: pip install gymnasium")
    sys.exit(1)

from rl_drone_env import AirSimDroneSearchEnv

# ── EXIT SCREEN (optional — comment out these 2 lines to disable) ──
try:
    from exit_screen import show_exit_screen    # LINE 1: import
    _HAS_EXIT_SCREEN = True
except ImportError:
    _HAS_EXIT_SCREEN = False

# ═══════════════════════════════════════════════════════════════
# TRAINING CONFIG
# ═══════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS     = 50_000
LEARNING_RATE       = 0.0003
BATCH_SIZE          = 128
BUFFER_SIZE         = 30_000
LEARNING_STARTS     = 100        # ← lowered: start learning fast
TARGET_UPDATE       = 500
EXPLORATION_FRAC    = 0.6        # ← more exploration (60%)
EXPLORATION_FINAL   = 0.05
GAMMA               = 0.99       # ← higher discount (episodes are longer now)
EVAL_FREQ           = 500       # ← more frequent eval
EVAL_EPISODES       = 3
SAVE_FREQ           = 200
LOG_DIR             = os.path.join(os.path.dirname(__file__), "tb_logs")
MODEL_DIR           = os.path.join(os.path.dirname(__file__), "rl_models")
BEST_MODEL_DIR      = os.path.join(os.path.dirname(__file__), "rl_best")


def make_env():
    """Create the training environment."""
    env = AirSimDroneSearchEnv(ip="127.0.0.1")
    env = Monitor(env)
    return env


def find_latest_checkpoint():
    """Find the most recent checkpoint to resume from."""
    # Check for best model first
    best_path = os.path.join(BEST_MODEL_DIR, "best_model.zip")
    if os.path.exists(best_path):
        return best_path, "best"
    
    # Check for final model (SB3 auto-appends .zip)
    final_path = os.path.join(MODEL_DIR, "dqn_drone_final.zip")
    if os.path.exists(final_path):
        return final_path, "final"
    
    # Check for numbered checkpoints
    pattern = os.path.join(MODEL_DIR, "dqn_drone_*_steps.zip")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        def get_step_num(path):
            base = os.path.basename(path)
            try:
                return int(base.split("_")[-2])
            except:
                return 0
        checkpoints.sort(key=get_step_num)
        return checkpoints[-1], f"checkpoint ({get_step_num(checkpoints[-1])} steps)"
    
    return None, None


def train():
    print("=" * 60)
    print("  RL TRAINING — DQN Target Finder v3 (Hover Fixed)")
    print("  RESUMABLE: Ctrl+C to stop, re-run to continue!")
    print("=" * 60)
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    
    # ── Create training env ──
    print("\n[1/3] Creating environment...")
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)  # Proper image transpose
    
    # ── Create eval env (same wrapper to avoid type mismatch) ──
    eval_env = DummyVecEnv([make_env])
    eval_env = VecTransposeImage(eval_env)
    
    # ── Check for existing model to resume ──
    resume_path, resume_type = find_latest_checkpoint()
    
    if resume_path:
        print(f"[2/3] RESUMING from {resume_type}: {resume_path}")
        model = DQN.load(
            resume_path,
            env=env,
            tensorboard_log=LOG_DIR,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            exploration_final_eps=EXPLORATION_FINAL,
        )
        model.learning_starts = 0
        print(f"  Loaded! Buffer size: {model.replay_buffer.size() if model.replay_buffer else 'N/A'}")
    else:
        print("[2/3] Creating NEW DQN model...")
        model = DQN(
            "MultiInputPolicy",
            env,
            learning_rate=LEARNING_RATE,
            verbose=1,
            batch_size=BATCH_SIZE,
            train_freq=4,
            target_update_interval=TARGET_UPDATE,
            learning_starts=LEARNING_STARTS,
            buffer_size=BUFFER_SIZE,
            max_grad_norm=10,
            exploration_fraction=EXPLORATION_FRAC,
            exploration_final_eps=EXPLORATION_FINAL,
            gamma=GAMMA,
            device="auto",
            tensorboard_log=LOG_DIR,
        )
    
    print(f"\n  Config:")
    print(f"    Total timesteps:  {TOTAL_TIMESTEPS:,}")
    print(f"    Learning rate:    {LEARNING_RATE}")
    print(f"    Batch size:       {BATCH_SIZE}")
    print(f"    Buffer size:      {BUFFER_SIZE:,}")
    print(f"    Gamma:            {GAMMA}")
    print(f"    Exploration:      {EXPLORATION_FRAC*100:.0f}% → {EXPLORATION_FINAL*100:.0f}%")
    print(f"    Eval frequency:   every {EVAL_FREQ:,} steps")
    
    # ── Callbacks (eval env properly wrapped now) ──
    eval_callback = EvalCallback(
        eval_env,                    # ← uses wrapped eval env
        best_model_save_path=BEST_MODEL_DIR,
        log_path=BEST_MODEL_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="dqn_drone",
    )
    
    # ── Train ──
    print("\n[3/3] Starting training!\n")
    print("  TIP: Open another terminal and run:")
    print(f"    tensorboard --logdir {LOG_DIR}")
    print("  Then open http://localhost:6006 in your browser")
    print("  TIP: Minimize Unreal window for ~2-3x speed boost!\n")
    
    run_name = f"dqn_search_{int(time.time())}"
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name=run_name,
            progress_bar=False,
            reset_num_timesteps=False if resume_path else True,
        )
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted — saving checkpoint...")
    
    # ── Save final model ──
    final_path = os.path.join(MODEL_DIR, "dqn_drone_final")
    model.save(final_path)
    print(f"\n[✓] Final model saved: {final_path}")
    print(f"[✓] Best model saved: {BEST_MODEL_DIR}/best_model")
    print(f"\n  To resume: python rl_train.py")
    print(f"  To run demo: python rl_play.py")
    
    env.close()
    eval_env.close()
    
    # ── EXIT SCREEN (optional — LINE 2: call it) ──
    if _HAS_EXIT_SCREEN:
        print("\n[🎮] Showing exit screen...")
        show_exit_screen()                       # LINE 2: show the screen


if __name__ == "__main__":
    train()
