"""
rl_train_multi.py - Train DQN Agent (Multi-Cone: Find ALL 3 Cones)
===================================================================
Same as rl_train.py but uses the multi-cone environment.
The drone must find ALL cones (3/3) in each episode.
Fully RESUMABLE.

REQUIRES:
  pip install stable-baselines3 gymnasium tensorboard numpy Pillow
  
  You need AT LEAST 3 Cone actors in Unreal Editor!
  (Cone, Cone2, Cone3 — see GUIDE.md for setup instructions)

HOW TO RUN:
  1. Unreal Editor → Press Play (make sure 3+ Cones exist)
  2. python rl_train_multi.py
  3. Ctrl+C to stop, re-run to resume
"""
import setup_path
import time
import os
import sys
import glob

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("ERROR: pip install stable-baselines3[extra]")
    sys.exit(1)

try:
    import gymnasium as gym
except ImportError:
    print("ERROR: pip install gymnasium")
    sys.exit(1)

from rl_drone_env_multi import AirSimDroneMultiConeEnv

# ═══════════════════════════════════════════════════════════════
# TRAINING CONFIG
# ═══════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS     = 80_000     # More steps needed for multi-target
LEARNING_RATE       = 0.0003
BATCH_SIZE          = 128
BUFFER_SIZE         = 40_000
LEARNING_STARTS     = 500
TARGET_UPDATE       = 500
EXPLORATION_FRAC    = 0.4
EXPLORATION_FINAL   = 0.05
GAMMA               = 0.95
EVAL_FREQ           = 5000
EVAL_EPISODES       = 3
SAVE_FREQ           = 5000
LOG_DIR             = os.path.join(os.path.dirname(__file__), "tb_logs_multi")
MODEL_DIR           = os.path.join(os.path.dirname(__file__), "rl_models_multi")
BEST_MODEL_DIR      = os.path.join(os.path.dirname(__file__), "rl_best_multi")


def make_env():
    env = AirSimDroneMultiConeEnv(ip="127.0.0.1")
    env = Monitor(env)
    return env


def find_latest_checkpoint():
    """Find the most recent checkpoint to resume from."""
    best_path = os.path.join(BEST_MODEL_DIR, "best_model.zip")
    if os.path.exists(best_path):
        return best_path, "best"
    
    pattern = os.path.join(MODEL_DIR, "dqn_multi_*_steps.zip")
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
    
    final_path = os.path.join(MODEL_DIR, "dqn_multi_final.zip")
    if os.path.exists(final_path):
        return final_path, "final"
    
    return None, None


def train():
    print("=" * 60)
    print("  RL TRAINING — DQN Multi-Cone Finder (Find ALL 3!)")
    print("  RESUMABLE: Ctrl+C to stop, re-run to continue!")
    print("=" * 60)
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    
    print("\n[1/3] Creating multi-cone environment...")
    env = DummyVecEnv([make_env])
    
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
    print(f"    Multi-cone mode:  Find ALL cones!")
    print(f"    Gamma:            {GAMMA}")
    print(f"    Eval frequency:   every {EVAL_FREQ:,} steps")
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=BEST_MODEL_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="dqn_multi",
    )
    
    print("\n[3/3] Starting training!\n")
    print("  TIP: tensorboard --logdir", LOG_DIR)
    print("  TIP: Minimize Unreal for speed!\n")
    
    run_name = f"dqn_multi_{int(time.time())}"
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback],
            tb_log_name=run_name,
            progress_bar=True,
            reset_num_timesteps=False if resume_path else True,
        )
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted — saving checkpoint...")
    
    final_path = os.path.join(MODEL_DIR, "dqn_multi_final")
    model.save(final_path)
    print(f"\n[✓] Final model saved: {final_path}")
    print(f"[✓] Best model saved: {BEST_MODEL_DIR}/best_model")
    print(f"\n  To resume: python rl_train_multi.py")
    print(f"  To run demo: python rl_play.py --model {BEST_MODEL_DIR}/best_model.zip --multi")
    
    env.close()


if __name__ == "__main__":
    train()
