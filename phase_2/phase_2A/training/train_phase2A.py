"""
Phase 2A Training Script — Shape Recognition, No Occlusion
===========================================================
Run from project root:
    python phase_2/phase_2A/training/train_phase2A.py

Loads a Phase 1.5 checkpoint (MlpPolicy, obs=262 — same architecture).
Trains through internal L1→L2→L3 curriculum automatically.
All checkpoints and logs stored in phase_2/phase_2A/training/.
"""

import sys
import os
import datetime

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _PROJ_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from phase_2.phase_2A.envs.phase2A_env import Phase2AEnv

# ── Dirs ──────────────────────────────────────────────────────────────────────
_TRAIN_DIR = _HERE   # phase_2/phase_2A/training/


class HumanReadableLogCallback(BaseCallback):
    """One line per completed episode → human-readable .txt log."""

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir       = log_dir
        self.log_file      = os.path.join(log_dir, "training_log_phase2A.txt")
        self.episode_count = 0
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write(
                f"{'Episode':<8} | {'TotalSteps':<12} | {'EpReward':<12} | "
                f"{'EpLen':<8} | {'Level':<6} | {'Reason':<28} | Found\n"
            )
            f.write("-" * 112 + "\n")

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done:
                continue
            self.episode_count += 1
            info = self.locals["infos"][i]
            ep_r   = f"{info['episode']['r']:.2f}" if "episode" in info else "N/A"
            ep_l   = str(info["episode"]["l"])      if "episode" in info else str(info.get("step", "?"))
            reason = info.get("termination_reason", "?")
            found  = info.get("survivor_found", False)
            level  = info.get("curriculum_level", "?")
            with open(self.log_file, "a") as f:
                f.write(
                    f"{self.episode_count:<8} | {self.num_timesteps:<12} | "
                    f"{ep_r:<12} | {ep_l:<8} | {str(level):<6} | "
                    f"{reason:<28} | {found}\n"
                )
        return True


class CurriculumMonitorCallback(BaseCallback):
    """Prints curriculum level + rates to stdout every N steps."""

    def __init__(self, print_every=10_000, verbose=0):
        super().__init__(verbose)
        self.print_every   = print_every
        self._last_printed = 0

    def _on_step(self):
        if self.num_timesteps - self._last_printed < self.print_every:
            return True
        self._last_printed = self.num_timesteps
        try:
            env = self.training_env.envs[0].unwrapped
            ws  = len(env._outcome_success)
            if ws > 0:
                sr = sum(env._outcome_success) / ws * 100
                cr = sum(env._outcome_crash)   / ws * 100
                tr = sum(env._outcome_timeout) / ws * 100
                print(
                    f"[{self.num_timesteps:>10} steps]  Level={env.curriculum_level}  "
                    f"Success={sr:.1f}%  Crash={cr:.1f}%  Timeout={tr:.1f}%  (window={ws})"
                )
            else:
                print(f"[{self.num_timesteps:>10} steps]  Level={env.curriculum_level}  window filling...")
        except Exception:
            pass
        return True


if __name__ == "__main__":
    print("=" * 68)
    print("  Phase 2A Training — Shape Recognition, No Occlusion")
    print("=" * 68)
    print()
    print("Paste a Phase 1.5 checkpoint path and press Enter.")
    print("  • Phase 1.5 .zip → direct weight load (obs=262, same arch)")
    print("  • Empty           → fresh model from scratch")
    print()
    raw = input("Checkpoint path: ").strip().strip("'\"")

    is_resume = "phase2a" in raw.lower() or "phase_2a" in raw.lower() if raw else False
    ckpt_path = raw if raw else None

    if ckpt_path:
        print(f"\n[INFO] Loading weights from: {ckpt_path}")
    else:
        print("\n[INFO] Starting fresh model.")

    run_name   = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    ckpt_dir   = os.path.join(_TRAIN_DIR, "checkpoints", run_name)
    tb_log_dir = os.path.join(_TRAIN_DIR, "logs", run_name)
    hr_log_dir = os.path.join(tb_log_dir, "human_readable")
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(hr_log_dir, exist_ok=True)

    print(f"\n[INFO] Checkpoints → {ckpt_dir}")
    print(f"[INFO] TensorBoard → {tb_log_dir}")
    print(f"[INFO] Human log   → {hr_log_dir}")

    # n_envs=1 for GUI; set render=True to watch
    def _make_env():
        return Phase2AEnv(render=True, curriculum_level=1)

    n_envs = 1
    env    = make_vec_env(_make_env, n_envs=n_envs)

    TOTAL_TIMESTEPS = 800_000

    if ckpt_path and os.path.exists(ckpt_path) and not is_resume:
        print("\n[INFO] Transferring Phase 1.5 weights (direct load — same arch)...")
        model = PPO.load(
            ckpt_path, env=env, device="cpu",
            tensorboard_log=tb_log_dir,
            custom_objects={
                "learning_rate": 5e-5,
                "ent_coef":      0.01,
                "clip_range":    0.15,
            }
        )
        print("[INFO] Weights loaded — fine-tuning for shape recognition.")
    elif ckpt_path and os.path.exists(ckpt_path) and is_resume:
        print("\n[INFO] Resuming Phase 2A training...")
        model = PPO.load(ckpt_path, env=env, device="cpu", tensorboard_log=tb_log_dir)
    else:
        print("\n[INFO] Creating fresh Phase 2A model...")
        model = PPO(
            "MlpPolicy", env,
            verbose=1, device="cpu",
            tensorboard_log=tb_log_dir,
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=128,
            n_epochs=8,
            ent_coef=0.01,
            clip_range=0.15,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=ckpt_dir,
        name_prefix="phase2A",
    )
    human_log_cb  = HumanReadableLogCallback(log_dir=hr_log_dir)
    curriculum_cb = CurriculumMonitorCallback(print_every=10_000)

    print(f"\n[INFO] Training for {TOTAL_TIMESTEPS:,} steps...")
    print("[INFO] TensorBoard: tensorboard --logdir phase_2/phase_2A/training/logs")
    print("[INFO] Ctrl+C to stop — model saved automatically.\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, human_log_cb, curriculum_cb],
            reset_num_timesteps=not is_resume,
            tb_log_name="phase2A",
        )
        final_path = os.path.join(ckpt_dir, "phase2A_final")
        model.save(final_path)
        print(f"\n[INFO] Training complete. Final model → {final_path}.zip")

    except KeyboardInterrupt:
        interrupted_path = os.path.join(ckpt_dir, "phase2A_interrupted")
        model.save(interrupted_path)
        print(f"\n[INFO] Interrupted. Model saved → {interrupted_path}.zip")
