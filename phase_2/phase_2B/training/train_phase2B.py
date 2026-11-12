"""
Phase 2B Training Script — Partial Occlusion, No GRU
======================================================
Run from project root:
    python phase_2/phase_2B/training/train_phase2B.py

Loads a Phase 2A checkpoint (MlpPolicy, obs=262 — same architecture).
Slightly lower LR than 2A to preserve re-acquisition skill.
All checkpoints + logs stored in phase_2/phase_2B/training/.
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

from phase_2.phase_2B.envs.phase2B_env import Phase2BEnv

_TRAIN_DIR = _HERE


class HumanReadableLogCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir       = log_dir
        self.log_file      = os.path.join(log_dir, "training_log_phase2B.txt")
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
            info   = self.locals["infos"][i]
            ep_r   = f"{info['episode']['r']:.2f}" if "episode" in info else "N/A"
            ep_l   = str(info["episode"]["l"])      if "episode" in info else str(info.get("step","?"))
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
        except Exception:
            pass
        return True


if __name__ == "__main__":
    print("=" * 68)
    print("  Phase 2B Training — Partial Occlusion, No GRU")
    print("=" * 68)
    print()
    print("Paste a Phase 2A checkpoint path and press Enter.")
    print("  • Phase 2A .zip → direct weight load (obs=262, same arch)")
    print("  • Empty          → fresh model")
    print()
    raw = input("Checkpoint path: ").strip().strip("'\"")

    is_resume = "phase2b" in raw.lower() or "phase_2b" in raw.lower() if raw else False
    ckpt_path = raw if raw else None

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

    def _make_env():
        # Start at Level 2 — model has already passed Level 1
        return Phase2BEnv(render=False, curriculum_level=2)

    n_envs = 1
    env    = make_vec_env(_make_env, n_envs=n_envs)

    TOTAL_TIMESTEPS = 1_500_000  # Extra room: L2 polish → promote to L3 → L3 training

    if ckpt_path and os.path.exists(ckpt_path) and not is_resume:
        print("\n[INFO] Loading Phase 2A weights (direct — same arch)...")
        model = PPO.load(
            ckpt_path, env=env, device="cpu",
            tensorboard_log=tb_log_dir,
            custom_objects={
                "learning_rate": 3e-5,   # lower than 2A
                "ent_coef":      0.01,
                "clip_range":    0.15,
            }
        )
        print("[INFO] Phase 2A weights loaded.")
    elif ckpt_path and os.path.exists(ckpt_path) and is_resume:
        model = PPO.load(ckpt_path, env=env, device="cpu", tensorboard_log=tb_log_dir)
    else:
        print("\n[INFO] Fresh Phase 2B model...")
        model = PPO(
            "MlpPolicy", env, verbose=1, device="cpu",
            tensorboard_log=tb_log_dir,
            learning_rate=3e-5, n_steps=2048, batch_size=128,
            n_epochs=8, ent_coef=0.01, clip_range=0.15,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // n_envs, save_path=ckpt_dir, name_prefix="phase2B"
    )
    human_log_cb  = HumanReadableLogCallback(log_dir=hr_log_dir)
    curriculum_cb = CurriculumMonitorCallback(print_every=10_000)

    print(f"\n[INFO] Training for {TOTAL_TIMESTEPS:,} steps...")
    print("[INFO] TensorBoard: tensorboard --logdir phase_2/phase_2B/training/logs\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, human_log_cb, curriculum_cb],
            reset_num_timesteps=not is_resume,
            tb_log_name="phase2B",
        )
        final_path = os.path.join(ckpt_dir, "phase2B_final")
        model.save(final_path)
        print(f"\n[INFO] Done. Final model → {final_path}.zip")
    except KeyboardInterrupt:
        p = os.path.join(ckpt_dir, "phase2B_interrupted")
        model.save(p)
        print(f"\n[INFO] Interrupted → {p}.zip")
