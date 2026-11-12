"""
Phase 2B GRU Training Script
==============================
Run from project root:
    python phase_2/phase_2B/training/train_phase2B_gru.py

Direct RecurrentPPO.load() from Phase 2A checkpoint.
lr=5e-5 | ent=0.005 | clip=0.12
Pass criteria L3: success>=65%, crash<=12%, timeout<=23%
"""

import sys, os, datetime

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _PROJ_ROOT)

from stable_baselines3.common.env_util  import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("\n[ERROR] pip install sb3-contrib"); sys.exit(1)

from phase_2.phase_2B.envs.phase2B_env_gru import Phase2BEnvGRU

_TRAIN_DIR = _HERE


class HumanReadableLogCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_file = os.path.join(log_dir, "training_log_phase2B_gru.txt")
        self.episode_count = 0
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write(f"{'Episode':<8} | {'TotalSteps':<12} | {'EpReward':<12} | "
                    f"{'EpLen':<8} | {'Level':<6} | {'CellsExp':<10} | {'Reason':<24} | Found\n")
            f.write("-" * 110 + "\n")

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done: continue
            self.episode_count += 1
            info = self.locals["infos"][i]
            ep_r = f"{info['episode']['r']:.2f}" if "episode" in info else "N/A"
            ep_l = str(info["episode"]["l"])      if "episode" in info else "N/A"
            with open(self.log_file, "a") as f:
                f.write(f"{self.episode_count:<8} | {self.num_timesteps:<12} | "
                        f"{ep_r:<12} | {ep_l:<8} | "
                        f"{str(info.get('curriculum_level','?')):<6} | "
                        f"{str(info.get('cells_explored','?')):<10} | "
                        f"{info.get('termination_reason','?'):<24} | "
                        f"{info.get('survivor_found', False)}\n")
        return True


class CurriculumMonitorCallback(BaseCallback):
    def __init__(self, print_every=20_000, verbose=0):
        super().__init__(verbose)
        self.print_every = print_every; self._last_printed = 0

    def _on_step(self):
        if self.num_timesteps - self._last_printed < self.print_every: return True
        self._last_printed = self.num_timesteps
        try:
            env = self.training_env.envs[0].unwrapped
            w   = len(env._outcome_success)
            if w > 0:
                sr = sum(env._outcome_success)/w*100
                cr = sum(env._outcome_crash)/w*100
                tr = sum(env._outcome_timeout)/w*100
                print(f"[{self.num_timesteps:>10} steps]  Level={env.curriculum_level}  "
                      f"Success={sr:.1f}%  Crash={cr:.1f}%  Timeout={tr:.1f}%  (w={w})")
        except Exception: pass
        return True


if __name__ == "__main__":
    print("=" * 68)
    print("  Phase 2B GRU Training — Partial Occlusion")
    print("=" * 68)
    print("\nPaste Phase 2A GRU checkpoint path (or Phase 2B to resume):")
    raw = input("Checkpoint path: ").strip().strip("'\"")

    is_resume = ("phase2b" in raw.lower() or "phase_2b" in raw.lower()) if raw else False
    ckpt_path = raw if raw else None

    run_name   = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    ckpt_dir   = os.path.join(_TRAIN_DIR, "checkpoints", run_name)
    tb_log_dir = os.path.join(_TRAIN_DIR, "logs",        run_name)
    hr_log_dir = os.path.join(tb_log_dir, "human_readable")
    for d in [ckpt_dir, tb_log_dir, hr_log_dir]: os.makedirs(d, exist_ok=True)

    n_envs = 1
    env = make_vec_env(
        lambda: Phase2BEnvGRU(render=False, curriculum_level=1), n_envs=n_envs)

    TOTAL_TIMESTEPS = 2_000_000

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"\n[INFO] Loading from: {ckpt_path}")
        overrides = ({"learning_rate": 5e-5, "ent_coef": 0.005, "clip_range": 0.12}
                     if not is_resume else {})
        model = RecurrentPPO.load(ckpt_path, env=env, device="cpu",
                                   tensorboard_log=tb_log_dir, custom_objects=overrides)
    else:
        print("\n[INFO] Fresh Phase 2B model.")
        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=1, device="cpu",
            tensorboard_log=tb_log_dir,
            learning_rate=5e-5, n_steps=2048, batch_size=64, n_epochs=10,
            ent_coef=0.005, clip_range=0.12, gamma=0.99, gae_lambda=0.95,
            policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1,
                               enable_critic_lstm=True,
                               net_arch=dict(pi=[256,128], vf=[256,128])),
        )

    print(f"\n[INFO] Checkpoints → {ckpt_dir}")
    print(f"[INFO] Training for {TOTAL_TIMESTEPS:,} steps...")
    print("[INFO] Pass criteria L3: success>=65%, crash<=12%, timeout<=23%\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[
                CheckpointCallback(save_freq=50_000//n_envs,
                                   save_path=ckpt_dir, name_prefix="phase2B_gru"),
                HumanReadableLogCallback(log_dir=hr_log_dir),
                CurriculumMonitorCallback(print_every=20_000),
            ],
            reset_num_timesteps=not is_resume, tb_log_name="phase2B_gru",
        )
        final = os.path.join(ckpt_dir, "phase2B_gru_final")
        model.save(final); print(f"\n[INFO] Done → {final}.zip")
    except KeyboardInterrupt:
        p_path = os.path.join(ckpt_dir, "phase2B_gru_interrupted")
        model.save(p_path); print(f"\n[INFO] Interrupted → {p_path}.zip")
