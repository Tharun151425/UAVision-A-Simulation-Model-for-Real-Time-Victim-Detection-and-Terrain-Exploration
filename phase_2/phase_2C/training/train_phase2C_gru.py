"""
Phase 2C GRU Training Script — Full Occlusion
================================================
Run from project root:
    python phase_2/phase_2C/training/train_phase2C_gru.py

Direct RecurrentPPO.load() from Phase 2B checkpoint.
Zero architecture surgery — same 519-obs, same lstm_hidden_size=128.

Hyperparameters (Phase 2C): lr=3e-5 | ent=0.005 | clip=0.10
Pass criteria L3: success>=55%, crash<=12%, timeout<=33%
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

from phase_2.phase_2C.envs.phase2C_env_gru import Phase2CEnvGRU

_TRAIN_DIR = _HERE


class HumanReadableLogCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_file = os.path.join(log_dir, "training_log_phase2C_gru.txt")
        self.episode_count = 0
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write(f"{'Episode':<8} | {'TotalSteps':<12} | {'EpReward':<12} | "
                    f"{'EpLen':<8} | {'Level':<6} | {'CellsExp':<10} | "
                    f"{'Reason':<28} | Found\n")
            f.write("-" * 120 + "\n")

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
                        f"{info.get('termination_reason','?'):<28} | "
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
    print("  Phase 2C GRU Training — Full Occlusion")
    print("=" * 68)
    print("\nPaste Phase 2B GRU checkpoint path (or Phase 2C to resume):")
    raw = input("Checkpoint path: ").strip().strip("'\"")

    is_resume = ("phase2c" in raw.lower() or "phase_2c" in raw.lower()) if raw else False
    ckpt_path = raw if raw else None

    run_name   = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    ckpt_dir   = os.path.join(_TRAIN_DIR, "checkpoints", run_name)
    tb_log_dir = os.path.join(_TRAIN_DIR, "logs",        run_name)
    hr_log_dir = os.path.join(tb_log_dir, "human_readable")
    for d in [ckpt_dir, tb_log_dir, hr_log_dir]: os.makedirs(d, exist_ok=True)

    n_envs = 1
    env = make_vec_env(
        lambda: Phase2CEnvGRU(render=False, curriculum_level=1), n_envs=n_envs)

    TOTAL_TIMESTEPS = 4_000_000

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"\n[INFO] Loading from: {ckpt_path}")
        overrides = ({"learning_rate": 3e-5, "ent_coef": 0.005, "clip_range": 0.10}
                     if not is_resume else {})
        model = RecurrentPPO.load(ckpt_path, env=env, device="cpu",
                                   tensorboard_log=tb_log_dir, custom_objects=overrides)
    else:
        print("\n[INFO] Fresh Phase 2C model (recommend loading Phase 2B checkpoint).")
        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=1, device="cpu",
            tensorboard_log=tb_log_dir,
            learning_rate=3e-5, n_steps=2048, batch_size=64, n_epochs=10,
            ent_coef=0.005, clip_range=0.10, gamma=0.99, gae_lambda=0.95,
            policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1,
                               enable_critic_lstm=True,
                               net_arch=dict(pi=[256,128], vf=[256,128])),
        )

    print(f"\n[INFO] Checkpoints → {ckpt_dir}")
    print(f"[INFO] TensorBoard → {tb_log_dir}")
    print(f"[INFO] Training for {TOTAL_TIMESTEPS:,} steps...")
    print("[INFO] Pass criteria L3: success>=55%, crash<=12%, timeout<=33%")
    print("[INFO] Ctrl+C saves automatically.\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[
                CheckpointCallback(save_freq=50_000//n_envs,
                                   save_path=ckpt_dir, name_prefix="phase2C_gru"),
                HumanReadableLogCallback(log_dir=hr_log_dir),
                CurriculumMonitorCallback(print_every=20_000),
            ],
            reset_num_timesteps=not is_resume, tb_log_name="phase2C_gru",
        )
        final = os.path.join(ckpt_dir, "phase2C_gru_final")
        model.save(final); print(f"\n[INFO] Done → {final}.zip")
    except KeyboardInterrupt:
        p_path = os.path.join(ckpt_dir, "phase2C_gru_interrupted")
        model.save(p_path); print(f"\n[INFO] Interrupted → {p_path}.zip")
