"""
Phase 2C Training Script — Full Occlusion with GRU Memory
==========================================================
Run from project root:
    python phase_2/phase_2C/training/train_phase2C.py

Architecture change: PPO → RecurrentPPO (sb3-contrib) with GRU.
Loads Phase 2B MlpPolicy weights via selective layer transfer.
GRU layers are frozen for first 50k steps then unfrozen.

Install sb3-contrib if not already present:
    pip install sb3-contrib

All checkpoints + logs stored in phase_2/phase_2C/training/.
"""

import sys
import os
import datetime

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _PROJ_ROOT)

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from phase_2.phase_2C.envs.phase2C_env import Phase2CEnv

_TRAIN_DIR = _HERE

GRU_WARMUP_STEPS = 50_000


import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

# ── Custom Features Extractor for 2B preservation ─────────────────────────────
class Phase2BFeaturesExtractor(BaseFeaturesExtractor):
    """Replicates exactly the Phase 2B policy network before the LSTM."""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, layer1: int = 256, layer2: int = 128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], layer1),
            nn.ReLU(),
            nn.Linear(layer1, layer2),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


# ── Expert Warmup Callback ───────────────────────────────────────────────────
class ExpertWarmupCallback(BaseCallback):
    """Freezes the 2B expert layers (features & action head) so the untrained GRU learns to pass data through first."""

    def __init__(self, warmup_steps=50_000, verbose=0):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self._unfrozen    = False
        self._froze_once  = False

    def _on_step(self):
        # Apply freeze on the first step
        if not self._froze_once:
            count = 0
            for name, param in self.model.policy.named_parameters():
                if "features_extractor" in name or "action_net" in name:
                    param.requires_grad = False
                    count += 1
            print(f"\n[Phase 2C] Frozen {count} expert 2B param groups for {self.warmup_steps:,} GRU warmup steps.")
            self._froze_once = True

        # Unfreeze after warmup
        if self.num_timesteps >= self.warmup_steps and not self._unfrozen:
            count = 0
            for name, param in self.model.policy.named_parameters():
                param.requires_grad = True
                count += 1
            self._unfrozen = True
            print(f"\n[Phase 2C] Expert layers unfrozen at step {self.num_timesteps} "
                  f"({count} total param groups active)\n")
        return True


class HumanReadableLogCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir       = log_dir
        self.log_file      = os.path.join(log_dir, "training_log_phase2C.txt")
        self.episode_count = 0
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write(
                f"{'Episode':<8} | {'TotalSteps':<12} | {'EpReward':<12} | "
                f"{'EpLen':<8} | {'Level':<6} | {'CellsExp':<10} | {'Reason':<28} | Found\n"
            )
            f.write("-" * 128 + "\n")

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
            cells  = info.get("cells_explored", "?")
            with open(self.log_file, "a") as f:
                f.write(
                    f"{self.episode_count:<8} | {self.num_timesteps:<12} | "
                    f"{ep_r:<12} | {ep_l:<8} | {str(level):<6} | "
                    f"{str(cells):<10} | {reason:<28} | {found}\n"
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
                    f"Success={sr:.1f}%  Crash={cr:.1f}%  Timeout={tr:.1f}%  (w={ws})"
                )
        except Exception:
            pass
        return True


if __name__ == "__main__":
    print("=" * 68)
    print("  Phase 2C Training — Full Occlusion with GRU Memory")
    print("=" * 68)
    print()
    print("Paste a Phase 2B checkpoint path and press Enter.")
    print("  • Phase 2B .zip → partial weight transfer into GRU policy")
    print("  • Empty          → fresh RecurrentPPO model")
    print()
    raw = input("Checkpoint path: ").strip().strip("'\"")

    is_resume = "phase2c" in raw.lower() or "phase_2c" in raw.lower() if raw else False
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

    # Import RecurrentPPO — requires sb3-contrib
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        print("\n[ERROR] sb3-contrib not found. Install with:")
        print("    pip install sb3-contrib")
        sys.exit(1)

    def _make_env():
        return Phase2CEnv(render=False, curriculum_level=1)  # render=False during training

    n_envs = 1
    env    = make_vec_env(_make_env, n_envs=n_envs)

    TOTAL_TIMESTEPS = 4_000_000

    # Default fallback kwargs if not loading a transfer checkpoint
    policy_kwargs = dict(
        features_extractor_class=Phase2BFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128, layer1=256, layer2=128),
        lstm_hidden_size=128,
        n_lstm_layers=1,
        enable_critic_lstm=True,
        net_arch=dict(pi=[], vf=[]), # GRU is now the last step before the action head
    )

    if ckpt_path and os.path.exists(ckpt_path) and not is_resume:
        print("\n[INFO] Loading Phase 2B model to detect architecture...")
        from stable_baselines3 import PPO
        old_model  = PPO.load(ckpt_path, device="cpu")
        old_params = old_model.policy.state_dict()

        # Auto-detect Phase 2B architecture layer sizes
        l1_size = old_params["mlp_extractor.policy_net.0.weight"].shape[0]
        l2_size = old_params["mlp_extractor.policy_net.2.weight"].shape[0]
        print(f"[INFO] Auto-detected Phase 2B architecture: Layer1={l1_size}, Layer2={l2_size}")

        policy_kwargs = dict(
            features_extractor_class=Phase2BFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=l2_size, layer1=l1_size, layer2=l2_size),
            lstm_hidden_size=l2_size,
            n_lstm_layers=1,
            enable_critic_lstm=True,
            net_arch=dict(pi=[], vf=[]), 
        )

        print("\n[INFO] Creating RecurrentPPO and transferring 100% of Phase 2B expert weights...")
        model = RecurrentPPO(
            "MlpLstmPolicy", env,
            verbose=1, device="cpu",
            tensorboard_log=tb_log_dir,
            learning_rate=2e-5,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            ent_coef=0.01,
            clip_range=0.1,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs=policy_kwargs,
        )

        new_params = model.policy.state_dict()

        mapping = {
            "mlp_extractor.policy_net.0.weight": "features_extractor.net.0.weight",
            "mlp_extractor.policy_net.0.bias":   "features_extractor.net.0.bias",
            "mlp_extractor.policy_net.2.weight": "features_extractor.net.2.weight",
            "mlp_extractor.policy_net.2.bias":   "features_extractor.net.2.bias",
            "action_net.weight":                 "action_net.weight",
            "action_net.bias":                   "action_net.bias",
            # Note: We intentionally let the Value Function (vf) and GRU initialize fresh.
            # The value function learns quickly and doesn't affect the drone's immediate flying skill.
        }

        transferred = 0
        for old_key, new_key in mapping.items():
            if old_key in old_params and new_key in new_params:
                new_params[new_key] = old_params[old_key].clone()
                transferred += 1
            else:
                print(f"[WARN] Could not map {old_key} -> {new_key}")

        model.policy.load_state_dict(new_params)
        print(f"[INFO] Successfully transferred {transferred}/{len(mapping)} Phase 2B expert layers into Custom Features Extractor & Action Head.")

    elif ckpt_path and os.path.exists(ckpt_path) and is_resume:
        print("\n[INFO] Resuming Phase 2C training...")
        model = RecurrentPPO.load(ckpt_path, env=env, device="cpu",
                                   tensorboard_log=tb_log_dir)
    else:
        print("\n[INFO] Fresh RecurrentPPO (GRU) model...")
        model = RecurrentPPO(
            "MlpLstmPolicy", env,
            verbose=1, device="cpu",
            tensorboard_log=tb_log_dir,
            learning_rate=2e-5,
            n_steps=1024,
            batch_size=256,
            n_epochs=4,
            ent_coef=0.01,
            clip_range=0.1,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs=policy_kwargs,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // n_envs, save_path=ckpt_dir, name_prefix="phase2C"
    )
    human_log_cb  = HumanReadableLogCallback(log_dir=hr_log_dir)
    curriculum_cb = CurriculumMonitorCallback(print_every=10_000)
    expert_warmup_cb = ExpertWarmupCallback(warmup_steps=GRU_WARMUP_STEPS)

    print(f"\n[INFO] Training for {TOTAL_TIMESTEPS:,} steps...")
    print("[INFO] TensorBoard: tensorboard --logdir phase_2/phase_2C/training/logs")
    print("[INFO] Phase 2B expert reflexes are FROZEN for first 50k steps so GRU can adapt to them.")
    print("[INFO] Ctrl+C to stop — model saved automatically.\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, human_log_cb, curriculum_cb, expert_warmup_cb],
            reset_num_timesteps=not is_resume,
            tb_log_name="phase2C",
        )
        final_path = os.path.join(ckpt_dir, "phase2C_final")
        model.save(final_path)
        print(f"\n[INFO] Training complete. Final model → {final_path}.zip")

    except KeyboardInterrupt:
        p_path = os.path.join(ckpt_dir, "phase2C_interrupted")
        model.save(p_path)
        print(f"\n[INFO] Interrupted → {p_path}.zip")
