# Baseline DQN checkpoint

`best_model.zip` is the best-evaluation checkpoint from the AirSim single-target
DQN experiments. It is included so the demo can be run without waiting for a
full training session. Treat it as a **research baseline**, not a validated
product.

## Model card

| Field | Value |
| --- | --- |
| Algorithm | Deep Q-Network (Stable-Baselines3 `DQN`) |
| Policy | `MultiInputPolicy` |
| Observation | depth image 84×84×1, 8 LiDAR sectors, 3-D target vector |
| Action space | `Discrete(5)` — forward, up, down, yaw left, yaw right |
| Training environment | AirSim + Unreal Blocks, single target |
| Training budget | 50,000-step target per run, resumable across sessions |
| Checkpoint selection | best mean evaluation reward |
| Intended use | simulation demo and pipeline validation |
| Out-of-scope | real aircraft control, safety-critical decisions |

## Evaluation history

`evaluations.npz` contains the `timesteps`, `results`, and `ep_lengths` arrays
written by Stable-Baselines3's `EvalCallback`. The recorded episodes are short
and the mean reward is negative, which means the policy frequently reaches the
timeout or collision terminal states. This is useful evidence that the
pipeline is wired correctly, but it is not a claim of strong search
performance.

## Reproduce

```bash
cd airsim
python rl_play.py --model rl_best/best_model.zip --episodes 10
```

To train a stronger policy:

```bash
python rl_train.py
```

The environment is stochastic (random spawn, target, obstacles, and yaw), so
compare policies over multiple episodes and multiple seeds before drawing
conclusions.

## Limitations

- Scene-specific: trained in the Blocks map, with a target actor named `Dwight`.
- Sensor-specific: changing the camera resolution, LiDAR channels, or
  `ClockSpeed` changes the observation distribution.
- No sim-to-real claim: the policy has not been transferred to hardware.
- The checkpoint is a `.zip` archive; load only models you trust.
