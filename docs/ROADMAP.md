# Roadmap

The v1.0 release is a reproducible research testbed, not a finished product.
This roadmap separates work that is already implemented from work that is
planned, so readers can tell the difference.

---

## v1.0 - Integrated research release

**Status: complete**

- [x] Legacy Unreal manual-flight and segmentation prototypes preserved on a
      dedicated branch.
- [x] AirSim Gymnasium environment with depth, LiDAR, and target-direction
      observations.
- [x] Resumable DQN training, evaluation, inference, and demo player.
- [x] Manual GTA-style flight controller and perception HUD.
- [x] PyBullet Phase 2A/2B/2C environments with MLP and GRU variants.
- [x] Shared first-hit raycast sensor and fog-of-war map.
- [x] Three-level curriculum gates and phase-specific rewards.
- [x] Smoke test, CI, issue templates, and contribution guide.
- [x] Architecture, setup, results, literature survey, and model card.

**Known limitations**

- The included DQN checkpoint is an early baseline with negative evaluation
  rewards.
- No Phase-2 checkpoint or multi-seed benchmark table is included.
- The Unreal project and assets are not distributed.
- Multi-agent coordination and hardware deployment are not implemented.

---

## v1.1 - Validate the curriculum

**Goal:** replace training targets with measured evidence.

- [ ] Train Phase 2A MLP and GRU to the final level.
- [ ] Train Phase 2B MLP and GRU through the occlusion curriculum.
- [ ] Train Phase 2C GRU through full occlusion.
- [ ] Evaluate every policy on 500 fixed seeds, deterministic actions.
- [ ] Report success, crash, timeout, episode return, length, and coverage.
- [ ] Add learning curves and confidence intervals to `docs/RESULTS.md`.
- [ ] Publish small, validated checkpoints with model cards.
- [ ] Add a `scripts/evaluate_phase2.py` harness shared by all phases.

**Definition of done:** a table in `docs/RESULTS.md` that another person can
reproduce from a clean checkout and the recorded commands.

---

## v1.2 - Strengthen the AirSim policy

**Goal:** make the high-fidelity demo more than a pipeline check.

- [ ] Ablate reward terms (heading, proximity, time, exploration).
- [ ] Compare DQN, PPO, and recurrent PPO on the same observation contract.
- [ ] Add frame stacking or a recurrent policy for temporal depth.
- [ ] Evaluate across scene layouts, not just the Blocks map.
- [ ] Add a deterministic demo mode with fixed seeds.
- [x] Publish a demonstration clip in `docs/media/`.

**Definition of done:** a policy that reliably reaches the target in the Blocks
scene across multiple seeds, with a documented failure rate.

---

## v1.3 - Perception integration

**Goal:** move beyond simulator ground-truth detection.

- [ ] Add a detector baseline (YOLO-family) trained on a small labelled set.
- [ ] Add temporal tracking and false-positive rejection.
- [ ] Compare detector-in-the-loop performance with ground-truth segmentation.
- [ ] Add altitude-aware localisation and distance estimation.
- [ ] Document the domain gap between rendered and real imagery.
- [ ] Keep the detector optional so the RL curriculum remains reproducible.

**Definition of done:** an end-to-end perception pipeline whose errors are
measured, not hidden.

---

## v2.0 - Multi-agent search

**Goal:** extend the single-agent curriculum to a cooperative swarm.

- [ ] Implement multi-agent environments behind a new phase.
- [ ] Add frontier assignment and de-confliction.
- [ ] Add a shared map update protocol with bounded communication.
- [ ] Train with parameter sharing and centralised critic (MAPPO or equivalent).
- [ ] Evaluate scalability from 2 to 8 agents.
- [ ] Measure redundancy, coverage rate, and time-to-find.
- [ ] Add agent-loss and communication-failure tests.

**Definition of done:** a multi-agent policy evaluated on more than one map size
and agent count, with communication assumptions stated.

This is the largest open item. The current `feature/sar-swarm` branch name
refers to the SAR curriculum, not a completed swarm implementation.

---

## v2.1 - Hardware-in-the-loop

**Goal:** test the sim-to-real path without overclaiming.

- [ ] Select a compute module (Jetson class) and flight controller.
- [ ] Add a hardware adapter that preserves the observation/action contract.
- [ ] Add a safety supervisor: geofence, altitude limits, manual override.
- [ ] Measure latency, power, and thermal behaviour.
- [ ] Run controlled indoor flights with a tether or prop guards.
- [ ] Publish a failure analysis, not just a highlight reel.

**Definition of done:** a documented hardware experiment with safety limits and
a clear statement of what transferred and what did not.

---

## Ongoing engineering work

- [ ] Add type hints and a lightweight lint configuration.
- [ ] Add unit tests for reward edge cases and curriculum promotion.
- [ ] Add a deterministic evaluation script for recurrent policies.
- [ ] Add a small sample dataset or synthetic scene generator.
- [ ] Improve logging: JSONL episodes in addition to text logs.
- [ ] Add a `make`/`just` task runner for common commands.
- [ ] Keep dependencies pinned and test on Python 3.10-3.12.

---

## How to contribute to the roadmap

Pick an item, open an issue describing the experiment, and keep the change
small enough to review. For training results, include the commit hash, reward
version, seed set, and raw evaluation file. See
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

Roadmap items are intentions, not promises. A checked box means the work is
merged and reproducible, not that it is finished forever.
