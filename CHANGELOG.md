# Changelog

All notable changes to UAVision are documented here. The project uses
[Semantic Versioning](https://semver.org/) for release tags and Conventional
Commit messages for history. Dates are in IST (UTC+05:30).

---

## [Unreleased]

### Planned

- Validated Phase-2 checkpoints and a multi-seed evaluation table.
- Shared evaluation harness for MLP and recurrent policies.
- Stronger AirSim policy with reward ablations.
- Detector-in-the-loop perception baseline.
- Multi-agent frontier coordination.

See [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## [1.0.0] - 2026-11-13

### Added

- Integrated `main` branch with the AirSim DQN stack and the PyBullet Phase-2
  curriculum.
- Phase 2A shape-recognition environments (262-value MLP and 519-value GRU).
- Phase 2B partial-occlusion environments with re-acquisition reward shaping.
- Phase 2C full-occlusion environments with fog-of-war search.
- Shared first-hit raycast sensor and fog-of-war map utilities.
- Lightweight smoke test for observation contracts.
- Architecture, setup, results, literature survey, and roadmap documents.
- SVG architecture and timeline diagrams.
- Demonstration video and environment captures for the AirSim stack.
- GitHub Actions smoke workflow.
- Issue templates, pull-request template, security policy, and contribution
  guide.

### Changed

- `FogOfWarMap.visit()` now returns a native Python `bool` instead of a NumPy
  boolean, matching its documented contract.
- Repository documentation is now organized around the integrated `main`
  branch rather than a single prototype.

### Known limitations

- The included AirSim DQN checkpoint is an early baseline with a negative mean
  evaluation reward and a 100% timeout rate in the recorded episodes.
- No Phase-2 checkpoint or benchmark table is included.
- Unreal project assets, build output, and simulator binaries are not
  distributed.
- Multi-agent coordination and hardware deployment are not implemented.

---

## [0.2.0] - 2026-11-12

### Added

- AirSim Gymnasium environments for single-target and multi-target search.
- Depth-image, LiDAR-sector, and target-direction observation contract.
- Resumable DQN training with checkpointing, evaluation, and TensorBoard logs.
- Inference and demo player with a LiDAR minimap.
- Manual GTA-style flight controller and perception HUD.
- Segmentation-based detection utilities and camera patch notes.
- AirSim `settings.json` for depth, RGB, segmentation, LiDAR, and distance
  sensors.
- Baseline DQN checkpoint, evaluation history, and model card.
- Operator guides and RL project explainer.

### Notes

- The baseline policy times out on all recorded evaluation episodes. It exists
  to validate the pipeline, not to claim strong search performance.

---

## [0.1.0] - 2026-11-10

### Added

- Legacy Unreal-era manual flight controllers.
- Segmentation-based object detection prototypes.
- Camera-relative "GTA-style" flight controller.
- Occupancy grid, object registry, and sample outputs.
- Legacy experiment notes and setup documentation.

### Notes

- Preserved on the `legacy/old-unreal` branch for provenance.
- The Unreal project itself is not distributed.

---

## Branch and tag map

| Tag | Branch | Contents |
| --- | --- | --- |
| `v0.1.0-legacy` | `legacy/old-unreal` | Manual flight and perception prototypes |
| `v0.2.0-airsim-rl` | `feature/airsim-rl` | AirSim DQN pipeline and baseline |
| `v1.0.0` | `main` | Integrated project and documentation |
