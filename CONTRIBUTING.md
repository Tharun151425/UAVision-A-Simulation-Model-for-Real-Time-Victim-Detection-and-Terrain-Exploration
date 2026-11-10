# Contributing to UAVision

Thanks for your interest. This repository is a research prototype, but it is
maintained with production-style hygiene: small commits, clear branch names,
reproducible experiments, and documented interfaces.

## Branch model

| Branch | Purpose |
| --- | --- |
| `main` | Stable, documented integration branch. |
| `legacy/old-unreal` | Archived first-generation manual-flight prototype. |
| `feature/airsim-rl` | AirSim + DQN search-agent work. |
| `feature/sar-swarm` | PyBullet Phase-2 SAR curriculum and recurrent policies. |

Work happens on a `feature/<topic>` branch and lands on `main` through a merge
commit. The legacy branch is intentionally archived and is not merged into
`main`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat(airsim): add target-relative observation term
fix(phase2): prevent false collision after teleport
docs(readme): add literature survey table
chore(ci): add Python syntax check
```

Guidelines:

- One logical change per commit.
- Explain *why* in the body when the change is non-obvious.
- Reference an issue when one exists (`Refs: #12`).
- Do not commit datasets, simulator builds, checkpoints, or `.uproject` files.

## Development setup

```bash
git clone https://github.com/Tharun151425/UAVision-A-Simulation-Model-for-Real-Time-Victim-Detection-and-Terrain-Exploration.git
cd UAVision-A-Simulation-Model-for-Real-Time-Victim-Detection-and-Terrain-Exploration
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

AirSim-specific dependencies and the optional Unreal setup are documented in
[`docs/SETUP.md`](docs/SETUP.md).

## Before opening a pull request

1. Run `python -m compileall -q airsim phase_2 sensors utils scripts`.
2. Run the lightweight smoke test: `python scripts/smoke_test.py`.
3. Update the relevant document under `docs/` if behaviour or interfaces change.
4. Keep the pull request focused; describe the experiment and the result.

## Reporting issues

Use the GitHub issue templates. Include:

- OS and Python version,
- simulator (AirSim/Unreal or PyBullet) and version,
- exact command,
- full traceback or screenshot,
- the smallest configuration that reproduces the problem.

## Code style

- Python: PEP 8, 4-space indentation, type hints where practical.
- Prefer explicit names over abbreviations in new code.
- Keep simulation constants at module level so experiments are reproducible.
- Do not add a dependency without a clear reason and a requirements update.

## License

By contributing you agree that your contribution is licensed under the MIT
License of this repository.
