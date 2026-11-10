# Security Policy

## Supported versions

This is a research prototype. Security fixes are applied to the `main` branch.

## Reporting a vulnerability

Please do **not** open a public issue for security-sensitive reports. Email the
maintainer at `tharun151425@gmail.com` with:

- a description of the issue,
- reproduction steps,
- affected files and versions,
- any suggested remediation.

You should receive an acknowledgement within 7 days.

## Scope notes

- The AirSim/Unreal simulator is a separate upstream project; report simulator
  vulnerabilities to Microsoft AirSim.
- Never commit API keys, SSH keys, or personal paths. The repository includes a
  `.gitignore` for common local artifacts, but contributors are responsible for
  checking their own commits.
- The demo checkpoints are data files, not executable code. Load only models
  from sources you trust.
