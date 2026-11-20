# Roadmap

[简体中文](ROADMAP_zh.md)

## Phase 1: Make macbot easy to try

- Add a demo video and GIF to the README.
- Publish a sample dataset and trained policy on Hugging Face Hub so users can replay and run
  in fake mode without an arm.
- Maintain CI, PyPI releases, contributor docs and issues labeled `good first issue`.
- **In progress:** SO-100/101 support has been exercised with fake arms and is untested on hardware.
  The SO-101 simulator has been exercised in simulation and provides no hardware validation.

## Phase 2: Validate more hardware and workflows

- Validate SO-100/101 on hardware.
- Document and verify a setup that uses only UVC cameras and needs no `sudo`.
- Train ACT and SmolVLA on Apple silicon using MPS.
- Build a community benchmark table for M1–M5 chips from `macbot bench` results submitted by PR.

## Phase 3: Extend execution and data tools

- Support more execution schemes as plugins, starting with the `AsyncChunk.accept()` hook.
- Run policy inference on a GPU server while the Mac handles the robot and cameras.
- Add failure tagging, one-command Hugging Face Hub upload, and dataset merging.
