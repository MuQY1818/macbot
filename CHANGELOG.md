# Changelog

[简体中文](CHANGELOG_zh.md)

All notable changes to macbot will be documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- A shared robot interface with a reBot B601 profile and generic LeRobot follower/leader
  configs for compatible joint-position arms, including SO-100/101, Koch, OMX, OpenArm and YAM.
- An SO-101 UVC configuration, the `macbot robots` command, and configurable fake joint layouts.
- A local SO-101 MuJoCo simulator with a scripted leader, scene reset and robot observation
  cameras (`type: robot`).
- An in-process collection driver and tests for generic arms, simulated cameras, export and diagnostics.
- Bilingual contributor, architecture, robot support and roadmap docs, community policies,
  GitHub issue and PR templates, and monthly GitHub Actions dependency updates.
- Development tooling with Ruff and pre-commit, with CI on macOS and Ubuntu for Python 3.12 and 3.13.

### Changed

- Collection, checks, rollout controls, and export use the configured joint layout and robot
  metadata. Policy startup checks checkpoint state/action dimensions; export preserves the
  robot type and accepts full-frame crops.
- `macbot doctor` checks the selected profile, including driver, serial and calibration
  checks for generic arms and local dependency checks for the bundled simulator.
- Added PyPI package metadata and a release workflow for Trusted Publishing.

## [0.1.0] - 2026-11-19

### Added

- Initial reBot B601-RS and Arm 102 reference workflow on Apple silicon with a PCAN-USB
  adapter, UVC and Orbbec cameras, and fake arm and cameras for hardware-free testing.
- Browser collection with instructions, episode checks and replay; LeRobot v3 export,
  optional binary gripper labels, and example training scripts.
- ACT and π0.5 inference on MPS with synchronous chunks, temporal ensembling, asynchronous
  chunks and Real-Time Chunking, plus rollout replay, MP4 export and inference benchmarks.
- Equipment diagnostics, setup and usage guides, and English and Simplified Chinese consoles.

[Unreleased]: https://github.com/MuQY1818/macbot/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MuQY1818/macbot/releases/tag/v0.1.0
