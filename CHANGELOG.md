# Changelog

[简体中文](CHANGELOG_zh.md)

All notable changes to macbot will be documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/). What each version number means and when it changes is in
[docs/releasing.md](docs/releasing.md).

## [Unreleased]

### Added

- `macbot init` guides USB arm selection, live camera assignment and manual LeRobot calibration.
  It checks a per-user profile that `collect` and `run` can find automatically. `init --yes`
  supports setups whose roles are already known.
- Load compatible LeRobot checkpoints with their saved processors. `macbot policies` lists tested
  coverage, and `bench --profile` breaks down generic inference. ACT and π0.5 keep their defaults.

### Changed

- Local consoles try nearby ports when a default is busy and print the bound URL. Explicit ports stay fixed.
- The base install now includes LeRobot's `feetech` and `dynamixel` extras and `pyserial` for
  serial arm discovery and support.
- README: a pixel-art workflow scene replaces the console screenshot at the top, and the tagline and
  package description now cover LeRobot policies in general rather than only ACT and π0.5.

## [0.3.1] - 2026-11-20

### Added

- Guided Real-Time Chunking on the MLX backend. `--backend mlx --mode rtc` now stays on MLX instead
  of falling back to PyTorch: about 170 ms per guided chunk on an M5 Pro against about 350 ms on
  PyTorch MPS, matching LeRobot's guided output within 0.07° mean / 0.35° p99.
- `--mlx-time-grid quadratic` for `run` and `bench`: with a reduced `--steps`, places the flow
  evaluations closer to the clean action end. Lossy; evaluate on your own checkpoint first.
- `tools/accel/`: matmul shape benchmarks, alternative solvers (quadratic grid, AB2, midpoint) and
  attention-ranked visual token dropping for offline study only.
- Versioning and release policy (`docs/releasing.md`).

### Changed

- The README benchmark tables are grouped by policy, backend and execution scheme.
- The default 10-step MLX path returns bit-identical actions to 0.3.0.

## [0.3.0] - 2026-11-20

### Added

- An optional MLX backend for π0.5 (`--backend mlx` or `policy.backend: mlx`, install
  `macbot[mlx]`). It loads LeRobot pi05 checkpoints directly and runs in fp16 with compiled
  execution, trimmed text padding, precomputed AdaRMS, fused RoPE and packed projections:
  about 165 ms per 50-step chunk on an M5 Pro against about 270 ms for PyTorch MPS, with actions
  within 0.08° mean / 0.42° p99 of the bf16 reference (well under the seed-to-seed spread).
  Guided RTC still uses PyTorch; `--mode auto` picks asynchronous chunks with MLX.
- `macbot bench --profile` prints a synchronized per-stage breakdown (vision, prefix, each
  denoising step) for both backends, and `--steps` for shorter flow schedules.
- `tools/accel/`: an offline fidelity harness that builds a fixed observation set from recorded
  episodes, compares any configuration with matched noise, and reports latency and action error
  against the seed-to-seed spread.
- Inference benchmark docs (`docs/acceleration.md`, `docs/acceleration_zh.md`).

### Fixed

- π0.5 checkpoints store a padded 32-wide state; the startup dimension check now reads the real
  state width from the normalizer, so `macbot run` accepts them again.

## [0.2.0] - 2026-11-19

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

### Fixed

- `macbot bench` uses the checkpoint's state dimension so it can benchmark policies for arms
  with fewer or more than seven joints.

## [0.1.0] - 2026-11-19

### Added

- Initial reBot B601-RS and Arm 102 reference workflow on Apple silicon with a PCAN-USB
  adapter, UVC and Orbbec cameras, and fake arm and cameras for hardware-free testing.
- Browser collection with instructions, episode checks and replay; LeRobot v3 export,
  optional binary gripper labels, and example training scripts.
- ACT and π0.5 inference on MPS with synchronous chunks, temporal ensembling, asynchronous
  chunks and Real-Time Chunking, plus rollout replay, MP4 export and inference benchmarks.
- Equipment diagnostics, setup and usage guides, and English and Simplified Chinese consoles.

[Unreleased]: https://github.com/MuQY1818/macbot/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/MuQY1818/macbot/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/MuQY1818/macbot/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MuQY1818/macbot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MuQY1818/macbot/releases/tag/v0.1.0
