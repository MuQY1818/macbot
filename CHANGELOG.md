# Changelog

[简体中文](CHANGELOG_zh.md)

All notable changes to macbot will be documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/). What each version number means and when it changes is in
[docs/releasing.md](docs/releasing.md).

## [Unreleased]

## [0.3.5] - 2026-11-21

### Changed

- Use camera-relative WASD, vertical and wrist controls, and a visible key legend in keyboard simulation collection.
- Show the keyboard focus, gripper height and table projection, and workspace limits in the simulator preview.

### Fixed

- Load cached EnvHub scenes without contacting the Hub and diagnose missing SOCKS support for uncached scenes.
- Store simulation data in `~/macbot-data/sim` when no `data_root` is set.

## [0.3.4] - 2026-11-21

### Added

- Opt-in Cartesian keyboard and calibrated SO-101 leader input in the simulation collection console.
- Generate seeded tabletop demonstrations with `macbot sim demos`, with environment outcomes and standard episode files.
- `macbot eval` runs seeded simulator trials and reports environment success with Wilson intervals.
- `eval --trace-only` keeps replayable joint traces and renders only at policy queries for SO-101 tabletop batches.
- The macOS tabletop simulator reports its hardware CGL renderer and warns before trying a software fallback.

## [0.3.3] - 2026-11-21

### Added

- `init --from-lerobot` imports serial arm settings, camera roles, recording rate and task instructions. Confirm Mac device assignments and reuse the existing calibration IDs.
- `run` and `bench` accept Hub checkpoint IDs with optional revisions. Hub trials record the resolved commit.
- `health` and `replay` read local and Hub LeRobot v3.0 datasets, including frame gaps, recorded traces and camera playback.
- The migration guide covers moving an existing LeRobot setup from Linux to a Mac.
- Opt-in `--backend auto` for `run` and `bench`, with checkpoint compatibility checks and a reason when it
  falls back to PyTorch. Trials save the backend decision and `macbot policies` lists implemented backends.
- `bench --mode rtc` times guided inference for RTC-capable backends without connecting a robot.
- π0 MLX backend for compatible 224px PaliGemma 2B checkpoints, with chunk and guided RTC paths.
  Both modes met the absolute fidelity limits on recorded observations and are eligible for `auto`.
- SmolVLA MLX chunks for the stock SmolVLM2-500M architecture. `auto` uses the public base
  benchmark to select MLX for compatible checkpoints in synchronous chunk mode.
- Explicit PyTorch MPS experiments for ACT and Diffusion: autocast (potentially lossy), convolution
  layout, single readback on ACT, and compilation on Diffusion. Existing defaults are unchanged.

### Changed

- The README starts with the workflow for bringing an existing LeRobot setup to the Mac.
- π0 and π0.5 on MLX can read checkpoints that save SigLIP under `vision_model`.
- `--backend auto` uses MLX for compatible π0.5 chunk and RTC inference. On recorded observations it was
  faster and closer to FP32 than the compiled BF16 path.

## [0.3.2] - 2026-11-20

### Added

- `--query-every K` makes `async` and `rtc` query every K policy steps, and `--rtc-schedule exp` uses the exponential guidance decay of the RTC paper.
- `chunk`, `async` and `rtc` runs save the chunks at every switch, and `tools/chunk_shift.py` measures how far each new chunk starts behind the current command.
- `run` and `collect` record follower motor feedback and show temperatures in the joint panel. The default
  watchdog holds the measured pose on a motor fault, high temperature or sustained arm tracking error, and
  archives the interruption separately. `health` reports motor maxima and watchdog events.
- `run --grip-lead SECONDS` closes a binary gripper when the policy plans to close that far ahead in the current
  chunk, which offsets the late close label of leader-thresholded demonstrations. Off by default. Executors gain
  a side-effect-free `peek(now)`; trials record `gp_ahead`.
- Opt-in leader takeover in `run`, with arm and gripper alignment, a gripper speed limit, recorded control states and hand-in-scene marks.
- Replay shows calibrated grasp-miss flags, and reviews can exclude trials from summaries and exports.
- `macbot export --rollouts` writes LeRobot v3 trials with frame labels and reviewed outcomes.
- `macbot export --rollouts --segments` splits policy and human control into separate training episodes,
  drops handover and short runs, and labels policy frames leading up to a takeover.
- Review saved trials from current and earlier sessions in the paused run console, with synchronized
  cameras, joint traces and per-chunk timing. Result and note edits are shared with `replay` and `health`.
- `macbot init` guides USB arm selection, live camera assignment and manual LeRobot calibration.
  It checks a per-user profile that `collect` and `run` can find automatically. `init --yes`
  supports setups whose roles are already known.
- Load compatible LeRobot checkpoints with their saved processors. `macbot policies` lists tested
  coverage, and `bench --profile` breaks down generic inference. ACT and π0.5 keep their defaults.

### Changed

- `collect`, `run` and `replay` open their console in the default browser like `init`, also under `sudo`
  (as the user who ran sudo). A console page left open reloads for the next session instead of a new tab opening.
  Pass `--no-browser` or set `MACBOT_NO_BROWSER=1` to only print the URL.
- Setup can preserve existing camera framing and calibration. Exports and trials record framing, and
  policy runs reject a known mismatch before connecting.
- Local consoles try nearby ports when a default is busy and print the bound URL. Explicit ports stay fixed.
- The base install now includes LeRobot's `feetech` and `dynamixel` extras and `pyserial` for
  serial arm discovery and support.
- README: a pixel-art workflow scene replaces the console screenshot at the top, and the tagline and
  package description now cover LeRobot policies in general rather than only ACT and π0.5.

### Fixed

- Takeover starts from a rough match (every arm joint within 30°, gripper state matching) and follows the leader with
  a one-second soft start. The first version required all six joints within 5° at once, which nobody can hold by
  hand. Brief leader dropouts (under 0.5 s) no longer end the takeover, and leader readings that the servo library
  holds are used as `collect` uses them. The library holds readings near 0° as possible power-cycle glitches, so
  the closed leader gripper (0°) dropped 30-46 % of takeover frames before.
- **Home** in the run console now moves at `run.home_vel` (default 25 deg/s) along a straight line in joint space
  with a short ramp. It used the policy limit `max_vel × speed` (96 deg/s at speed 0.8), far too fast.
- Console pages no longer break after an upgrade because the browser kept an older cached `common.js`.
- `macbot init` no longer fails on macOS while scanning USB devices (`ioreg` returns one root entry), and a camera's
  serial port is no longer offered as an arm.

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

[Unreleased]: https://github.com/MuQY1818/macbot/compare/v0.3.5...HEAD
[0.3.5]: https://github.com/MuQY1818/macbot/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/MuQY1818/macbot/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/MuQY1818/macbot/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/MuQY1818/macbot/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/MuQY1818/macbot/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/MuQY1818/macbot/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MuQY1818/macbot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MuQY1818/macbot/releases/tag/v0.1.0
