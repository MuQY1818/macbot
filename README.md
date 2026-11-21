<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/logo.svg" alt="macbot" height="72">
  </picture>
</p>

<p align="center"><b>LeRobot on a MacBook.</b><br>
Bring your LeRobot arms, calibration, datasets and checkpoints to a MacBook, with fast inference on Apple silicon.</p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20silicon-18171F">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-F2701A">
  <img alt="LeRobot" src="https://img.shields.io/badge/LeRobot-0.7.0-1F9D63">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-6B6A75">
  <a href="https://github.com/MuQY1818/macbot/actions/workflows/test.yml"><img alt="tests" src="https://github.com/MuQY1818/macbot/actions/workflows/test.yml/badge.svg"></a>
</p>

<p align="center"><b>English</b> · <a href="https://github.com/MuQY1818/macbot/blob/main/README_zh.md">中文</a></p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/scene-dark.svg">
    <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/scene.svg" alt="Collect on the robot, export a LeRobot dataset, train on a GPU, bring the checkpoint back and run it on the MacBook" width="100%">
  </picture>
</p>

## Start here

Already using LeRobot on Linux? Follow the [migration guide](docs/migrate.md) to reuse your
calibration and import your command. Setting up from scratch? See the [quickstart](docs/quickstart.md).
Use Python 3.12+ on Apple silicon. LeRobot datasets and checkpoints move between machines as-is.

```bash
pip install macbot
macbot doctor
macbot init
# Already have a LeRobot command? Use macbot init --from-lerobot "$LEROBOT_COMMAND" instead.
macbot collect
macbot health
macbot export --out ./dataset
# Train with lerobot-train on a GPU machine, then bring the checkpoint back.
macbot run --policy ./pretrained_model       # or owner/model@revision
macbot replay
```

The reBot B601-RS is verified on hardware. The [SO-101 tabletop workflow](docs/simulation.md) runs
end to end in simulation (demos, export, LeRobot training, seeded evaluation); a trained policy
solving the task is not shown yet. See [robot support](docs/robots.md) for the scope of each profile.
For π0.5, use an Apple silicon Mac with at least 32 GB of memory. See [policy support](docs/policies.md)
for tested checkpoints.

Measured on a MacBook Pro (M5 Pro, 48 GB). Approximate time per action chunk:

| Policy | Backend | Execution | Latency |
|---|---|---|---:|
| π0.5 | PyTorch MPS | Chunk 50 | 270 ms |
| π0.5 | MLX | Chunk 50 | **165 ms** |
| π0.5 | PyTorch MPS | RTC 50 | 350 ms |
| π0.5 | MLX | RTC 50 | **170 ms** |
| SmolVLA | PyTorch MPS | Chunk 50 | 296 ms |
| SmolVLA | MLX | Chunk 50 | **53 ms** |
| X-VLA | PyTorch MPS | Chunk 30 | 235 ms |
| ACT | PyTorch MPS | Chunk 100 | 25 ms |

π0.5 uses our SO-101-class fine-tune on recorded observations, with 10 denoising steps (compiled PyTorch
BF16 or MLX FP16). RTC is Real-Time Chunking guidance. SmolVLA uses the public `lerobot/smolvla_base`
checkpoint at 10 steps (MPS BF16 backbone or compiled MLX FP16). X-VLA uses a public BF16 SO-101 fine-tune;
ACT uses our trained checkpoint on MPS FP32. For methods and more results, see
[docs/acceleration.md](https://github.com/MuQY1818/macbot/blob/main/docs/acceleration.md), other policies in
[docs/policies.md](https://github.com/MuQY1818/macbot/blob/main/docs/policies.md).

| Data collection | Measured |
|---|---:|
| Teleoperation loop | 60 Hz |
| Recording | 30 fps |
| Camera frame age at sampling, p95 | ≤ 40 ms |

<p align="center"><img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/collect_en.jpg" alt="The macbot collection console" width="100%"></p>

| Run a policy | Replay a trial |
|---|---|
| <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/run_en.jpg" alt="Policy runner"> | <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/replay_en.jpg" alt="Trial replay"> |

- `init` guides arm and camera selection and checks saved LeRobot calibration.
- `collect` records with live cameras and checks each episode; try `collect --fake` without an arm.
- `health --dataset` and `replay --dataset` inspect local or Hub LeRobot v3.0 data in place.
- `run` loads LeRobot checkpoints and records each trial with cameras and timing.
- `replay` aligns trial video, joint traces and action chunks. See the [workflow guide](docs/guide.md).

## What it took to run this on macOS

- **CAN under sudo:** macbot preloads PCBUSB and gives recorded files back to the invoking user.
- **Camera selection:** AVFoundation unique IDs keep UVC roles separate from iPhone Continuity Camera.
- **π0.5 inference:** macbot compiles the PyTorch MPS path and offers an optional MLX backend; see the benchmarks above.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for development and hardware reports,
[architecture](docs/architecture.md) for the code layout, and the [roadmap](ROADMAP.md) for planned work.

## Safety

macbot moves a real arm. Every console starts paused or idle, limits joint speed and range, clamps the
gripper force, pauses when a camera stalls, and returns to zero slowly before disabling the motors. Keep the
workspace clear and a hand near the power switch, and start new policies at `--speed 0.3`.

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot). π0.5 and the augmentation recipe come from
[openpi](https://github.com/Physical-Intelligence/openpi); Real-Time Chunking from Black et al., 2025.
The reBot arms are by [Seeed Studio](https://github.com/Seeed-Projects). JetBrains Mono is under the SIL
Open Font License.

## License

Apache-2.0. See [LICENSE](https://github.com/MuQY1818/macbot/blob/main/LICENSE).
