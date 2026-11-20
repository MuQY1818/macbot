<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="macbot" height="72">
  </picture>
</p>

<p align="center"><b>Real-robot learning on a MacBook.</b><br>
Teleoperate, record, export to LeRobot and run ACT or π0.5 on the robot, all from a Mac on Apple silicon.</p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20silicon-18171F">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-F2701A">
  <img alt="LeRobot" src="https://img.shields.io/badge/LeRobot-0.7.0-1F9D63">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-6B6A75">
</p>

<p align="center"><b>English</b> · <a href="README_zh.md">中文</a></p>

<p align="center"><img src="docs/img/collect_en.jpg" alt="The macbot collection console" width="100%"></p>

## Why

Real-robot setups usually assume an Ubuntu workstation with an NVIDIA GPU next to the arm. We found a
MacBook is enough for everything that happens at the robot: teleoperated data collection, data checks,
export to LeRobot, and closed-loop policy inference, π0.5 included. Training still goes to a GPU machine.
macbot is the toolkit we use for that, cleaned up so others can reuse it.

Measured on a MacBook Pro (M5 Pro, 48 GB), PyTorch MPS:

| | |
|---|---|
| π0.5 inference, 50-step action chunk, 10 denoising steps | **~280 ms** (bf16 + `torch.compile`) |
| π0.5 with Real-Time Chunking guidance | **~350 ms** |
| ACT inference, 100-step chunk | **~25 ms** |
| Teleoperation loop / recording grid | 60 Hz / 30 fps |
| Camera frame age at sampling time, p95 | **≤ 40 ms** (scene UVC + wrist Orbbec) |

## What is inside

- **Collection console** in the browser: live cameras, one-key record / save / discard, per-episode language
  instruction, automatic quality checks (frame age, repeated frames, joint jumps, timing), instant replay of
  any episode with joint traces.
- **Export** to a LeRobot v3 dataset, optionally with a binary (open / closed) gripper action.
- **Policy runner** console for ACT and π0.5 with four execution schemes (synchronous chunks, temporal
  ensembling, asynchronous chunks, Real-Time Chunking), velocity and joint limits, pause / home / quit with a
  slow two-stage return to zero, and every trial saved with video and timing.
- **Replay** of saved trials (all cameras in sync, command vs. measured traces, chunk boundaries) and
  side-by-side **MP4** export.
- **`macbot doctor`** checks packages, MPS, the CAN driver, cameras and calibration before you start.
- **`--fake`** on every robot command: simulated arm and cameras, so the whole pipeline runs on any Mac.
- English and Chinese in every web page (toggle in the top bar), light and dark mode.

| Run a policy | Replay a trial |
|---|---|
| <img src="docs/img/run_en.jpg" alt="Policy runner"> | <img src="docs/img/replay_en.jpg" alt="Trial replay"> |

## Hardware

| | Used here | Notes |
|---|---|---|
| Follower arm | Seeed reBot B601-RS | driven through LeRobot's `rebot_b601_follower` |
| Leader arm | Seeed reBot Arm 102 | USB serial |
| CAN adapter | PEAK PCAN-USB | macOS driver: PCBUSB |
| Scene camera | any UVC webcam (1080p) | opened by AVFoundation uniqueID |
| Wrist camera | Orbbec Gemini 2 | color stream through `pyorbbecsdk2` |
| Computer | MacBook Pro, Apple silicon | 32 GB+ memory recommended for π0.5 |

Other arms that LeRobot supports can be added in `macbot/robot/`; the consoles only need `connect`,
`get_observation`, `send_action` and `disconnect`.

## Quick start

```bash
git clone https://github.com/MuQY1818/macbot && cd macbot
conda create -n macbot python=3.12 -y && conda activate macbot
pip install -e ".[orbbec,pi]"          # drop pi if you only use ACT

macbot doctor                          # what is ready, what is missing
macbot collect --fake                  # the collection console with a simulated arm and cameras
```

Open <http://127.0.0.1:8766>. Press **S** to record, **Enter** to save, **←** to discard.
For the real robot, install the PCAN driver and set your camera IDs first: see [docs/setup.md](docs/setup.md).

## Workflow

```bash
sudo -E $(which python) -m macbot.cli collect          # 1. teleoperate and record (sudo: Orbbec SDK)
macbot health                                          # 2. per-episode tracking and timing report
macbot export --gripper binary                         # 3. LeRobot dataset in ~/macbot-data/datasets/
DATASET=... ./train/train_pi05.sh my_run               # 4. train on a GPU machine (train_act.sh for ACT)
sudo -E $(which python) -m macbot.cli run --policy runs/my_run/checkpoints/last/pretrained_model
macbot replay                                          # 5. browse trials;  macbot video <rollout.npz> for an MP4
```

The full walkthrough, including the data format and the execution schemes, is in [docs/guide.md](docs/guide.md).
Put your settings in a YAML file and pass `macbot --config my.yaml <command>`; anything you leave out falls
back to [macbot/default.yaml](macbot/default.yaml). [configs/example.yaml](configs/example.yaml) is a starting point.

## What it took to run this on macOS

- **CAN under sudo.** The Orbbec SDK needs root, and dyld ignores `/usr/local/lib` for root processes, so
  `libPCBUSB.dylib` is preloaded explicitly before the CAN backend looks for it.
- **The right camera.** Opening a UVC camera by name or index can grab an iPhone through Continuity
  Camera. macbot opens cameras by AVFoundation uniqueID through a small Swift helper, and still finds the
  camera when it moves to another USB port.
- **π0.5 on MPS.** bf16 weights under autocast; the masked third camera slot is skipped (identical output,
  ~70 ms saved); `torch.compile` on the denoising step and the language model cuts per-kernel dispatch
  overhead (~410 → ~280 ms); the random weight initialization that the checkpoint overwrites anyway is
  skipped, which saves about 80 s per load.
- **Files written as root** are handed back to the user who ran `sudo`.

## Repository layout

```
macbot/
  cli.py            entry point: macbot <command>
  default.yaml      every setting, with comments
  robot/            reBot follower with feedforward, leader, safe shutdown, simulated arm
  cameras/          UVC (Swift helper), Orbbec, simulated cameras
  collect/          collection server and episode checks
  data/             export to LeRobot, data health report
  policy/           ACT / π0.5 wrappers, execution schemes, binary gripper, runner, benchmark
  viz/              trial replay server, MP4 export
  web/              pages, theme, shared scripts (en / zh)
train/              lerobot-train with openpi-style augmentation, example launch scripts
scripts/            double-click launchers for Finder
tools/              logo generator, screenshot renderer
tests/              fast tests without hardware
```

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

Apache-2.0. See [LICENSE](LICENSE).
