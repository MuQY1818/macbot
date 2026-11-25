# Moving a LeRobot setup from Linux to a Mac

[简体中文](migrate.zh.md)

Keep your LeRobot calibration files, v3.0 datasets and complete policy checkpoints. Use the Mac
for collection, checks, inference and replay, and keep training on your GPU machine.
Start with the Python and camera setup in [quickstart](quickstart.md).

## Bring the calibration files

Copy the JSON files for the **same physical arms** from Linux. The standard directory on either
machine is `~/.cache/huggingface/lerobot/calibration/`. Under that directory, SO-100 and SO-101
share the driver names `robots/so_follower/<id>.json` and `teleoperators/so_leader/<id>.json`.
Other drivers use their own driver names in the same two groups. LeRobot honors
`HF_HOME`, `HF_LEROBOT_HOME` and `HF_LEROBOT_CALIBRATION`; explicit
`--robot.calibration_dir` and `--teleop.calibration_dir` override the default for those devices.
If a custom directory moved, substitute its Mac path in the command before importing.

Keep the IDs from Linux. A port change does not require recalibration. The wizard shows which
files it found and only reads them. If a known calibration is missing, check the ID, directory
and file permissions before using the wizard's separate `lerobot-calibrate` command.

## Import the command

Paste the whole `lerobot-record` or `lerobot-teleoperate` command as one quoted argument, or
paste just its flags. For example:

```bash
macbot init --from-lerobot 'lerobot-record
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
  --dataset.fps=30 --dataset.single_task="place the cube in the box"'
```

- **Arms:** SO-100/101, Koch and OMX serial pairs. The importer keeps both types, IDs, optional
  calibration directories and the joint settings `use_degrees` and `max_relative_target` when supplied.
  macOS USB serial paths often look like `/dev/cu.usbmodem*` or `/dev/cu.usbserial*`. A matching
  port is preselected; choose the corresponding physical arm from the USB list when it differs.
- **Cameras:** OpenCV camera names become macbot roles, and explicit width, height and fps are kept.
  Use previews to match each role: Linux `/dev/video*` paths and numeric indices have no reliable
  counterpart on macOS. The saved profile uses AVFoundation unique IDs. Check that the Mac camera
  can produce the requested mode and that the view matches your training data. The picker excludes
  iPhone and Continuity Camera.
- **Recording:** `--dataset.fps` sets the recording rate; `--dataset.single_task` fills the
  instruction. `lerobot-teleoperate --fps` sets the teleoperation control rate.
- **Other options:** The importer lists flags it did not map, such as `--dataset.repo_id` and
  `--resume`, and camera options such as rotation and FOURCC. For other drivers or camera types,
  use a manual YAML following [robot support](robots.md) and [camera setup](setup.md#4-cameras).
  Pass `--yes` only when the port paths and camera unique IDs already match; otherwise use the picker.

After saving, use the same profile for `doctor`, `collect` and `run`. The default profile is
`~/.config/macbot/config.yaml`; to use another path, put `--config profile.yaml` before the subcommand.
On SO-101 with UVC cameras, run as your normal user and grant Terminal camera access in macOS.
For the reBot/Orbbec setup, see [setup](setup.md) for the SDK and `sudo -E` commands:
macbot preloads PCBUSB for the reference CAN profile and hands files it creates under sudo back to
the invoking user. For inference under sudo, you can download a Hub checkpoint as your normal user
and pass its local snapshot directory to `run --policy`.

## Move data and checkpoints in both directions

```bash
# Read an existing lerobot-record dataset on the Mac, without changing it.
macbot health --dataset ./recordings
macbot replay --dataset owner/recordings

# Record new data on the Mac, check it, then produce standard LeRobot v3.0.
macbot collect
macbot health
macbot export --out ./new-dataset

# After training on your GPU machine, bring the complete checkpoint back.
macbot run --policy ./pretrained_model --fake --device cpu --autostart 3
macbot replay
```

`health` counts missing recorded frame indices and late timestamp gaps. LeRobot may synthesize
timestamps on a fixed grid, so these checks cannot measure camera drops before a frame was written.
`replay` shows episode durations, recorded state and actions, and each camera feed. Both accept
local v3.0 Parquet datasets (MP4 or embedded RGB images) and Hub dataset IDs, optionally
`owner/name@revision`. Replay fetches frame and video shards as needed. It does not write reviews
or convert the dataset. Convert older datasets with LeRobot before opening them.

Copy the entire exported dataset to the GPU machine and use it with `lerobot-train`.
Copy the resulting `pretrained_model` folder back, or publish its contents to the Hub and use
`run --policy owner/model@revision` or `bench --policy owner/model@revision`. The Hub resolver prints
the commit it fetched and records it in saved trials. Include weights, both processor JSON files,
and their referenced state files. Keep camera names, resolution, joint order and action units
aligned with the checkpoint. See [policy support](policies.md) for layouts and model requirements.

| Path | Evidence |
|---|---|
| Reference reBot profile | Prior hardware test |
| Command import and remapping | Synthetic device tests |
| Dataset inspection and playback | LeRobot 0.7.0 local output; mocked Hub |
| Hub checkpoints | Public ACT downloaded, benchmarked and run on a fake arm (CPU) |
| SO-101 collection and export | Prior fake and sim runs |
| Generic arm motion on the Mac | Not verified |

See [robot support](robots.md) and [policy support](policies.md) for the scope of those checks.
This migration workflow was exercised without moving hardware.
