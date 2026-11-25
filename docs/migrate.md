# Migrate from Linux

[简体中文](migrate.zh.md)

If you already use LeRobot on Linux, you can keep your calibration files, LeRobot v3.0 datasets and complete policy
checkpoints. Use the Mac for collection, checks, inference and replay, and keep training on your GPU machine.

| Keep from Linux | Redo on the Mac |
|---|---|
| Calibration files of the same physical arms | Serial port paths |
| The `lerobot-record` or `lerobot-teleoperate` command | Camera assignment (Linux `/dev/video*` has no macOS counterpart) |
| Datasets and checkpoints | Nothing; they are read as they are |

Set up Python first, as in step 1 of the [Quickstart](quickstart.md#1-install).

## Bring the calibration files

Copy the JSON files of the **same physical arms** from Linux to the same place on the Mac:

```
~/.cache/huggingface/lerobot/calibration/
  robots/so_follower/<id>.json           SO-100 and SO-101 followers
  teleoperators/so_leader/<id>.json      SO-100 and SO-101 leaders
```

Other drivers use their own driver names in the same two folders. Keep the IDs from Linux. A new port does not
require recalibration.

??? note "Custom calibration folders"

    LeRobot honors `HF_HOME`, `HF_LEROBOT_HOME` and `HF_LEROBOT_CALIBRATION`. An explicit
    `--robot.calibration_dir` or `--teleop.calibration_dir` overrides the default for that device. If a custom folder
    moved, put its Mac path into the command before you import it.

The wizard shows which files it found and only reads them. If a known calibration is missing, check the ID, folder
and file permissions before you run the wizard's `lerobot-calibrate` command.

## Import the command

Paste the whole `lerobot-record` or `lerobot-teleoperate` command as one quoted argument, or paste only its flags:

```bash
macbot init --from-lerobot 'lerobot-record
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
  --dataset.fps=30 --dataset.single_task="place the cube in the box"'
```

The wizard opens with the imported values filled in.

- **Arms.** SO-100/101, Koch and OMX serial pairs. The importer keeps both types, the IDs, optional calibration
  folders, and the joint settings `use_degrees` and `max_relative_target` when given. macOS serial paths look like
  `/dev/cu.usbmodem*` or `/dev/cu.usbserial*`. A matching port is preselected; when it differs, choose the physical
  arm from the USB list.
- **Cameras.** OpenCV camera names become macbot roles, and explicit width, height and fps are kept. Match each role
  from the previews, because Linux `/dev/video*` paths and numeric indices have no reliable counterpart on macOS. The
  saved profile uses AVFoundation unique IDs. Check that the Mac camera supports the requested mode and that the view
  matches your training data. The picker skips iPhone and Continuity Camera.
- **Recording.** `--dataset.fps` sets the recording rate and `--dataset.single_task` fills the instruction.
  `lerobot-teleoperate --fps` sets the teleoperation rate.
- **Everything else.** The importer lists the flags it did not map, such as `--dataset.repo_id`, `--resume`, and
  camera options like rotation and FOURCC. For other drivers or camera types, write the YAML by hand following
  [Robots](robots.md) and [Setup](setup.md#cameras).

Pass `--yes` only when the port paths and camera unique IDs already match; otherwise use the picker.

After saving, use the same profile for `doctor`, `collect` and `run`. The default profile is
`~/.config/macbot/config.yaml`; for another file, put `--config profile.yaml` before the subcommand. With SO-101 and
UVC cameras, run as your normal user and allow Terminal to use the camera.

??? note "reBot and Orbbec under sudo"

    See [Setup](setup.md#rebot-b601-rs-reference-setup) for the SDK and the `sudo -E` commands. macbot preloads PCBUSB
    for the reference CAN profile and gives files it creates under sudo back to the invoking user. For inference under
    sudo, download a Hub checkpoint as your normal user and pass its local snapshot folder to `run --policy`.

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

### Reading existing datasets

`health` and `replay` accept local v3.0 Parquet datasets (MP4 or embedded RGB images) and Hub dataset IDs, optionally
pinned as `owner/name@revision`. Replay fetches frame and video shards as needed. Neither writes reviews or converts
the dataset. Convert older datasets with LeRobot before you open them.

- `health` counts missing frame indices and late timestamp gaps. LeRobot may write timestamps on a fixed grid, so
  these checks cannot see camera drops that happened before a frame was written.
- `replay` shows episode durations, the recorded state and actions, and each camera feed.

### Training and bringing the checkpoint back

Copy the whole exported dataset to the GPU machine and train with `lerobot-train`. Then either copy the resulting
`pretrained_model` folder back, or publish its contents to the Hub and use `run --policy owner/model@revision` or
`bench --policy owner/model@revision`. The Hub resolver prints the commit it fetched and records it in saved trials.

!!! warning "A checkpoint needs all of its files and the same robot layout"

    Include the weights, both processor JSON files and the state files they reference. Keep camera names, resolution,
    joint order and action units the same as in training. [Policies](policies.md) lists the layouts and model
    requirements.

??? note "What has been tested"

    | Path | Evidence |
    |---|---|
    | Reference reBot profile | Earlier hardware test |
    | Command import and remapping | Synthetic device tests |
    | Dataset inspection and playback | LeRobot 0.7.0 local output; mocked Hub |
    | Hub checkpoints | Public ACT downloaded, benchmarked and run on a fake arm (CPU) |
    | SO-101 collection and export | Earlier fake and simulated runs |
    | Generic arm motion on the Mac | Not verified |

    This migration workflow was exercised without moving hardware. See [Robots](robots.md) and
    [Policies](policies.md) for the scope of these checks.
