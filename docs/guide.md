# Guide: collect, export, train, run

[中文](guide_zh.md)

All paths below assume the default `data_root: ~/macbot-data` and a task called `cubes_to_box`.

```
~/macbot-data/
  raw/cubes_to_box/episode_000.npz ...      recorded episodes
  datasets/cubes_to_box_bin/                exported LeRobot dataset (+ gripper.json)
  rollouts/cubes_to_box/rollout_000.npz ... policy trials
  logs/                                     one log file per session
```

## 1. Collect

```bash
sudo -E $(which python) -m macbot.cli --config my.yaml collect
```

The page opens at <http://127.0.0.1:8766>.

| Key | Action |
|---|---|
| **S** | start recording |
| **Enter** or **→** | save the episode |
| **←** | discard it |

Edit the instruction box while idle; each episode stores its own instruction, which becomes the
language prompt for π0.5. Click **Replay** on any row to scrub through it with all cameras and the
joint traces. **Delete** moves the file to `deleted/`; numbers are never reused.

When an episode is saved, macbot checks it. Episodes shorter than `record.min_episode_s` are rejected.
These problems are flagged as warnings:

- frame age p95 above `checks.max_cam_age_s`
- more than `checks.max_repeat_ratio` repeated frames
- a joint jumping more than `checks.max_joint_step_deg` in one frame
- more than `checks.max_late_ratio` late samples

**Quit & home** (or Ctrl-C) brings the follower back to zero in two slow stages, then disables it.

### Episode format

| Key | Shape | Content |
|---|---|---|
| `qpos` | (N, 7) | measured follower joints, degrees |
| `action` | (N, 7) | command sent to the follower (gripper: leader reading) |
| `timestamp` | (N,) | k / fps, the fixed sampling grid |
| `wall_time` | (N,) | actual sampling time |
| `<cam>_jpeg`, `<cam>_offsets` | | JPEG stream; frame i is `jpeg[off[i]:off[i+1]]` |
| `<cam>_age`, `<cam>_repeat` | (N,) | frame age when sampled; whether it repeats the previous frame |
| `meta` | JSON | task, instruction, fps, cameras, follower feedforward, check result |

## 2. Check the data

```bash
macbot health                  # per-episode report
macbot health --split 27       # medians before / from episode 27, e.g. after a repair
```

For every episode the report lists:

- the lag that best aligns command and measurement, and the p95 tracking error at that lag
- the offset while the leader is still, where calibration drift shows up
- the gripper range
- the worst frame interval and the number of repeated camera frames

## 3. Export

```bash
macbot export                              # gripper action = leader angle
macbot export --gripper binary             # gripper action = 0 (open) / 1 (closed), writes gripper.json
```

The binary gripper works better than the raw angle for pick-and-place.

- **Labels:** "closed" starts when the leader reading rises above `--close-above` (-40°) and lasts until
  it falls below `--open-below` (-80°).
- **Deployment:** "open" is sent as `--open-deg` (-150°) and "closed" as `--close-deg` (0°). The gripper
  force clamp limits how hard it squeezes.

Episodes that failed the checks are skipped. Add `--skip-warnings` to skip episodes with warnings too.

## 4. Train

Training runs on a Linux machine with NVIDIA GPUs. Copy the dataset folder over, then:

```bash
DATASET=~/data/cubes_to_box_bin GPUS=0,1,2,3 ./train/train_pi05.sh cubes_pi05
DATASET=~/data/cubes_to_box_bin ./train/train_act.sh cubes_act
```

- `train_pi05.sh` fine-tunes `lerobot/pi05_base` through `train/train_augmented.py`. That script is
  `lerobot-train` plus openpi-style image augmentation on the GPU: crop and rotation on the scene camera,
  color jitter on all cameras.
- Set `WRIST_SHIFT=0.03` to add a small shift to the wrist camera. This helps when the wrist mount moved
  between recording sessions.

Copy `checkpoints/<step>/pretrained_model` back to the Mac. For a binary-gripper dataset, put
`gripper.json` next to it (or one or two folders above it).

## 5. Run a policy

```bash
sudo -E $(which python) -m macbot.cli run --policy ~/policies/cubes_pi05/004000 --speed 0.3
```

The page at <http://127.0.0.1:8767> starts **paused**.

| Key | Action |
|---|---|
| **S** | start |
| **Space** | pause (holds the current command) |
| **H** | move slowly to the start pose |
| **Y** / **N** | save the trial as success / failure |

Each trial is saved to `rollouts/<task>/` with all frames, the command and measured joints, chunk start
times and inference times.

### Execution schemes (`--mode`)

| Mode | What it does | Default for |
|---|---|---|
| `chunk` | infer, execute the first `--k` steps, infer again; the arm waits during inference | |
| `ensemble` | infer continuously; each target is the exp(m · age) weighted mean of all chunks covering now (ACT's temporal ensembling) | ACT |
| `async` | infer in the background while the current chunk keeps running; the new chunk is time-aligned with the old one | |
| `rtc` | `async` plus Real-Time Chunking: the unexecuted part of the old chunk guides denoising of the new one, so consecutive chunks join smoothly | π0.5 |

All schemes interpolate between policy steps for the 60 Hz control loop.

- **Speed:** `--speed` scales time. At 0.5, a 30 fps policy is played at 15 steps per second.
- **Safety limits:** arm joints move at most `run.max_vel × speed` degrees per second and stay within the
  joint limits.
- **Pause on stall:** if a camera stops for 0.5 s, the run pauses.

### Adding your own execution scheme

Executors live in `macbot/policy/executors.py` and share a four-method interface:

```python
class MyExecutor(Executor):
    def start(self): ...                  # start threads, if any
    def reset(self, active: bool): ...    # drop chunks; active=True: plan from the current observation
    def target(self, now: float): ...     # -> (joint target (7,), chunk id) or (None, id)
    def describe(self) -> str: ...
```

`AsyncChunk.accept()` is the hook for deciding whether, and at which step, to switch to a newly inferred
chunk. Subclass it to try new switching rules without touching the control loop.

## 6. Review

```bash
macbot replay                                   # http://127.0.0.1:8768
macbot video ~/macbot-data/rollouts/cubes_to_box/rollout_003.npz --height 720
```

`macbot bench --policy DIR` times inference on this Mac with blank images.

## Simulated hardware

`collect`, `run` and every page work with `--fake`: a first-order-lag arm and synthetic cameras. With
`run --fake` the policy is real, so it is a full check of loading, compilation and the executors before
you touch the robot. `--autostart SECONDS` runs, saves a trial and quits on its own.
