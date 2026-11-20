# Guide: collect, export, train, run

[中文](guide_zh.md)

All paths below assume the default `data_root: ~/macbot-data` and a task called `cubes_to_box`.
The commands with `sudo` use the reference Orbbec camera. For generic arms with UVC cameras, use
`macbot --config my.yaml <command>` as a normal user. See [setup](setup.md#lerobot-arms-with-uvc-cameras).

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

Open the URL printed in Terminal (port 8766 when available).

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

**Quit & home** (or Ctrl-C) returns the reference reBot to zero in two stages. For a generic
LeRobot arm it returns to the starting pose, then disconnects.

### Episode format

| Key | Shape | Content |
|---|---|---|
| `qpos` | (N, J) | measured follower joints in robot units; J is the joint count |
| `action` | (N, J) | joint commands (reference reBot gripper: leader reading before the clamp) |
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
macbot export                              # keep the recorded actions in robot units
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

Open the URL printed in Terminal (port 8767 when available). The page starts **paused**.

| Key | Action |
|---|---|
| **S** | start |
| **Space** | pause (holds the current command) |
| **H** | move slowly to the start pose |
| **Y** / **N** | save the trial as success / failure |

Each trial is saved to `rollouts/<task>/` with all frames, the command and measured joints, chunk start
times and inference times.

While paused, click a row under **Saved trials** to replay it on the run page. The list starts with
the five newest trials from this task, including previous sessions; **Show all** opens the full list.
Scrub or play the synchronized cameras and joint traces. The position readout shows the current chunk
and its inference time. Use **Back to live** to return to the cameras. Starting or homing closes
the player; pause again to review another trial.

Change the trial's **Result**, enter an optional one-line **Note**, then click **Save** in the replay
panel. Edits live in `rollout_XXX.review.json` beside the recording. Keep the two files together when
copying a reviewed trial. The original recording is unchanged.

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
    def target(self, now: float): ...     # -> (joint target, chunk id) or (None, id)
    def describe(self) -> str: ...
```

`AsyncChunk.accept()` is the hook for deciding whether, and at which step, to switch to a newly inferred
chunk. Subclass it to try new switching rules without touching the control loop.

## 6. Review

```bash
macbot replay                                   # open the URL printed in Terminal
macbot video ~/macbot-data/rollouts/cubes_to_box/rollout_003.npz --height 720
macbot health --data ~/macbot-data/rollouts/cubes_to_box
```

Standalone replay uses the same player and annotations as the run console. When `health --data`
points to a rollout folder, each trial's report includes its reviewed result and note.

`macbot bench --policy DIR` times inference on this Mac with blank images.

## Fake hardware

`collect`, `run` and every page work with `--fake`: a first-order-lag arm and synthetic cameras. With
`run --fake` the policy is real, so it is a full check of loading, compilation and the executors before
you touch the robot. `--autostart SECONDS` runs, saves a trial and quits on its own.
Fake recordings go under `<task>_fake`; pass `export --data` or `health --data` to read that
directory, and use `replay --dir` for fake rollouts. For unlisted followers, set `robot.joints`
to the ordered joint names when using `--fake`.

## Simulation

Install the [MuJoCo extra](setup.md#local-mujoco-simulator) and use the same config throughout:

```bash
macbot --config configs/so101_sim.yaml collect
macbot --config configs/so101_sim.yaml health
macbot --config configs/so101_sim.yaml export
macbot --config configs/so101_sim.yaml run --policy path/to/pretrained_model
macbot --config configs/so101_sim.yaml replay
```

Press **S** to reset the scene and record the scripted leader's motion. This is pipeline test data;
the script does not solve the task. The saved episode can be replayed in the collection console.
For policy runs, use a checkpoint with six joint state/action values in the same order and units,
and with matching camera keys and sizes (`front` and `wrist`, both 320×240 for the example).
**Start** resets the scene for a new trial. **Pause** freezes it; save or discard the trial
before using **Reset scene**. `--autostart SECONDS` also works in simulation.
Keep simulated and hardware data in separate task folders.
