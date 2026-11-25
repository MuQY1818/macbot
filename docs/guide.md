# Workflow guide

[简体中文](guide.zh.md)

This page walks through one task from recording to a reviewed policy run. It assumes you finished the
[Quickstart](quickstart.md), the default `data_root: ~/macbot-data` and a task called `cubes_to_box`.

```
~/macbot-data/
  raw/cubes_to_box/episode_000.npz ...      recorded episodes
  datasets/cubes_to_box_bin/                exported LeRobot dataset (+ gripper.json)
  rollouts/cubes_to_box/rollout_000.npz ... policy trials
  logs/                                     one log file per session
```

!!! note "Commands with sudo"

    The commands on this page run as a normal user, which works for UVC cameras. The Orbbec SDK may need root;
    in that case start `collect` and `run` with `sudo -E` as shown in the [Quickstart](quickstart.md#special-hardware).
    A config other than the default is passed as `macbot --config my.yaml <command>`; see
    [Setup](setup.md#lerobot-arms-with-uvc-cameras).

## 1. Collect

```bash
macbot collect
```

The console opens in the browser (port 8766 when it is free).

| Key | Action |
|---|---|
| ++s++ | Start recording |
| ++enter++ or ++arrow-right++ | Save the episode |
| ++arrow-left++ | Discard the episode |

- **Instruction.** Edit the instruction box while idle. Each episode stores its own instruction, which becomes
  the language prompt for π0.5.
- **Replay.** Click **Replay** on any row to scrub through the episode with all cameras and the joint traces.
- **Delete.** **Delete** moves the file to `deleted/`. Episode numbers are never reused.
- **Quit.** **Quit & home** (or ++ctrl+c++) returns the reference reBot to zero in two stages. A generic LeRobot
  arm returns to its starting pose. Then macbot disconnects.

macbot checks every saved episode. Episodes shorter than `record.min_episode_s` are rejected. These problems
are flagged as warnings:

- frame age p95 above `checks.max_cam_age_s`
- more than `checks.max_repeat_ratio` repeated frames
- a joint jumping more than `checks.max_joint_step_deg` in one frame
- more than `checks.max_late_ratio` late samples

??? note "Episode file format"

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
macbot health --split 27       # medians before and from episode 27, for example after a repair
```

For every episode the report lists:

- the lag that best aligns command and measurement, and the p95 tracking error at that lag
- the offset while the leader is still, where calibration drift shows up
- the gripper range
- the worst frame interval and the number of repeated camera frames

`macbot health` also shows per-joint motor maxima and time-stamped watchdog events. Use `--data` to read a run
directory instead of the recordings.

### Motor health and the watchdog

The joint panel in `collect` and `run` shows each motor's temperature and marks the hottest one.

| Condition | Default | What happens |
|---|---|---|
| Temperature above `safety.temp_warn_c` | 60 °C | Highlighted; **Start** stays locked until the motor is below `safety.temp_resume_c` (55 °C) |
| Static command away from the start pose for `safety.hold_warn_s` | 30 s | The panel suggests returning to the start pose to rest the motors |
| Fault bit reported, or temperature above `safety.temp_alarm_c` | 70 °C | The watchdog stops the policy or teleoperation |
| Arm joint error above `safety.track_deg` for longer than `safety.track_s` | 8°, 0.3 s | The watchdog stops the policy or teleoperation |

When the watchdog trips, it commands the measured pose and locks motion commands for the rest of the session.
The tracking check compares against the last command actually sent and leaves out the gripper, because holding
an object stalls the gripper by design.

!!! warning "Support the arm after a trip"

    A motor that has lost torque may not hold the pose. Support the arm if needed, find the cause, then restart.
    Quitting still returns to zero slowly, so the arm does not drop when the motors are disabled.

??? note "Thresholds, telemetry and recorded fields"

    The three temperature settings are conservative macbot thresholds. The
    [RS-06 manual](https://files.seeedstudio.com/products/RobStride/Product%20Literature/RS06/RS06User%20Manual251112.pdf)
    describes a 145 °C thermistor fault threshold, not a recommended working temperature for the arm.

    The B601-RS reuses its existing position feedback: one temperature in °C, reported torque in N·m (motor sign),
    status bits and any cached fault and warning words. This feedback has no separate rotor temperature or current.
    Other hardware may have no motor telemetry. Fake and simulated values are synthetic. For generic followers,
    tracking uses the robot's native joint units.

    Recordings interrupted by the watchdog go to `watchdog/` under the data directory. `health` lists them
    separately, and they stay out of ordinary results and exports. NPZ files store `motor_temp_c`,
    `motor_rotor_c`, `motor_torque_nm`, `motor_status`, `motor_fault` and `motor_warning` in `joint_names` order
    (`NaN` or `-1` when unavailable), plus `watchdog_events` in `meta`.

    `safety.enabled: false` turns off the watchdog decisions and keeps the telemetry and temperature display.

## 3. Export

```bash
macbot export                              # keep the recorded actions in robot units
macbot export --gripper binary             # gripper action 0 (open) or 1 (closed), writes gripper.json
```

Episodes that failed the checks are skipped. Add `--skip-warnings` to skip episodes with warnings too.

For pick-and-place, the binary gripper works better than the raw angle.

- **Labels.** "Closed" starts when the leader reading rises above `--close-above` (−40°) and lasts until it falls
  below `--open-below` (−80°).
- **Deployment.** "Open" is sent as `--open-deg` (−150°) and "closed" as `--close-deg` (0°). The gripper force clamp
  limits how hard it squeezes.

## 4. Train

Training runs on a Linux machine with NVIDIA GPUs. Copy the dataset folder there, then run:

```bash
DATASET=~/data/cubes_to_box_bin GPUS=0,1,2,3 ./train/train_pi05.sh cubes_pi05
DATASET=~/data/cubes_to_box_bin ./train/train_act.sh cubes_act
```

- `train_pi05.sh` fine-tunes `lerobot/pi05_base` through `train/train_augmented.py`. That script is
  `lerobot-train` plus openpi-style image augmentation on the GPU: crop and rotation on the scene camera, color
  jitter on all cameras.
- `WRIST_SHIFT=0.03` adds a small shift to the wrist camera. It helps when the wrist mount moved between
  recording sessions.

Copy `checkpoints/<step>/pretrained_model` back to the Mac. For a binary-gripper dataset, put `gripper.json` next
to it, or one or two folders above it.

!!! tip "Close the gripper earlier with `--grip-lead`"

    The binary label switches to "closed" only when the leader passes `--close-above`, when the fingers are already
    mostly shut. A policy trained on it commands the close about that late, and the real gripper then still has
    its whole travel ahead (about 0.5 s on the reBot soft gripper). `run --grip-lead 0.35` also closes when the
    policy plans to close within the next 0.35 s of the current chunk. Opening is unchanged. Each trial records the
    look-ahead value as `gp_ahead` next to `gp`.

## 5. Run a policy

```bash
macbot run --policy ~/policies/cubes_pi05/004000 --speed 0.3
```

The run console opens in the browser (port 8767 when it is free) and starts **paused**.

| Key | Action |
|---|---|
| ++s++ | Start |
| ++space++ | Pause and hold the current command |
| ++h++ | Move slowly to the start pose |
| ++y++ / ++n++ | Save the trial as success or failure |
| ++t++ | Take over by hand, or hand back to the policy (needs `--takeover`) |
| ++m++ | Mark when a hand enters the scene; control does not change |

Each trial is saved to `rollouts/<task>/` with all frames, the commanded and measured joints, chunk start times
and inference times. To time inference alone on this Mac, run `macbot bench --policy DIR`; it uses blank images.

**Safety limits.** Arm joints move at most `run.max_vel × speed` degrees per second and stay within the joint
limits. `--speed` scales time: at 0.5, a 30 fps policy plays at 15 steps per second. **Home** moves at
`run.home_vel` (default 25 degrees per second for the farthest joint, independent of `--speed`), and all arm joints
arrive together after a short ramp. If a camera stops for 0.5 s, the run pauses.

### Review trials in the run console

While paused, click a row under **Saved trials** to replay it. The list starts with the five newest trials of
this task, including earlier sessions; **Show all** opens the full list. Scrub or play the synchronized cameras and
joint traces; the position readout shows the current chunk and its inference time. **Back to live** returns to the
cameras. Starting or homing closes the player.

In the replay panel you can change the **Result**, add a one-line **Note**, or select **Exclude trial** with a
reason (for example, GPU contention), then click **Save**. Excluded trials stay viewable but are left out of success
totals, health summaries and exports. Edits are stored in `rollout_XXX.review.json` beside the recording, and the
recording itself is unchanged. Keep the two files together when you copy a reviewed trial.

### Execution modes

`--mode` chooses how chunks are executed. All modes interpolate between policy steps for the 60 Hz control loop.

| Mode | What it does | Default for |
|---|---|---|
| `chunk` | Infer, execute the first `--k` steps, infer again. The arm waits during inference. | |
| `ensemble` | Infer continuously. Each target is the exp(m · age) weighted mean of all chunks that cover the current time (ACT's temporal ensembling). | ACT |
| `async` | Infer in the background while the current chunk keeps running. The new chunk is aligned in time with the old one. | |
| `rtc` | `async` plus Real-Time Chunking: the unexecuted part of the old chunk guides the denoising of the new one, so consecutive chunks join smoothly. | π0.5 |

`async` and `rtc` query again as soon as the previous inference ends, so each chunk runs for only a few steps.
`--query-every K` queries every `K` policy steps instead, like the execution horizon in the RTC paper. `rtc` guides
the new chunk with weight 1 on the actions executed during inference and a weight that decays to zero at
`--rtc-horizon`. `--rtc-schedule exp` uses the paper's exponential decay instead of the linear default. With a
50-step chunk, `--query-every 15 --rtc-horizon 35 --rtc-schedule exp` matches the paper.

### Manual takeover

Takeover lets you correct the policy with the leader arm in the middle of a trial. Enable it with `--takeover` or
`run.takeover: true`; this connects the calibrated leader configured for `collect`.

1. Press ++t++ or **Take over** while the policy runs. The follower holds its current command.
2. Move the leader by hand until it matches the follower. The alignment panel shows what still has to match.
3. The follower then follows the leader, starting softly.
4. Press ++t++ again to hand back. The policy plans again from the current observation.

Pause cancels manual control, and ++s++ then resumes the policy. Home, save and quit still work. `--fake` and the
simulator use scripted leaders that must pass the same alignment check.

??? note "Alignment and following in detail"

    **Arm joints** must each be within `run.takeover_align_deg` (default 30°).

    **Gripper.** With a binary policy, the leader must be above `export.gripper_close_above` if the follower is
    commanded closed, or below `export.gripper_open_below` if it is commanded open. A leader reading between the
    two thresholds keeps the follower on hold. With a raw gripper, the leader must be within
    `run.takeover_gripper_align_deg` (default 15°) of the follower command.

    **Soft start.** The follower moves along a straight line at `run.home_vel` and speeds up to the normal limit
    (`run.max_vel × speed`) within one second, or sooner once every joint is within 2°. The recording labels the soft
    start as alignment (`2`); human frames (`1`) start after it.

    **While following**, the arm uses the policy's speed and joint limits, and the gripper command moves at most
    `300 × speed` degrees per second. The reBot gripper keeps the collection force clamp. Leader readings that the
    servo library holds (near 0°, such as the closed gripper) are used as they are, as in `collect`.

    **Leader dropouts.** A missing leader reading holds the follower. If the reading returns within 0.5 s, the
    follower keeps following after a new soft start; a longer dropout returns to alignment.

??? note "What a takeover trial records"

    The trial adds `takeover` (`0` policy, `1` following, `2` aligning and holding) and `leader` (one reading per
    joint and frame, `NaN` when unavailable). Metadata records `takeover_count` (requests) and `takeover_seconds`
    (time actually following). ++m++ or **Mark hand in scene** adds a timestamp to `hand_marks` while a trial is open,
    even without takeover. Replay shades aligning and following on the timeline and traces, and marks each hand
    entry.

## 6. Review

```bash
macbot replay                                   # open the URL printed in Terminal
macbot video ~/macbot-data/rollouts/cubes_to_box/rollout_003.npz --height 720
macbot health --data ~/macbot-data/rollouts/cubes_to_box
```

Standalone replay uses the same player and annotations as the run console. When `health --data` points to a
rollout folder, it shows the reviewed result, the note and the grasp misses of each trial. Replay marks the start
of each detected empty close.

??? note "How empty grasps are detected"

    The detector compares the measured gripper near the end of a commanded close, before it reopens, with two
    calibrated levels. It skips a close already in progress when recording begins and a last close that never
    reopens. A partial or unsettled close is inconclusive.

    The built-in profile applies to real reBot recordings with binary endpoints near −120° and 0° (the reference soft
    gripper). On other setups, set `robot.grasp_miss: {held: ..., empty: ..., tolerance: ...}` in the gripper's
    units, or `false` to turn detection off. Define `close_above` and `open_below` inside that profile if the
    command thresholds differ from the export config. Without calibration, or with overlapping tolerance bands, the
    result is unknown.

### Export reviewed trials for fine-tuning

```bash
macbot export --rollouts --data ~/macbot-data/rollouts/cubes_to_box --out ~/data/cubes_rollouts
macbot export --with-rollout-labels --data ~/macbot-data/raw/cubes_to_box --out ~/data/cubes_demos
```

Rollout exports use the same `observation.state`, `action`, camera keys, image sizes and framing metadata as
demonstrations, so the two can be merged. They keep successes, failures and corrections, and skip excluded trials.

| Column or field | Meaning |
|---|---|
| `takeover` | per frame: `0` policy, `1` human, `2` aligning |
| `grasp_miss` | per frame: inside a detected empty close |
| `hand_mark` | per frame: within ±0.5 s of a hand mark |
| `success` | reviewed result of the trial (episode metadata) |
| `source` | `"rollout"`, or `"demonstration"` with `--with-rollout-labels` |
| grasp counts | `grasp_miss_count=null` means the trial has no usable calibration |

A zero frame label means no miss was flagged. `--with-rollout-labels` gives a binary demonstration export the same
columns with zero frame labels, `source="demonstration"` and `success=true`; the standard demonstration export is
unchanged.

??? note "How actions are labelled in rollout exports"

    Arm actions are the saved follower commands. For binary policies, the gripper is `1` when `gp ≥ 0.5` and `0`
    otherwise. During human control, the leader reading goes through the same `--close-above` and `--open-below`
    hysteresis as a demonstration export (defaults from `export`). Holding frames and frames without valid policy
    output use the saved gripper command; raw policies also use command hysteresis. When `gp` is available, the
    label expresses the policy's intent, so it can differ briefly from the physical command while the deploy-time
    open debounce is active.

!!! warning "Before merging with demonstrations"

    Match fps, robot and joint order, camera keys, image sizes and `meta/camera_framing.json`. Older trials may lack
    framing fields; verify their setup before merging. Keep the verified framing and the `gripper.json` sidecars with
    the merged dataset. Compute advantages on the training machine.

### Export training segments for DAgger and advantage conditioning

```bash
macbot export --rollouts --segments --min-segment-s 1.0 --pre-takeover-s 1.7
```

Without `--segments`, each trial is exported as one episode. With it, the export applies three rules.

1. **Drop alignment frames** (`takeover=2`). Alignment, holds, soft starts and leader dropouts record how the
   control system hands over, which should not become a training target.
2. **Split at handovers.** Each uninterrupted policy run (`0`) or human run (`1`) becomes its own episode, so action
   chunks stay on one side of a handover, including an immediate return from human to policy. Frame indices and
   timestamps restart at zero in each episode. LeRobot pads chunks at the episode end; use `action_is_pad` to mask
   padding or select only full chunks in training.
3. **Mark the lead-up to an intervention.** `pre_takeover` (`uint8`) marks policy frames within `--pre-takeover-s`
   of the next takeover request in the original trial (default 1.7 s, about 50 frames at 30 fps). Recordings have
   no button timestamps, so the window ends at the first sampled transition from policy to alignment or human
   control. A leader dropout during human control does not start another window. Set the window to `0` to turn
   marking off.

Runs shorter than `--min-segment-s` (default 1.0 s, measured as `frames / fps`) are dropped.

??? note "Segment metadata and merging"

    Each remaining episode has `source="rollout_policy"` or `"rollout_human"`, `trial_name` (the NPZ filename),
    `original_start_time` (seconds from the trial start) and `segment_index` (zero-based among policy and human runs,
    counted before short runs are dropped). The reviewed `success` and the grasp counts refer to the whole trial.
    Excluded trials stay excluded, and `grasp_miss` and `hand_mark` stay aligned to the retained frames. The summary
    reports the included trials, segments by source, dropped alignment frames and short segments, and the retained
    human seconds and pre-takeover frames.

    Segment exports add a column that `--with-rollout-labels` does not write. For a schema-matched merge with
    demonstrations, add a zero `uint8` `pre_takeover` column with shape `(1,)` to the demonstrations.

## Without hardware

**Fake hardware.** `collect`, `run` and every console work with `--fake`: a first-order-lag arm and synthetic
cameras. With `run --fake` the policy is real, so it checks loading, compilation and the executors before you touch
the robot. `--autostart SECONDS` runs, saves a trial and quits on its own. Fake recordings go under `<task>_fake`;
pass `--data` to `export` or `health` and `--dir` to `replay` to read them. For followers that macbot does not
list, set `robot.joints` to the ordered joint names when you use `--fake`.

**Simulation.** Install the [MuJoCo extra](setup.md#local-mujoco-simulator) and use the same config for every step:

```bash
macbot --config configs/so101_sim.yaml collect
macbot --config configs/so101_sim.yaml health
macbot --config configs/so101_sim.yaml export
macbot --config configs/so101_sim.yaml run --policy path/to/pretrained_model
macbot --config configs/so101_sim.yaml replay
```

Press ++s++ to reset the scene and record the scripted leader's motion. This is test data for the pipeline; the
script does not solve the task. The saved episode can be replayed in the collection console. For policy runs, use a
checkpoint with six joint state and action values in the same order and units, and with matching camera keys and
sizes (`front` and `wrist`, both 320×240 in the example). **Start** resets the scene for a new trial. **Pause**
freezes it; save or discard the trial before you use **Reset scene**. `--autostart SECONDS` also works in
simulation. Keep simulated and hardware data in separate task folders.

The [Simulation](simulation.md) page covers Cartesian keyboard control, scripted demonstrations, training on a CUDA
machine and seeded policy evaluation on the Mac.

## For developers

### Add an execution mode

Executors live in `macbot/policy/executors.py` and share a four-method interface:

```python
class MyExecutor(Executor):
    def start(self): ...                  # start threads, if any
    def reset(self, active: bool): ...    # drop chunks; active=True: plan from the current observation
    def target(self, now: float): ...     # -> (joint target, chunk id) or (None, id)
    def describe(self) -> str: ...
```

`AsyncChunk.accept()` decides whether, and at which step, to switch to a newly inferred chunk. Subclass it to try
new switching rules without touching the control loop.

### Measure progress regression at chunk switches

The arm lags its command by a few control steps, so a policy that plans from the measured arm can return a chunk
that starts behind the current command. To measure this, `chunk`, `async` and `rtc` save every switch in the trial:

| Field | Content |
|---|---|
| `ex_chunk` | the new chunk |
| `ex_prev` | the chunk it replaces (`NaN` for the first chunk) |
| `ex_p` | the index in the old chunk that the new chunk's first action replaces |
| `ex_entry` | the index at which the new chunk starts to run |
| `ex_q_obs` | the measured state that the policy saw |

This adds about 1 MB per minute of trial. Then run:

```bash
python tools/chunk_shift.py ~/macbot-data/rollouts/<task>
```

For each trial and execution mode, it prints the shift `delta_hat` that best aligns each new chunk with the old one
(positive when the new chunk starts behind the command), once over the first 10 actions and once after the prefix
that `rtc` guides onto the old chunk, together with the tracking lag between command and measured arm.
