# Architecture

[简体中文](architecture.zh.md)

```text
Robot + cameras -> collect -> raw NPZ -> export -> LeRobot dataset
                                                       |
                                                     train
                                                       |
Robot + cameras <-> run <- policy checkpoint <---------+
                     |
                 rollout NPZ -> replay / video
```

## Commands and configuration

`macbot/cli.py` dispatches the commands. `macbot/config.py` lays the user's YAML over `macbot/default.yaml`; the
`cameras` map is replaced as a whole rather than merged. `macbot/web/` holds the HTTP handlers, templates and shared
assets of the four consoles (setup, collect, run and replay).

The setup wizard (`macbot/init_wizard.py`) builds the profile, using `robot/usb.py` and `cameras/discovery.py` to find
devices and show previews. When you start collection from the wizard, it releases its previews and hands over to the
separate `collect` command. `lerobot_command.py` parses an imported LeRobot command as plain data, and `hub.py`
resolves Hub IDs to pinned files in the standard cache.

## Modules

| Module | What it does |
|---|---|
| `robot/` | Opens the reBot profile, a generic LeRobot arm, the simulator or a fake arm |
| `cameras/` | UVC cameras (Swift and AVFoundation), Orbbec, fake cameras and simulator observations |
| `collect/` | The leader-to-follower loop, recording on a fixed time grid, episode checks and scripted simulator demos |
| `data/` | Health reports, empty-grasp detection, reading LeRobot datasets and exporting to LeRobot v3.0 |
| `policy/` | Loading policies, the executors, the run loop, evaluation and the MLX backends (`mlx_*.py`) |
| `viz/` | Replay of trials and MP4 export |
| `safety.py` | Motor temperature and tracking watchdog |
| `doctor.py` | Checks of drivers, devices, calibration and simulator setup |
| `train/` (top level) | Example training scripts for the GPU machine |

## Extension points

### Robots

The [`Robot`](../macbot/robot/base.py) interface exposes ordered joint names, an optional gripper and limits,
`connect(leader=...)`, `read()`, `read_leader()`, `send()`, `home_pose()` and `shutdown()`. `send()` returns the
values that were actually commanded, after any robot-specific limiting. The driver defines the joint units, and each
episode records the joint layout and robot type. [Robots](robots.md#adding-a-robot-class) describes how to add one.

### Cameras

The [camera interface](../macbot/cameras/__init__.py) is structural. `latest()` returns a BGR frame and its timestamp,
or `None` before the first frame; `age()` reports how stale the frame is; `close()` frees the device. A `type: robot`
camera reads a named simulator observation, and the rest of the pipeline treats it like any other camera.

### Execution modes

The runner keeps inference out of the control loop through an [`Executor`](../macbot/policy/executors.py).
Synchronous chunks predict on demand, temporal ensembling averages overlapping predictions, and asynchronous chunks
predict in a background thread. Real-Time Chunking adds guidance to the asynchronous path, and `AsyncChunk.accept()`
decides when a new chunk takes over. Start and pause, velocity limits and recording live in the runner, independent
of the executor. The [Workflow guide](guide.md#add-an-execution-mode) shows the interface.

## Policies and backends

The generic adapter loads a LeRobot policy together with its saved processor pair. If the policy exposes LeRobot's
chunk interface, the adapter runs those chunks synchronously; otherwise it calls `select_action` on every policy tick
and leaves observation history and action queues to LeRobot. ACT and π0.5 keep their own wrappers and default
execution modes.

`policy/backends.py` separates the backends a policy can run on from the configurations qualified for
`--backend auto`. The decision uses the checkpoint metadata and a fixed table, without timing anything, and the runner
records it in each trial.
