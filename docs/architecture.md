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

`macbot/cli.py` selects a command and `macbot/config.py` overlays a YAML file on `macbot/default.yaml`.
The `cameras` map is replaced as a whole. `macbot/web/` supplies the HTTP handlers, templates and
shared assets for setup, collect, run and replay. `macbot/init_wizard.py` assembles the setup profile;
`robot/usb.py` and `cameras/discovery.py` provide device discovery and preview. The setup console
hands off to the separate `collect` command after releasing its previews.

| Module | Responsibility |
|---|---|
| `robot/` | Opens the reBot profile, a generic LeRobot arm, a simulator or a fake arm. |
| `cameras/` | Provides UVC (Swift/AVFoundation), Orbbec, fake and robot-observation camera sources. |
| `collect/` | Drives the leader-to-follower loop, records joint commands and observations on a fixed time grid, and checks each episode. |
| `data/`, `train/` | Inspects and exports raw episodes to LeRobot v3; example training scripts produce policy checkpoints. |
| `policy/`, `viz/` | Loads LeRobot policies, controls the arm and records trials; replays those rollout files or exports them as MP4. |

The [`Robot`](../macbot/robot/base.py) interface exposes ordered joint names, optional gripper and
limits, `connect(leader=...)`, `read()`, `read_leader()`, `send()`, `home_pose()` and `shutdown()`.
`send()` returns the commanded values, after any robot-specific limiting. Episode metadata records
the joint layout and robot type. Joint units are defined by the driver.

The [camera interface](../macbot/cameras/__init__.py) is structural: `latest()` returns a BGR frame
and timestamp (or `None` before the first frame), `age()` exposes staleness, and `close()` cleans
up resources. A `type: robot` camera reads a named simulator observation; the rest of the pipeline
uses it like any other camera.

The runner separates inference from the control loop through an
[`Executor`](../macbot/policy/executors.py). Synchronous chunks predict on demand, temporal
ensembling combines overlapping predictions, and asynchronous chunks predict in a background
thread. Real-Time Chunking adds guidance to the asynchronous path. `AsyncChunk.accept()`
decides when a new chunk takes effect. The runner handles start/pause, velocity limits and
recording independently of the executor.

The generic adapter loads a LeRobot policy and its saved processor pair. It uses the upstream
chunk contract when available and runs those chunks synchronously. Otherwise it calls
`select_action` on each policy tick, so LeRobot manages observation history and queued actions.
ACT and π0.5 keep their original wrappers and execution defaults.

`policy/backends.py` distinguishes implemented modes from configurations qualified for
`--backend auto`. It checks checkpoint metadata and a static table without timing inference.
The runner records the decision in trial metadata.
