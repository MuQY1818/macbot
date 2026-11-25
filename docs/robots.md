# Robots

[简体中文](robots.zh.md)

| `robot.type` | Robot | Status |
|---|---|---|
| `rebot_b601` | Seeed reBot B601-RS follower and Arm 102 leader | Tested on hardware on Apple silicon |
| `lerobot` | SO-100/101, Koch, OMX, OpenArm, YAM and other compatible LeRobot arms | Fake mode only; no hardware validation yet |
| `sim` | SO-101 tabletop in MuJoCo | Whole workflow runs end to end |

- **`lerobot`** arms are configured through LeRobot. In fake mode, the SO-101 collection and export path was exercised.
  Other arms have a built-in joint layout or can supply `robot.joints`.
- **`sim`** covers keyboard teleoperation, scripted demonstrations, export, ACT training and seeded evaluation.
  Synthetic keyboard input completed the task, and a person placed one cube by hand. A trained policy that solves the
  task has not been shown yet.

!!! note "`macbot robots` is not a list of tested arms"

    `macbot robots` lists the LeRobot types registered by the installed packages, not the arms that have been
    validated.

## Requirements for any arm

- The follower takes joint-position (`.pos`) actions and reports the same joints.
- The leader used for collection has the same joint names as the follower.
- A policy checkpoint matches the joint order, units and camera inputs of the arm it runs on.

## Generic LeRobot arms

[configs/so101.yaml](../configs/so101.yaml) is a complete SO-101 example with one UVC camera. Install the LeRobot
extra for your arm: `lerobot[feetech]==0.7.0` for SO-100/101, `lerobot[dynamixel]==0.7.0` for Koch and OMX.

```yaml
robot:
  type: lerobot
  follower: {type: so101_follower, port: /dev/tty.usbmodem_FOLLOWER, id: follower}
  leader: {type: so101_leader, port: /dev/tty.usbmodem_LEADER, id: leader}
  limits: {gripper: [0, 100]}
```

| Arm | `follower.type` | `leader.type` |
|---|---|---|
| SO-100/101 | `so101_follower` | `so101_leader` |
| Koch | `koch_follower` | `koch_leader` |
| OMX | `omx_follower` | `omx_leader` |

For other arms, replace both types and fill in the fields that their LeRobot configs require. If fake mode has no
built-in layout for the arm, list its joints in order under `robot.joints`. Limits and binary gripper thresholds are
in the driver's units. Put cameras under the top-level `cameras`, and leave out `leader` if you only run policies.
Use the same IDs and calibration folders for recording and for running.

## reBot B601-RS

Start from [configs/example.yaml](../configs/example.yaml) and the reBot section of [Setup](setup.md#rebot-b601-rs-reference-setup).
The reference profile reaches the follower through a PCAN-USB adapter with the PCBUSB driver and finds the Arm 102
leader's serial port on its own. Gravity feedforward also needs the arm's URDF.

```yaml
robot:
  type: rebot_b601
  can_channel: can0
  follower_id: follower1
  leader_id: rebot_arm_102_leader
```

## Local simulation

[Simulation](simulation.md) covers setup, keyboard teleoperation, scripted demonstrations and evaluation. The complete
config is [configs/so101_sim.yaml](../configs/so101_sim.yaml); it drives the arm with a sine wave for smoke tests.
Run collection without `--fake`, and keep the control rates equal to the scene's. The pinned EnvHub scene downloads
and runs Python code on first use.

```yaml
robot:
  type: sim
  env:
    type: envhub
    hub_path: lerobot/so101-tabletop-mujoco@4684f500b94d07194f2c860314c4e27cb9349e8c
    trust_remote_code: true
cameras:
  front: {type: robot, save_size: [320, 240]}
  wrist: {type: robot, save_size: [320, 240]}
```

## Calibration and checks

macbot always connects with `calibrate=False`, so arms must be calibrated with LeRobot first. `macbot init` looks for
the calibration files and shows the matching `lerobot-calibrate` commands. Run them separately with the same device
types, ports, IDs and optional `calibration_dir`, using `--robot.*` for the follower and `--teleop.*` for the leader.
Simulation without a physical leader, and fake mode, need no calibration.

`macbot --config my.yaml doctor` checks the selected profile without moving anything:

- For generic arms, it checks that the drivers import, the serial paths exist and the calibration files are present,
  without opening a device.
- For the simulator, it checks the config and the MuJoCo import, without loading the scene.

A passing check does not mean the calibration is good or that the arm moves correctly.

## Adding a robot class

If an upstream LeRobot driver already fits, use `robot.type: lerobot`. Otherwise:

1. Implement the [`Robot` interface](../macbot/robot/base.py): ordered `joints`, optional `gripper` and `limits`,
   `info()` metadata, and the connection, observation, command and shutdown behavior of the device.
2. Register the profile in [`open_robot`](../macbot/robot/__init__.py).
3. Keep the calibration guard, and document what happens on shutdown.
4. Add a matching fake layout, an example config and tests that need no hardware.
