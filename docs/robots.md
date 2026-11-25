# Robot support

[简体中文](robots.zh.md)

| `robot.type` | Robot | Validation in macbot |
|---|---|---|
| `rebot_b601` | Seeed reBot B601-RS follower and Arm 102 leader | Tested on hardware on Apple silicon. |
| `lerobot` | SO-100/101, Koch, OMX, OpenArm, YAM and other compatible LeRobot arms | Configured through LeRobot. Fake mode only: the SO-101 collect/export path was exercised; other arms have layout support or can supply `robot.joints`. No hardware validation. |
| `sim` | SO-101 tabletop in MuJoCo | Keyboard teleoperation (synthetic input completed the task; a person placed one cube by hand), scripted demos, export, ACT training and seeded evaluation run end to end. A trained policy solving the task is not shown. |

Run `macbot robots` to list registered LeRobot types. The list reflects installed packages;
it is not a list of validated arms. Generic followers must accept joint-position (`.pos`)
actions and observe those same joints. A leader used for collection must have the same joint
names. Policy checkpoints must match the joint order, units and camera inputs.

## reBot B601-RS

Start with [`configs/example.yaml`](../configs/example.yaml) and the [setup guide](setup.md).
The reference profile uses PCAN-USB with PCBUSB and auto-detects the Arm 102 serial port.
Gravity feedforward additionally needs a URDF.

```yaml
robot:
  type: rebot_b601
  can_channel: can0
  follower_id: follower1
  leader_id: rebot_arm_102_leader
```

## Generic LeRobot arms

[`configs/so101.yaml`](../configs/so101.yaml) is a complete SO-101 example with one UVC
camera. Install the LeRobot extra for your arm, for example `lerobot[feetech]==0.7.0`
for SO-100/101 or `lerobot[dynamixel]==0.7.0` for Koch and OMX. Replace both device types
for another arm, and supply the fields its LeRobot configs require. Use top-level `cameras`;
omit `leader` for policy inference without a teleoperator.

```yaml
robot:
  type: lerobot
  follower: {type: so101_follower, port: /dev/tty.usbmodem_FOLLOWER, id: follower}
  leader: {type: so101_leader, port: /dev/tty.usbmodem_LEADER, id: leader}
  limits: {gripper: [0, 100]}
```

For Koch use `koch_follower` and `koch_leader`; for OMX use `omx_follower` and `omx_leader`.
For other models, consult the installed LeRobot config and list the ordered `robot.joints`
when fake mode has no built-in layout. Set limits and any binary gripper thresholds in the
driver's units. Reuse the same IDs and calibration directories when recording or running.

## Local simulation

Follow the [simulation guide](simulation.md) for setup, keyboard teleoperation, scripted
demonstrations, and policy evaluation. The complete [`configs/so101_sim.yaml`](../configs/so101_sim.yaml)
keeps scripted sine input for smoke tests. Run collection without `--fake`.
The pinned EnvHub scene downloads and executes Python on first use; the control rates must match it.

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

`macbot init` checks for calibration and supplies the matching `lerobot-calibrate` commands.
Run them separately with the same device types,
ports, IDs and optional `calibration_dir`. Use `--robot.*` for the follower and `--teleop.*`
for the leader; see the [setup guide](setup.md) for commands. macbot connects with
`calibrate=False`. Simulation without a physical leader, and fake mode, need no calibration.

Run `macbot --config my.yaml doctor` to check the selected profile. For generic arms it checks
driver imports, configured serial paths and calibration file presence without opening a device.
For the bundled simulator it checks the local configuration and MuJoCo import without loading
the scene. A passing check does not verify calibration quality or arm motion.

## Adding a robot class

Use `robot.type: lerobot` if the upstream driver already fits. Otherwise implement the
[`Robot` interface](../macbot/robot/base.py) and register the profile in
[`open_robot`](../macbot/robot/__init__.py). Define ordered `joints`, optional `gripper` and
`limits`, and `info()` metadata. Implement the connection, observation, command and shutdown
behavior for that device; preserve the calibration guard and document what happens on shutdown.
Add a matching fake layout, a config example and tests with no hardware.
