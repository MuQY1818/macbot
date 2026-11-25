# Setup

[简体中文](setup.zh.md)

The [Quickstart](quickstart.md) wizard writes a working config for most setups. This page is for writing or editing
a config by hand, for the reBot reference setup and for troubleshooting.

## Python environment

```bash
python3.12 -m venv ~/.venvs/macbot && source ~/.venvs/macbot/bin/activate
pip install macbot
xcode-select --install       # swiftc, used to build the UVC camera helper on first use
```

Any Python 3.12 environment works, including conda. Optional extras add dependencies:

| Extra | Adds |
|---|---|
| `macbot[pi]` | LeRobot's π0 and π0.5 dependencies |
| `macbot[mlx]` | the same plus MLX, for the [accelerated backends](acceleration.md) |
| `macbot[orbbec]` | the Orbbec SDK (see the version note under [Special hardware](quickstart.md#special-hardware)) |
| `macbot[sim]` | MuJoCo for the [local simulator](#local-mujoco-simulator) |

## LeRobot arms with UVC cameras

This covers SO-100/101, Koch, OMX and other arms with a LeRobot 0.7.0 driver. Install the driver extra for your arm.
For SO-100/101:

```bash
pip install macbot 'lerobot[feetech,core-scripts]==0.7.0'
macbot robots            # registered follower and leader types
lerobot-find-port        # serial ports of the two arms
macbot cameras           # cameras and their unique IDs
```

1. Start from [configs/so101.yaml](../configs/so101.yaml) and set `robot.type: lerobot`.
2. Choose `follower.type` and `leader.type` from `macbot robots` and fill in each device's LeRobot options. The list
   shows registered types; driver and OS support depend on the device.
3. Calibrate with `lerobot-calibrate`, using the same device type, port and ID as in the config. macbot uses the
   saved calibration.
4. Keep the leader and follower in the same joint units. For a different robot, set `robot.limits` in its native
   units.
5. Replace the `cameras` section with your UVC devices and their unique IDs (see [Cameras](#cameras)).

Run `macbot --config my.yaml collect` as your normal user; no Orbbec SDK or `sudo` is needed. macbot builds the UVC
helper with Swift on first use, and macOS asks for camera access. The arms must start aligned.

## Cameras

```bash
macbot cameras
```

lists every video device with its AVFoundation unique ID. Put each camera's ID into the config:

```yaml
cameras:
  top:
    type: uvc
    unique_id: "0x1100001234abcd"
    crop: [625, 140, 1610, 1080]    # x0 y0 x1 y1 on the full frame; [0, 0, 0, 0] keeps it whole
    save_size: [492, 470]
```

!!! warning "Use unique IDs, not names or indices"

    With an iPhone nearby, macOS may hand you the Continuity Camera instead of the camera you meant.

A USB camera's unique ID encodes the port. If you move the camera to another socket, macbot matches it by vendor and
product.

## reBot B601-RS reference setup

The reference setup is a reBot B601-RS follower on a PCAN-USB adapter, a reBot 102 leader and an Orbbec wrist camera.

### CAN driver (PCAN-USB)

Download the **PCBUSB** library for macOS from [mac-can.com](https://www.mac-can.com) and run its installer. It
places `libPCBUSB.dylib` in `/usr/local/lib`. Plug in the PCAN-USB adapter, power the follower and check with
`macbot doctor`.

### Calibration

macbot never re-zeroes the arms. It always connects with `calibrate=False` and refuses to connect without a
calibration file. Calibrate once with LeRobot, using the IDs from your config (`robot.follower_id`, default
`follower1`, and `robot.leader_id`, default `rebot_arm_102_leader`):

```bash
lerobot-calibrate --robot.type=rebot_b601_follower --robot.id=follower1 --robot.port=can0 \
  --robot.motor_family=rs --robot.can_adapter=socketcan
lerobot-calibrate --teleop.type=rebot_102_leader --teleop.id=rebot_arm_102_leader --teleop.port=/dev/cu.usbserial-XXXX
```

The files go to `~/.cache/huggingface/lerobot/calibration/`. Keep a copy.

### Orbbec camera and sudo

Install the SDK as described under [Special hardware](quickstart.md#special-hardware). The Orbbec SDK opens USB
devices as root, so commands that use an Orbbec camera run under `sudo -E`; `-E` keeps your Python environment.
macbot gives files written under sudo back to your user.

### Gravity feedforward (optional)

`robot.gravity_compensation` adds a gravity torque to each MIT command, which noticeably reduces sag at the shoulder
and elbow. It needs the arm's URDF from Seeed's reBot description package, which macbot does not include. Set
`robot.urdf` to its path. Without it, macbot uses velocity feedforward only.

### First run

```bash
macbot collect --fake                                  # check the console without hardware
sudo -E $(which python) -m macbot.cli joints           # read both arms; nothing moves
sudo -E $(which python) -m macbot.cli collect          # record with the real arms
```

Before `collect`, put both arms near the zero pose. macbot refuses to start teleoperation when leader and follower
differ by more than `teleop.align_max_deg`.

To start from Finder, double-click `scripts/collect.command` or `scripts/run.command`.

## Local MuJoCo simulator

```bash
pip install -e '.[sim]'                  # from a source checkout
macbot --config configs/so101_sim.yaml collect
```

No camera, serial device, CAN adapter or `sudo` is needed. The example uses Python 3.12 on Apple silicon, MuJoCo
3.3.7, offscreen rendering and a scripted leader.

!!! warning "Remote code"

    The first launch downloads the pinned SO-101 EnvHub code, scene and model assets. The example explicitly enables
    `trust_remote_code`, so use only EnvHub code you have reviewed and trust. A cached snapshot can run offline.

`robot.env` takes a local LeRobot environment config with absolute joint position actions and matching `.pos` state
names. Set `teleop.hz` and `run.hz` to the environment's control rate (30 in the example). For `type: robot` cameras,
`key` selects a rendered observation and defaults to the camera name. `save_size` sets the output size, and an
optional `crop: [x0, y0, x1, y1]` is applied before resizing.

The [Workflow guide](guide.md#without-hardware) lists the policy and replay commands, and [Simulation](simulation.md)
covers keyboard control and evaluation.
