# Setting up macbot on a Mac

[中文](setup_zh.md)

For guided setup, start with the [quickstart](quickstart.md). The manual workflows below are
useful for custom profiles and troubleshooting.

## 1. Python environment

```bash
conda create -n macbot python=3.12 -y && conda activate macbot
pip install macbot                 # use macbot[pi] for pi0.5 dependencies
xcode-select --install             # swiftc, used to build the UVC camera helper on first use
```

For an Orbbec wrist camera, see the SDK installation notes in the [quickstart](quickstart.md).

## 2. CAN driver (PCAN-USB)

Download the **PCBUSB** library for macOS from [mac-can.com](https://www.mac-can.com) and run its
installer. It places `libPCBUSB.dylib` in `/usr/local/lib`. Plug in the PCAN-USB adapter, power the
follower, and check with `macbot doctor`.

## 3. Calibration

macbot never re-zeroes the arms; it always connects with `calibrate=False` and refuses to connect
without a calibration file. Calibrate once with LeRobot, using the ids from your config
(`robot.follower_id`, default `follower1`, and `robot.leader_id`, default `rebot_arm_102_leader`):

```bash
lerobot-calibrate --robot.type=rebot_b601_follower --robot.id=follower1 --robot.port=can0 \
  --robot.motor_family=rs --robot.can_adapter=socketcan
lerobot-calibrate --teleop.type=rebot_102_leader --teleop.id=rebot_arm_102_leader --teleop.port=/dev/cu.usbserial-XXXX
```

The files land in `~/.cache/huggingface/lerobot/calibration/`. Keep a copy.

## 4. Cameras

```bash
macbot cameras
```

lists every video device with its AVFoundation uniqueID. Put the scene camera's ID into your config:

```yaml
cameras:
  top:
    type: uvc
    unique_id: "0x1100001234abcd"
    crop: [625, 140, 1610, 1080]    # x0 y0 x1 y1 on the full frame; [0, 0, 0, 0] keeps it whole
    save_size: [492, 470]
```

Do not rely on camera names or indices: with an iPhone nearby, macOS may hand you the Continuity Camera.
A USB camera's uniqueID encodes the port; if you move the camera, macbot matches it by vendor and product.

The Orbbec SDK opens USB devices as root, so commands that use an Orbbec camera run under `sudo -E`
(`-E` keeps your conda environment). Files written under sudo are given back to your user.

## 5. Gravity feedforward (optional)

`robot.gravity_compensation` adds a gravity torque to each MIT command, which noticeably reduces sag on
the shoulder and elbow. It needs the arm's URDF from Seeed's reBot description package (not included
here). Set `robot.urdf` to its path. Without it macbot runs with velocity feedforward only.

## 6. First run

```bash
macbot collect --fake                                  # check the page without hardware
sudo -E $(which python) -m macbot.cli joints           # read both arms; nothing moves
sudo -E $(which python) -m macbot.cli collect          # the real thing
```

Before `collect`, put both arms near the zero pose: macbot refuses to start teleoperation when leader
and follower differ by more than `teleop.align_max_deg`.

To launch from Finder, see `scripts/`: double-click `collect.command` or `run.command`.

## LeRobot arms with UVC cameras

Use a Python 3.12 environment and install macbot plus the LeRobot 0.7.0 driver extra. For SO-100/101:

```bash
pip install macbot 'lerobot[feetech,core-scripts]==0.7.0'
xcode-select --install
macbot robots
lerobot-find-port
macbot cameras
```

Start from [configs/so101.yaml](../configs/so101.yaml). Set `robot.type: lerobot`, choose the
`follower.type` and `leader.type` from `macbot robots`, and fill in each device's LeRobot
options. The list shows registered types; driver and OS support depend on the device. Calibrate with
`lerobot-calibrate` using the same device type, port and ID as in the config. macbot uses the
saved calibration, and the arms must start aligned. Keep the leader and follower in the same
joint units. For a different robot, configure `robot.limits` in its native units.

Replace the `cameras` section with only your UVC devices and set their unique IDs. No Orbbec SDK or
`sudo` is needed for the SO-101/UVC example. macbot builds the UVC helper with Swift on first use;
grant camera access when macOS asks. Run `macbot --config my.yaml collect` as your normal user.

## Local MuJoCo simulator

```bash
pip install -e '.[sim]'                  # from the source checkout
macbot --config configs/so101_sim.yaml collect
```

The example uses Python 3.12 on Apple silicon, MuJoCo 3.3.7, offscreen rendering, and a scripted leader.
The first launch downloads the pinned SO-101 EnvHub code, scene and model assets. The example
explicitly enables `trust_remote_code`; use only EnvHub code you have reviewed and trust. A cached
snapshot can run offline. No camera, serial device, CAN adapter or `sudo` is needed.

`robot.env` takes a local LeRobot environment config with absolute joint position actions and matching
`.pos` state names. Set `teleop.hz` and `run.hz` to the environment's control rate (30 for the example).
For `type: robot` cameras, `key` selects a rendered observation and defaults to the camera name.
Use `save_size` for the output dimensions and optional `crop: [x0, y0, x1, y1]` before resizing.
See the [guide](guide.md#simulation) for policy and replay commands.
