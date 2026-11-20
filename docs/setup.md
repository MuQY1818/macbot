# Setting up a Mac for the real robot

[中文](setup_zh.md)

Everything here is one-time. `macbot doctor` tells you which steps are still missing.

## 1. Python environment

```bash
conda create -n macbot python=3.12 -y && conda activate macbot
pip install "macbot[orbbec,pi]"    # orbbec: wrist camera SDK, pi: pi0.5 dependencies
xcode-select --install             # swiftc, used to build the UVC camera helper on first use
```

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

The Orbbec SDK opens USB devices as root, so commands that use the wrist camera run under `sudo -E`
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
