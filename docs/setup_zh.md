# 让 Mac 连上真机

[English](setup.md)

以下步骤都只需做一次。`macbot doctor` 会告诉你还缺哪一步。

## 1. Python 环境

```bash
conda create -n macbot python=3.12 -y && conda activate macbot
pip install -e ".[orbbec,pi]"      # orbbec：腕部相机 SDK；pi：π0.5 依赖
xcode-select --install             # 提供 swiftc，首次使用 UVC 相机时编译助手程序
```

## 2. CAN 驱动（PCAN-USB）

从 [mac-can.com](https://www.mac-can.com) 下载 macOS 版 **PCBUSB** 库并运行安装程序，它会把 `libPCBUSB.dylib` 装到 `/usr/local/lib`。插上 PCAN-USB，给从臂上电，用 `macbot doctor` 检查。

## 3. 标定

macbot 从不重新校零：连接时一律 `calibrate=False`，没有标定文件就拒绝连接。请先用 LeRobot 标定一次，id 与配置里一致（`robot.follower_id` 默认 `follower1`，`robot.leader_id` 默认 `rebot_arm_102_leader`）：

```bash
lerobot-calibrate --robot.type=rebot_b601_follower --robot.id=follower1 --robot.port=can0 \
  --robot.motor_family=rs --robot.can_adapter=socketcan
lerobot-calibrate --teleop.type=rebot_102_leader --teleop.id=rebot_arm_102_leader --teleop.port=/dev/cu.usbserial-XXXX
```

标定文件在 `~/.cache/huggingface/lerobot/calibration/`，记得备份。

## 4. 相机

```bash
macbot cameras
```

会列出所有视频设备和它们的 AVFoundation uniqueID。把俯视相机的 ID 写进配置：

```yaml
cameras:
  top:
    type: uvc
    unique_id: "0x1100001234abcd"
    crop: [625, 140, 1610, 1080]    # 在整幅画面上的 x0 y0 x1 y1；[0, 0, 0, 0] 表示不裁剪
    save_size: [492, 470]
```

不要按相机名字或编号打开：附近有 iPhone 时，macOS 可能给你的是连续互通相机。USB 相机的 uniqueID 里带着端口信息，换了 USB 口后 macbot 会按厂商和型号重新匹配。

Orbbec SDK 要以 root 身份打开 USB 设备，所以用到腕部相机的命令要加 `sudo -E`（`-E` 保留 conda 环境）。sudo 下写出的文件会交还给你的用户。

## 5. 重力前馈（可选）

`robot.gravity_compensation` 在每条 MIT 指令里加上重力矩，能明显减少肩、肘关节下垂。它需要 Seeed reBot 描述包里的 URDF（本仓库不附带），把 `robot.urdf` 设成它的路径即可。没有 URDF 时只用速度前馈。

## 6. 第一次运行

```bash
macbot collect --fake                                  # 不接硬件，先看网页
sudo -E $(which python) -m macbot.cli joints           # 读两条臂的关节角，不会动
sudo -E $(which python) -m macbot.cli collect          # 正式采集
```

开始采集前，把两条臂都摆到零位附近：主从臂相差超过 `teleop.align_max_deg` 时，macbot 不会开始遥操作。

想在访达里双击启动，见 `scripts/` 里的 `collect.command` 和 `run.command`。
