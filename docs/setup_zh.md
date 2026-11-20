# 在 Mac 上安装 macbot

[English](setup.md)

第 1–6 节适用于 reBot 参考配置。其他机械臂和无硬件仿真的流程见文末。

## 1. Python 环境

```bash
conda create -n macbot python=3.12 -y && conda activate macbot
pip install "macbot[orbbec,pi]"    # orbbec：腕部相机 SDK；pi：π0.5 依赖
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

Orbbec SDK 要以 root 身份打开 USB 设备，所以用到 Orbbec 相机的命令要加 `sudo -E`（`-E` 保留 conda 环境）。sudo 下写出的文件会交还给你的用户。

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

## LeRobot 机械臂与 UVC 相机

使用 Python 3.12 环境，为 macbot 安装对应的 LeRobot 0.7.0 驱动。以 SO-100/101 为例：

```bash
pip install macbot 'lerobot[feetech,core-scripts]==0.7.0'
xcode-select --install
macbot robots
lerobot-find-port
macbot cameras
```

以 [configs/so101.yaml](../configs/so101.yaml) 为起点。设置 `robot.type: lerobot`，从
`macbot robots` 列出的类型中选择主从臂，并填入各自的 LeRobot 配置。列表显示已注册的类型，所需驱动和操作系统支持视设备而定。先用 `lerobot-calibrate` 标定，设备类型、端口和 ID 必须与配置一致。macbot 使用已有标定；启动时要对齐两臂并使用相同的关节单位。其他机械臂的 `robot.limits` 请按设备自身单位设置。

将整个 `cameras` 配置替换为你使用的 UVC 设备，填入各自的 uniqueID。SO-101 与 UVC 相机的示例无需 Orbbec SDK 或 `sudo`。首次打开 UVC 相机时会用 Swift 编译助手程序，并可能需要在 macOS 中授予相机权限。用普通用户运行 `macbot --config my.yaml collect`。

## 本地 MuJoCo 仿真

```bash
pip install -e '.[sim]'                  # 在源码目录执行
macbot --config configs/so101_sim.yaml collect
```

示例使用 Python 3.12、MuJoCo 3.3.7、离屏渲染和脚本主臂，已在 Apple 芯片 Mac 上验证。首次运行会下载固定版本的 SO-101 EnvHub 代码、场景和模型资源。示例显式开启 `trust_remote_code`，请只运行你已审阅并信任的环境代码。缓存完整后可离线运行，无需相机、串口、CAN 适配器或 `sudo`。

`robot.env` 使用本地 LeRobot 环境配置，要求绝对关节位置指令以及同名的 `.pos` 状态。`teleop.hz` 和 `run.hz` 应与环境控制频率一致（示例为 30 Hz）。`type: robot` 相机的 `key` 指向渲染图像，默认取相机名；`save_size` 设置输出大小，`crop: [x0, y0, x1, y1]` 可在缩放前裁剪。策略和回放命令见[使用指南](guide_zh.md#物理仿真)。
