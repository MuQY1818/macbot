# 环境配置

[English](setup.md)

[快速开始](quickstart.zh.md)里的向导能为大多数设备写好配置。本页适用于手动编写或修改配置、配置 reBot 参考设备，以及排查问题。

## Python 环境

```bash
python3.12 -m venv ~/.venvs/macbot && source ~/.venvs/macbot/bin/activate
pip install macbot
xcode-select --install       # 提供 swiftc，首次使用 UVC 相机时编译辅助程序
```

任何 Python 3.12 环境都可以，包括 conda。可选的附加依赖：

| 附加项 | 增加的内容 |
|---|---|
| `macbot[pi]` | LeRobot 的 π0 和 π0.5 依赖 |
| `macbot[mlx]` | 上面的依赖加上 MLX，用于[加速后端](acceleration.zh.md) |
| `macbot[orbbec]` | Orbbec SDK（版本问题见[特殊硬件](quickstart.zh.md#特殊硬件)） |
| `macbot[sim]` | MuJoCo，用于[本地仿真](#本地-mujoco-仿真) |

## LeRobot 机械臂与 UVC 相机

适用于 SO-100/101、Koch、OMX 以及其他有 LeRobot 0.7.0 驱动的机械臂。按机械臂安装对应的驱动依赖，以 SO-100/101 为例：

```bash
pip install macbot 'lerobot[feetech,core-scripts]==0.7.0'
macbot robots            # 已注册的从臂和主臂类型
lerobot-find-port        # 两只机械臂的串口
macbot cameras           # 相机及其唯一 ID
```

1. 以 [configs/so101.yaml](../configs/so101.yaml) 为起点，设置 `robot.type: lerobot`。
2. 从 `macbot robots` 中选择 `follower.type` 和 `leader.type`，并填写各设备的 LeRobot 参数。列表显示的是已注册的类型，驱动和系统支持情况取决于具体设备。
3. 用 `lerobot-calibrate` 标定，设备类型、端口和 ID 与配置一致。macbot 使用保存下来的标定。
4. 主臂和从臂使用相同的关节单位。其他机器人请按它自己的单位设置 `robot.limits`。
5. 把 `cameras` 部分换成你的 UVC 相机和它们的唯一 ID（见[相机](#相机)）。

以普通用户运行 `macbot --config my.yaml collect`，不需要 Orbbec SDK，也不需要 `sudo`。macbot 首次使用时会用 Swift 编译 UVC 辅助程序，macOS 会请求相机权限。启动时两只机械臂要对齐。

## 相机

```bash
macbot cameras
```

会列出所有视频设备及其 AVFoundation 唯一 ID。把每台相机的 ID 写进配置：

```yaml
cameras:
  top:
    type: uvc
    unique_id: "0x1100001234abcd"
    crop: [625, 140, 1610, 1080]    # 在完整画面上的 x0 y0 x1 y1；[0, 0, 0, 0] 表示不裁剪
    save_size: [492, 470]
```

!!! warning "用唯一 ID，不要用名称或序号"

    附近有 iPhone 时，macOS 可能给你连续互通相机，而不是你想要的那台。

USB 相机的唯一 ID 包含接口信息。相机换到其他接口后，macbot 会按厂商和产品型号重新匹配。

## reBot B601-RS 参考配置

参考配置是 PCAN-USB 适配器上的 reBot B601-RS 从臂、reBot 102 主臂，加一台 Orbbec 腕部相机。

### CAN 驱动（PCAN-USB）

从 [mac-can.com](https://www.mac-can.com) 下载 macOS 版 **PCBUSB** 库并运行安装程序，它会把 `libPCBUSB.dylib` 装到 `/usr/local/lib`。插上 PCAN-USB 适配器，给从臂上电，用 `macbot doctor` 检查。

### 标定

macbot 从不重新校零。它连接时一律使用 `calibrate=False`，没有标定文件就拒绝连接。请先用 LeRobot 标定一次，ID 与配置一致（`robot.follower_id` 默认 `follower1`，`robot.leader_id` 默认 `rebot_arm_102_leader`）：

```bash
lerobot-calibrate --robot.type=rebot_b601_follower --robot.id=follower1 --robot.port=can0 \
  --robot.motor_family=rs --robot.can_adapter=socketcan
lerobot-calibrate --teleop.type=rebot_102_leader --teleop.id=rebot_arm_102_leader --teleop.port=/dev/cu.usbserial-XXXX
```

标定文件保存在 `~/.cache/huggingface/lerobot/calibration/`，请留一份备份。

### Orbbec 相机与 sudo

按[特殊硬件](quickstart.zh.md#特殊硬件)安装 SDK。Orbbec SDK 以 root 身份打开 USB 设备，所以使用 Orbbec 相机的命令要用 `sudo -E` 运行，`-E` 保留当前的 Python 环境。macbot 会把 sudo 下写出的文件交还给你的用户。

### 重力前馈（可选）

`robot.gravity_compensation` 会在每条 MIT 指令中加入重力力矩，明显减少肩部和肘部的下垂。它需要 Seeed reBot 描述包里的 URDF，macbot 不附带这个文件。把 `robot.urdf` 设为它的路径。没有 URDF 时，macbot 只使用速度前馈。

### 第一次运行

```bash
macbot collect --fake                                  # 不接硬件先看控制台
sudo -E $(which python) -m macbot.cli joints           # 读取两只机械臂，不会运动
sudo -E $(which python) -m macbot.cli collect          # 用真实机械臂录制
```

运行 `collect` 之前，把两只机械臂都放在零位附近。主从臂相差超过 `teleop.align_max_deg` 时，macbot 不会启动遥操作。

想从访达启动，双击 `scripts/collect.command` 或 `scripts/run.command`。

## 本地 MuJoCo 仿真

```bash
pip install -e '.[sim]'                  # 在源码目录中运行
macbot --config configs/so101_sim.yaml collect
```

不需要相机、串口设备、CAN 适配器或 `sudo`。示例使用 Apple 芯片上的 Python 3.12、MuJoCo 3.3.7、离屏渲染和脚本主臂。

!!! warning "远程代码"

    首次启动会下载固定版本的 SO-101 EnvHub 代码、场景和模型资源。示例显式开启了 `trust_remote_code`，请只使用你审查过并信任的 EnvHub 代码。缓存好的快照可以离线运行。

`robot.env` 接受本地 LeRobot 环境配置，要求动作是绝对关节位置，状态名为对应的 `.pos`。`teleop.hz` 和 `run.hz` 应与环境的控制频率一致（示例为 30）。`type: robot` 相机的 `key` 选择一路渲染图像，默认取相机名。`save_size` 设置输出尺寸，可选的 `crop: [x0, y0, x1, y1]` 在缩放之前裁剪。

策略和回放命令见[工作流程](guide.zh.md#没有硬件时)，键盘控制和评测见[仿真](simulation.zh.md)。
