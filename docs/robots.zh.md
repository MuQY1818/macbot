# 机械臂

[English](robots.md)

| `robot.type` | 机械臂 | 状态 |
|---|---|---|
| `rebot_b601` | Seeed reBot B601-RS 从臂与 Arm 102 主臂 | 已在 Apple 芯片 Mac 上实机测试 |
| `lerobot` | SO-100/101、Koch、OMX、OpenArm、YAM，以及其他兼容的 LeRobot 机械臂 | 仅在假硬件模式下验证，尚未实机验证 |
| `sim` | MuJoCo 中的 SO-101 桌面场景 | 整个工作流程端到端跑通 |

- **`lerobot`** 机械臂通过 LeRobot 配置。假硬件模式下已验证 SO-101 的采集和导出；其他型号有内置的关节布局，或者可以自己提供 `robot.joints`。
- **`sim`** 已跑通键盘遥操作、脚本示范、导出、ACT 训练和按种子评测。合成的键盘输入完成了任务，真人手动放入了一个方块。训练出的策略能否完成任务，目前还没有验证。

!!! note "`macbot robots` 不是已测试机械臂的列表"

    `macbot robots` 列出的是已安装的包注册的 LeRobot 类型，不代表这些机械臂已经验证过。

## 所有机械臂的共同要求

- 从臂接受关节位置（`.pos`）动作，并报告同样的关节。
- 采集用的主臂与从臂的关节名相同。
- 策略权重与所运行机械臂的关节顺序、单位和相机输入一致。

## 通用 LeRobot 机械臂

[configs/so101.yaml](../configs/so101.yaml) 是一份完整的 SO-101 示例，带一台 UVC 相机。按机械臂安装对应的 LeRobot 依赖：SO-100/101 用 `lerobot[feetech]==0.7.0`，Koch 和 OMX 用 `lerobot[dynamixel]==0.7.0`。

```yaml
robot:
  type: lerobot
  follower: {type: so101_follower, port: /dev/tty.usbmodem_FOLLOWER, id: follower}
  leader: {type: so101_leader, port: /dev/tty.usbmodem_LEADER, id: leader}
  limits: {gripper: [0, 100]}
```

| 机械臂 | `follower.type` | `leader.type` |
|---|---|---|
| SO-100/101 | `so101_follower` | `so101_leader` |
| Koch | `koch_follower` | `koch_leader` |
| OMX | `omx_follower` | `omx_leader` |

其他机械臂请替换两个类型，并填写它们的 LeRobot 配置要求的字段。如果假硬件模式没有这款机械臂的内置布局，请在 `robot.joints` 下按顺序列出关节名。限位和二值夹爪阈值使用驱动自己的单位。相机写在顶层的 `cameras` 下；只运行策略时可以不写 `leader`。录制和运行时请使用相同的 ID 和标定目录。

## reBot B601-RS

从 [configs/example.yaml](../configs/example.yaml) 和[环境配置](setup.zh.md#rebot-b601-rs-参考配置)里的 reBot 部分开始。参考配置通过 PCAN-USB 适配器和 PCBUSB 驱动连接从臂，并自动找到 Arm 102 主臂的串口。重力前馈还需要机械臂的 URDF。

```yaml
robot:
  type: rebot_b601
  can_channel: can0
  follower_id: follower1
  leader_id: rebot_arm_102_leader
```

## 本地仿真

安装、键盘遥操作、脚本示范和评测见[仿真](simulation.zh.md)。完整配置是 [configs/so101_sim.yaml](../configs/so101_sim.yaml)，它用正弦波驱动机械臂，用于快速检查流程。采集时不要加 `--fake`，控制频率要与场景一致。固定版本的 EnvHub 场景首次使用时会下载并运行 Python 代码。

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

## 标定与检查

macbot 连接时一律使用 `calibrate=False`，所以机械臂要先用 LeRobot 标定。`macbot init` 会查找标定文件，并给出对应的 `lerobot-calibrate` 命令。请单独运行这些命令，设备类型、串口、ID 和可选的 `calibration_dir` 与配置一致；从臂用 `--robot.*`，主臂用 `--teleop.*`。没有实体主臂的仿真和假硬件模式不需要标定。

`macbot --config my.yaml doctor` 检查所选配置，不会移动任何东西：

- 通用机械臂：检查驱动能否导入、串口路径是否存在、标定文件是否齐全，不打开设备。
- 仿真：检查配置和 MuJoCo 能否导入，不加载场景。

检查通过不代表标定准确，也不代表机械臂运动正常。

## 新增机械臂类

如果 LeRobot 已有合适的驱动，直接用 `robot.type: lerobot`。否则：

1. 实现 [`Robot` 接口](../macbot/robot/base.py)：有序的 `joints`、可选的 `gripper` 和 `limits`、`info()` 元信息，以及设备的连接、观测、指令和退出行为。
2. 在 [`open_robot`](../macbot/robot/__init__.py) 里注册这个配置。
3. 保留标定保护，并写明退出时会发生什么。
4. 添加对应的假硬件布局、示例配置，以及不需要硬件的测试。
