# 机械臂支持

[English](robots.md)

| `robot.type` | 机械臂 | 在 macbot 中的验证情况 |
|---|---|---|
| `rebot_b601` | Seeed reBot B601-RS 从臂与 Arm 102 主臂 | 已在 Apple 芯片 Mac 上验证真机。 |
| `lerobot` | SO-100/101、Koch、OMX、OpenArm、YAM，以及其他兼容的 LeRobot 机械臂 | 通过 LeRobot 配置。仅 fake 模式：SO-101 已验证采集和导出；其他型号有内置关节布局或可提供 `robot.joints`。未验证真机。 |
| `sim` | MuJoCo 中的 SO-101 桌面场景 | 采集、导出、ACT 推理和回放已在仿真中跑通（随机权重的 ACT，未测任务成功率）。采集用脚本主臂，不是遥操作。 |

运行 `macbot robots` 可列出已注册的 LeRobot 类型。列表取决于已安装的包，不代表验证结果。
通用从臂必须接受关节位置（`.pos`）动作，并能观测这些关节。用于采集的主臂必须有相同的关节名。
策略检查点必须和关节顺序、单位及相机输入匹配。

## reBot B601-RS

从 [`configs/example.yaml`](../configs/example.yaml) 和[安装指南](setup_zh.md)开始。
参考配置通过 PCBUSB 使用 PCAN-USB，并自动检测 Arm 102 的串口。重力前馈另需 URDF。

```yaml
robot:
  type: rebot_b601
  can_channel: can0
  follower_id: follower1
  leader_id: rebot_arm_102_leader
```

## 通用 LeRobot 机械臂

[`configs/so101.yaml`](../configs/so101.yaml) 是使用一台 UVC 相机的完整 SO-101 示例。
安装对应的 LeRobot 扩展，例如 SO-100/101 使用 `lerobot[feetech]==0.7.0`，
Koch 和 OMX 使用 `lerobot[dynamixel]==0.7.0`。换用其他机械臂时，替换主从臂类型并
填写 LeRobot 配置所需的字段。相机放在顶层 `cameras`；只跑策略且不用遥操作时可省略 `leader`。

```yaml
robot:
  type: lerobot
  follower: {type: so101_follower, port: /dev/tty.usbmodem_FOLLOWER, id: follower}
  leader: {type: so101_leader, port: /dev/tty.usbmodem_LEADER, id: leader}
  limits: {gripper: [0, 100]}
```

Koch 使用 `koch_follower` 和 `koch_leader`，OMX 使用 `omx_follower` 和 `omx_leader`。
其他型号请查阅已安装的 LeRobot 配置；若 fake 模式没有内置布局，请提供按顺序排列的 `robot.joints`。
限位与二值夹爪阈值使用驱动定义的单位。采集和运行时沿用相同的 ID 和标定目录。

## 本地仿真

安装 `macbot[sim]`（源码目录可运行 `pip install -e ".[sim]"`），使用完整的
[`configs/so101_sim.yaml`](../configs/so101_sim.yaml)。运行 `macbot --config configs/so101_sim.yaml collect`，
不加 `--fake`。仿真使用脚本主臂，各频率须与场景匹配。固定版本的 EnvHub 场景会在首次使用时下载并执行 Python 代码。

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

`macbot init` 检查标定文件并提供对应命令。请单独运行 `lerobot-calibrate`，使用相同的设备类型、串口、ID
及可选的 `calibration_dir`。从臂使用 `--robot.*`，主臂使用 `--teleop.*`，命令见
[安装指南](setup_zh.md)。macbot 以 `calibrate=False` 连接。仿真和 fake 模式不需要标定。

运行 `macbot --config my.yaml doctor` 检查所选配置。对于通用机械臂，
它检查驱动导入、配置中的串口路径和标定文件是否存在，不打开设备。
对于内置模拟器，它检查本地配置和 MuJoCo 导入，不加载场景。检查通过不代表标定质量或真机运动已经验证。

## 新增机械臂类

如果现有上游驱动符合要求，优先使用 `robot.type: lerobot`。否则实现
[`Robot` 接口](../macbot/robot/base.py)，在 [`open_robot`](../macbot/robot/__init__.py)
里注册配置。定义有序的 `joints`、可选的 `gripper` 和 `limits`，以及 `info()` 元信息。
实现设备的连接、观测、命令和退出行为，保留标定保护，并说明退出时的行为。
添加对应的 fake 布局、配置示例和无硬件测试。
