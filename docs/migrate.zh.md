# 从 Linux 迁移

[English](migrate.md)

如果你已经在 Linux 上使用 LeRobot，标定文件、LeRobot v3.0 数据集和完整的策略权重都可以继续用。在 Mac 上采集、检查、推理和回放，训练仍放在 GPU 机器上。

| 从 Linux 沿用 | 在 Mac 上重新设置 |
|---|---|
| 同一对实体机械臂的标定文件 | 串口路径 |
| `lerobot-record` 或 `lerobot-teleoperate` 命令 | 相机分配（Linux 的 `/dev/video*` 在 macOS 上没有对应） |
| 数据集和权重 | 无需处理，直接读取 |

先按[快速开始](quickstart.zh.md#1-安装)第 1 步装好 Python 环境。

## 迁移标定文件

把**同一对实体机械臂**的标定 JSON 从 Linux 拷到 Mac 上的相同位置：

```
~/.cache/huggingface/lerobot/calibration/
  robots/so_follower/<id>.json           SO-100 和 SO-101 从臂
  teleoperators/so_leader/<id>.json      SO-100 和 SO-101 主臂
```

其他驱动在这两个目录下使用各自的驱动名。保留 Linux 上的设备 ID。换了端口不需要重新标定。

??? note "自定义标定目录"

    LeRobot 会读取 `HF_HOME`、`HF_LEROBOT_HOME` 和 `HF_LEROBOT_CALIBRATION`。显式指定 `--robot.calibration_dir` 或 `--teleop.calibration_dir` 时，对应设备优先使用指定的目录。如果自定义目录换了位置，导入之前先在命令里改成 Mac 上的路径。

向导会显示找到了哪些文件，并且只读取它们。如果应该有的标定文件没找到，先核对 ID、目录和读取权限，再运行向导给出的 `lerobot-calibrate` 命令。

## 导入原命令

把整条 `lerobot-record` 或 `lerobot-teleoperate` 命令放进一个带引号的参数，也可以只贴参数：

```bash
macbot init --from-lerobot 'lerobot-record
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
  --dataset.fps=30 --dataset.single_task="place the cube in the box"'
```

向导打开时会填好导入的值。

- **机械臂。** 支持 SO-100/101、Koch 和 OMX 串口主从臂。导入时保留两侧的类型、ID、可选的标定目录，以及给出的关节设置 `use_degrees` 和 `max_relative_target`。macOS 的串口路径通常形如 `/dev/cu.usbmodem*` 或 `/dev/cu.usbserial*`。匹配的端口会被预先选中；不一致时，在 USB 列表中选择对应的实体机械臂。
- **相机。** OpenCV 相机名会成为 macbot 的相机角色，显式给出的宽、高和 fps 会保留。请根据预览匹配每个角色，因为 Linux 的 `/dev/video*` 路径和数字序号在 macOS 上没有可靠的对应关系。保存的配置使用 AVFoundation 唯一 ID。请确认 Mac 上的相机支持所需的画面模式，并且视角与训练数据一致。选择列表会跳过 iPhone 和连续互通相机。
- **录制。** `--dataset.fps` 设置录制频率，`--dataset.single_task` 填入语言指令。`lerobot-teleoperate --fps` 设置遥操作频率。
- **其他参数。** 没有映射的参数会在终端列出，例如 `--dataset.repo_id`、`--resume`，以及旋转、FOURCC 等相机选项。其他驱动或相机类型请参照[机械臂](robots.zh.md)和[环境配置](setup.zh.md#相机)手动编写 YAML。

只有端口路径和相机唯一 ID 都已经对得上时才传 `--yes`，否则请在向导里选择。

保存后，`doctor`、`collect` 和 `run` 都使用同一份配置。默认配置是 `~/.config/macbot/config.yaml`；要用其他文件，把 `--config profile.yaml` 放在子命令之前。使用 SO-101 和 UVC 相机时，以普通用户运行，并在 macOS 中允许终端使用相机。

??? note "sudo 下的 reBot 和 Orbbec"

    SDK 和 `sudo -E` 命令见[环境配置](setup.zh.md#rebot-b601-rs-参考配置)。参考 CAN 配置下 macbot 会预加载 PCBUSB，并把 sudo 下创建的文件交还给调用它的用户。需要在 sudo 下推理时，可以先以普通用户下载 Hub 上的权重，再把本地快照目录传给 `run --policy`。

## 双向迁移数据和权重

```bash
# 在 Mac 上读取已有的 lerobot-record 数据集，不做任何修改。
macbot health --dataset ./recordings
macbot replay --dataset owner/recordings

# 在 Mac 上录制新数据并检查，然后导出标准的 LeRobot v3.0。
macbot collect
macbot health
macbot export --out ./new-dataset

# 在 GPU 机器上训练后，把完整的权重拿回来。
macbot run --policy ./pretrained_model --fake --device cpu --autostart 3
macbot replay
```

### 读取已有数据集

`health` 和 `replay` 接受本地 v3.0 Parquet 数据集（MP4 或内嵌 RGB 图像）和 Hub 数据集 ID，可以用 `owner/name@revision` 固定版本。回放会按需下载帧和视频分片。两者都不会写入评审，也不会转换数据集。更早格式的数据集请先用 LeRobot 转换。

- `health` 统计缺失的帧序号和偏大的时间戳间隔。LeRobot 可能按固定网格写时间戳，所以这些检查看不到写帧之前发生的相机丢帧。
- `replay` 显示片段时长、录制的状态和动作，以及每一路相机画面。

### 训练并拿回权重

把导出的整个数据集拷到 GPU 机器上，用 `lerobot-train` 训练。之后可以把生成的 `pretrained_model` 目录拷回来，也可以把其中的内容发布到 Hub，再用 `run --policy owner/model@revision` 或 `bench --policy owner/model@revision`。Hub 解析器会打印实际下载的提交，并记录到保存的试验中。

!!! warning "权重要带齐所有文件，机器人配置要一致"

    请带上权重、两个处理器 JSON 文件以及它们引用的状态文件。相机名、分辨率、关节顺序和动作单位要与训练时一致。目录结构和模型要求见[策略](policies.zh.md)。

??? note "已经验证过的部分"

    | 环节 | 依据 |
    |---|---|
    | 参考 reBot 配置 | 之前的硬件测试 |
    | 命令导入与重映射 | 模拟设备测试 |
    | 数据集检查与回放 | LeRobot 0.7.0 本地输出；模拟的 Hub |
    | Hub 权重 | 公开 ACT 权重已下载、测速，并在假硬件机械臂上运行（CPU） |
    | SO-101 采集与导出 | 之前的假硬件和仿真运行 |
    | 通用机械臂在 Mac 上的运动 | 未验证 |

    这套迁移流程是在不移动硬件的情况下验证的。这些检查的范围见[机械臂](robots.zh.md)和[策略](policies.zh.md)。
