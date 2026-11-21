# 把 LeRobot 环境从 Linux 搬到 Mac

[English](migrate.md)

保留原有 LeRobot 校准文件、v3.0 数据集和完整策略 checkpoint。在 Mac 上采集、检查、推理和回放，
训练仍放在 GPU 机器上。先按[快速开始](quickstart_zh.md)装好 Python 和相机依赖。
如果已安装的版本在 `macbot init --help` 里没有 `--from-lerobot`，在源码目录运行
`python -m pip install -e .`。

## 迁移校准文件

从 Linux 复制**同一对实体机械臂**的校准 JSON。两边的默认目录都是
`~/.cache/huggingface/lerobot/calibration/`。在这个目录下，SO-100 和 SO-101
共用驱动名，文件分别是 `robots/so_follower/<id>.json` 和 `teleoperators/so_leader/<id>.json`。
其他驱动在这两组目录下使用自己的驱动名。LeRobot 读取 `HF_HOME`、`HF_LEROBOT_HOME`
和 `HF_LEROBOT_CALIBRATION`；显式指定 `--robot.calibration_dir` 或
`--teleop.calibration_dir` 时，对应设备优先使用指定目录。自定义目录若已搬家，先在命令里改为 Mac 上的路径。

保留 Linux 上的设备 ID。换端口不需要重新校准。向导显示找到的文件，只读取校准。
如果原有文件没找到，先核对 ID、目录和读取权限，再决定是否需要单独运行向导提示的 `lerobot-calibrate`。

## 导入原命令

把整条 `lerobot-record` 或 `lerobot-teleoperate` 命令放进一个带引号的参数，也可以只贴参数。例如：

```bash
macbot init --from-lerobot 'lerobot-record
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
  --dataset.fps=30 --dataset.single_task="place the cube in the box"'
```

- **机械臂：**支持 SO-100/101、Koch 和 OMX 串口配对。保留两臂类型、ID、自定义校准目录，以及命令中
  提供的 `use_degrees` 和 `max_relative_target`。Mac 串口常见路径是 `/dev/cu.usbmodem*`
  或 `/dev/cu.usbserial*`。能匹配的端口会预选，不同的端口从 USB 列表里选对应的实体臂。
- **相机：**OpenCV 相机名成为 macbot 的相机角色，保留显式指定的宽、高和帧率。Linux 的 `/dev/video*`
  路径和数字索引在 macOS 上没有可靠对应关系，请对照预览分配角色。保存时使用 AVFoundation unique ID。
  确认相机支持原采集模式，画面与训练数据一致。相机列表排除 iPhone 和连续互通相机。
- **采集：**`--dataset.fps` 设置录制帧率，`--dataset.single_task` 填入任务指令，
  `lerobot-teleoperate --fps` 设置遥操作控制频率。
- **其他选项：**导入器会列出未映射的选项，例如 `--dataset.repo_id`、`--resume`、相机旋转和 FOURCC。
  其他驱动或相机类型请按[机械臂支持](robots_zh.md)和[相机设置](setup_zh.md#4-相机)手动写 YAML。
  只有端口、相机 unique ID 都已匹配才使用 `--yes`，否则用向导选设备。

保存后，`doctor`、`collect` 和 `run` 使用同一份配置。默认路径是 `~/.config/macbot/config.yaml`，
也可以在子命令前加 `--config profile.yaml`。SO-101 配 UVC 相机用普通用户运行，在 macOS 给终端
授权相机权限。reBot/Orbbec 组合的 SDK 和 `sudo -E` 用法见[安装指南](setup_zh.md)：
macbot 会为参考 CAN 配置预载 PCBUSB，并把以 sudo 创建的文件交回调用用户。
如果需要在 sudo 下推理，可以先用普通用户下载 Hub checkpoint，再让 `run --policy` 指向下载好的本地快照。

## 双向迁移数据与 checkpoint

```bash
# 在 Mac 上只读打开已有的 lerobot-record 数据集。
macbot health --dataset ./recordings
macbot replay --dataset owner/recordings

# 在 Mac 上采集、检查，再导出为标准 LeRobot v3.0。
macbot collect
macbot health
macbot export --out ./new-dataset

# 在 GPU 机器训练后，把完整 checkpoint 搬回来。
macbot run --policy ./pretrained_model --fake --device cpu --autostart 3
macbot replay
```

`health` 统计缺失的记录帧序号和过长的时间间隔。LeRobot 可能按固定帧率生成时间戳，因此这项检查
无法测到写入前相机丢掉的帧。`replay` 显示 episode 时长、记录的状态和动作，以及各路相机画面。
两条命令都支持本地 v3.0 Parquet 数据集（MP4 或内嵌 RGB 图片），也可使用 Hub ID，
格式为 `owner/name` 或 `owner/name@revision`。回放按需下载帧数据和视频分片，
不写评审结果，也不转换数据。旧版数据请先用 LeRobot 转换。

把整个导出目录移到 GPU 机器，用 `lerobot-train` 训练。将生成的 `pretrained_model` 目录复制回来，
或将其内容发布到 Hub，通过 `run --policy owner/model@revision` 或 `bench --policy owner/model@revision`
加载。命令会显示实际使用的 Hub 提交，试验元数据也会记录它。目录应包含权重、两套 processor JSON 和
它们引用的状态文件。保持相机名、分辨率、关节顺序和动作单位与权重匹配，具体见[策略支持](policies_zh.md)。

| 路径 | 证据 |
|---|---|
| reBot 参考配置 | 此前真机测试 |
| 导入命令、重选设备 | 模拟设备测试 |
| 数据集检查与回放 | LeRobot 0.7.0 本地输出、模拟 Hub |
| SO-101 采集与导出 | 此前 fake 和仿真测试 |
| 通用机械臂在 Mac 上运动 | 未验证 |

验证范围见[机械臂支持](robots_zh.md)和[策略支持](policies_zh.md)。本次迁移流程没有驱动真机。
