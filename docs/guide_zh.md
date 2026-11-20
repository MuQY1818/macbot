# 使用指南：采集、导出、训练、部署

[English](guide.md)

下文默认 `data_root: ~/macbot-data`，任务名为 `cubes_to_box`。

```
~/macbot-data/
  raw/cubes_to_box/episode_000.npz ...      采集的数据
  datasets/cubes_to_box_bin/                导出的 LeRobot 数据集（含 gripper.json）
  rollouts/cubes_to_box/rollout_000.npz ... 策略试验记录
  logs/                                     每次运行一个日志文件
```

## 1. 采集

```bash
sudo -E $(which python) -m macbot.cli --config my.yaml collect
```

网页地址是 <http://127.0.0.1:8766>。

| 按键 | 作用 |
|---|---|
| **S** | 开始录制 |
| **回车** 或 **→** | 保存这一条 |
| **←** | 丢弃这一条 |

待机时可以修改指令框，每条数据单独记录自己的指令，这条指令就是 π0.5 的语言提示。点任意一行的“回放”，可以拖动查看各路画面和关节曲线。“删除”会把文件移到 `deleted/`，编号不会复用。

保存时会自动检查。时长不足 `record.min_episode_s` 的不保存。以下情况记为警告：

- 画面延迟 p95 超过 `checks.max_cam_age_s`
- 重复帧比例超过 `checks.max_repeat_ratio`
- 某个关节一帧内跳变超过 `checks.max_joint_step_deg`
- 迟到采样比例超过 `checks.max_late_ratio`

点“结束并回零”或按 Ctrl-C，从臂会分两段慢速回零，然后失能。

### 数据格式

| 键 | 形状 | 内容 |
|---|---|---|
| `qpos` | (N, 7) | 从臂实测关节角，单位度 |
| `action` | (N, 7) | 发给从臂的指令（夹爪取主臂读数） |
| `timestamp` | (N,) | k / fps，固定采样网格 |
| `wall_time` | (N,) | 实际采样时刻 |
| `<cam>_jpeg`、`<cam>_offsets` | | JPEG 流，第 i 帧是 `jpeg[off[i]:off[i+1]]` |
| `<cam>_age`、`<cam>_repeat` | (N,) | 采样时画面已过去多久；是否和上一帧重复 |
| `meta` | JSON | 任务、指令、fps、相机、从臂前馈设置、检查结果 |

## 2. 检查数据

```bash
macbot health                  # 逐条报告
macbot health --split 27       # 比较第 27 条之前和之后的中位数，比如维修前后
```

每条数据会列出：

- 让指令和实测对得最齐的滞后，以及这个滞后下的 p95 跟踪误差
- 主臂静止时的偏差，标定漂移会在这里显出来
- 夹爪范围
- 最长的帧间隔和重复帧数量

## 3. 导出

```bash
macbot export                              # 夹爪动作 = 主臂角度
macbot export --gripper binary             # 夹爪动作 = 0（张开）/ 1（闭合），并写出 gripper.json
```

做抓取放置任务时，二值夹爪比原始角度效果好。

- **标签：** 主臂读数高于 `--close-above`（-40°）时开始记为闭合，直到低于 `--open-below`（-80°）才记为张开。
- **部署：** 张开发 `--open-deg`（-150°），闭合发 `--close-deg`（0°）。夹紧力由夹爪限力控制。

检查不通过的数据会自动跳过。加 `--skip-warnings` 时，有警告的也跳过。

## 4. 训练

训练放在带 NVIDIA GPU 的 Linux 机器上。先把数据集文件夹拷过去，然后：

```bash
DATASET=~/data/cubes_to_box_bin GPUS=0,1,2,3 ./train/train_pi05.sh cubes_pi05
DATASET=~/data/cubes_to_box_bin ./train/train_act.sh cubes_act
```

- `train_pi05.sh` 通过 `train/train_augmented.py` 微调 `lerobot/pi05_base`。这个脚本就是 `lerobot-train`，外加在 GPU 上做的 openpi 风格图像增强：俯视相机做裁剪和旋转，所有相机做颜色抖动。
- 设 `WRIST_SHIFT=0.03` 可以给腕部相机加轻微平移。腕部支架在两次采集之间挪过位置时有用。

把 `checkpoints/<步数>/pretrained_model` 拷回 Mac。如果是二值夹爪数据集，把 `gripper.json` 放在它旁边（或上一两级目录）。

## 5. 跑策略

```bash
sudo -E $(which python) -m macbot.cli run --policy ~/policies/cubes_pi05/004000 --speed 0.3
```

网页地址是 <http://127.0.0.1:8767>，启动后处于**暂停**状态。

| 按键 | 作用 |
|---|---|
| **S** | 开始 |
| **空格** | 暂停（保持当前指令） |
| **H** | 慢速回到起始姿态 |
| **Y** / **N** | 把这次试验存为成功 / 失败 |

每次试验存到 `rollouts/<任务>/`，包含所有画面、指令和实测关节、每个块的开始时刻和推理耗时。

### 执行方式（`--mode`）

| 模式 | 做法 | 默认用于 |
|---|---|---|
| `chunk` | 推理一次，执行前 `--k` 步，再推理；推理期间机械臂等待 | |
| `ensemble` | 持续推理，每个时刻的目标是覆盖当前时刻的所有动作块按 exp(m · age) 加权平均（ACT 的时间集成） | ACT |
| `async` | 当前块继续执行的同时在后台推理，新块与旧块在时间上对齐 | |
| `rtc` | 在 `async` 基础上加 Real-Time Chunking：用旧块里还没执行的部分引导新块的去噪，前后两块衔接平滑 | π0.5 |

所有执行方式都会在策略步之间插值，供 60 Hz 控制环使用。

- **速度：** `--speed` 按比例缩放时间。0.5 倍速时，30 fps 的策略按每秒 15 步执行。
- **安全限制：** 臂关节速度不超过 `run.max_vel × speed` 度每秒，并始终在关节限位之内。
- **断流暂停：** 任一相机断流 0.5 秒就自动暂停。

### 加入自己的执行方式

执行器在 `macbot/policy/executors.py` 里，接口只有四个方法：

```python
class MyExecutor(Executor):
    def start(self): ...                  # 如果有线程，在这里启动
    def reset(self, active: bool): ...    # 清空动作块；active=True 时从当前观测重新规划
    def target(self, now: float): ...     # 返回 (关节目标 (7,), 块编号)，或 (None, 编号)
    def describe(self) -> str: ...
```

`AsyncChunk.accept()` 是一个钩子，决定是否切到新推理出的块、从哪一步开始切。继承它就能试验新的切换规则，不用改控制环。

## 6. 回看

```bash
macbot replay                                   # http://127.0.0.1:8768
macbot video ~/macbot-data/rollouts/cubes_to_box/rollout_003.npz --height 720
```

`macbot bench --policy DIR` 用空白图像测这台 Mac 上的推理耗时。

## 模拟硬件

`collect`、`run` 和所有网页都支持 `--fake`：机械臂用一阶滞后模型模拟，相机用合成画面代替。`run --fake` 时策略是真的在推理，所以能在碰真机之前完整检查加载、编译和执行器。`--autostart 秒数` 会自动运行一段时间、保存一次试验，然后退出。
