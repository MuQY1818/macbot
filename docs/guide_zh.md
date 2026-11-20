# 使用指南：采集、导出、训练、部署

[English](guide.md)

下文默认 `data_root: ~/macbot-data`，任务名为 `cubes_to_box`。
带 `sudo` 的命令用于 Orbbec 参考配置。通用机械臂只使用 UVC 相机时，可用普通用户运行 `macbot --config my.yaml <command>`，见[安装指南](setup_zh.md#lerobot-机械臂与-uvc-相机)。

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

打开终端显示的地址（8766 可用时使用该端口）。

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

点“结束并回零”或按 Ctrl-C，参考 reBot 从臂会分两段慢速回零。通用 LeRobot 从臂会回到连接时的姿态，然后断开。

### 数据格式

| 键 | 形状 | 内容 |
|---|---|---|
| `qpos` | (N, J) | 从臂实测关节位置，采用设备单位；J 为关节数 |
| `action` | (N, J) | 发给从臂的关节指令（参考 reBot 的夹爪取限制前的主臂读数） |
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

二值标签要等主臂超过 `--close-above` 才变成「闭合」，这时手指已经合了大半。用它训练的策略也会这么晚才发出闭合，
而真实夹爪还要走完整个行程（reBot 软夹爪约 0.5 s）。`run --grip-lead 0.35` 在当前动作块里策略计划 0.35 s
后闭合时就提前闭合，张开不受影响。每次试验在 `gp` 旁边另存这个提前值 `gp_ahead`。

## 5. 跑策略

```bash
sudo -E $(which python) -m macbot.cli run --policy ~/policies/cubes_pi05/004000 --speed 0.3
```

打开终端显示的地址（8767 可用时使用该端口），启动后处于**暂停**状态。

| 按键 | 作用 |
|---|---|
| **S** | 开始 |
| **空格** | 暂停（保持当前指令） |
| **H** | 慢速回到起始姿态 |
| **Y** / **N** | 把这次试验存为成功 / 失败 |
| **T** | 手动接管 / 交还策略（需要启用 `--takeover`） |
| **M** | 标记手进入画面的时刻，控制方式不变 |

每次试验存到 `rollouts/<任务>/`，包含所有画面、指令和实测关节、每个块的开始时刻和推理耗时。

暂停后，点击 **已保存的试验** 中的任意一行，就能在运行页回放。列表按保存时间倒序显示最近
5 条，包含这个任务以往的试验；点击 **显示全部** 查看其余记录。拖动进度条或播放可同步查看
各路相机和关节曲线，进度下方显示当前推理块及其耗时。点击 **回到实时画面** 返回相机预览。
开始运行或回起始姿态会关闭回放；暂停后可以继续查看。

在回放面板选择 **结果**，填写可选的单行 **备注**；也可勾选 **排除此试验** 并填写原因，例如 GPU
资源争用。点击 **保存**。排除后仍可回放，但不计入成功率、健康汇总和导出。修改记录在原文件旁的
`rollout_XXX.review.json`，原始录制保持不变。复制已评审的试验时，请同时复制这两个文件。

### 手动接管

运行时加 `--takeover`，或在 YAML 中设置 `run.takeover: true`。程序会连接 `collect` 使用的已校准领臂。
策略运行时按 **T** 或点击 **手动接管**，从臂保持当前指令。按页面显示的关节角度差手动移动领臂，
所有臂关节大致到位，都在 `run.takeover_align_deg`（默认 30°）以内，而且夹爪也对齐后开始跟随：从臂先按
`run.home_vel` 沿直线慢速追上领臂（起步有短暂加速），所有关节差到 0.5° 以内后再按正常速度上限跟随。从这时起的帧才标为人工操作（1），追赶过程与对齐一样标为 2，
纠正数据从操作者自己的动作开始。使用二值夹爪策略时，
若从臂当前指令为闭合，领臂夹爪读数必须高于 `export.gripper_close_above`；若为张开，则必须低于
`export.gripper_open_below`。读数介于两条阈值之间时，从臂继续保持。使用原始角度时，领臂夹爪与从臂夹爪
指令的差值须在 `run.takeover_gripper_align_deg`（默认 15°）以内。对齐面板会提示还需调整哪些关节。

跟随时继续应用策略的臂关节速度限制和限位，夹爪指令每秒最多变化 `300 × speed` 度，reBot 夹爪继续使用
采集时的力限制。领臂读数失效时，从臂保持当前指令；读数恢复后需要重新对齐。

再按 **T**，策略从当前观测重新规划。暂停会取消手动控制，按 **S** 可恢复策略。回起始姿态、保存和退出
仍可用。`--fake` 和仿真使用脚本领臂，也要通过同一对齐检查。

开启接管后，试验增加 `takeover`（`0` 策略、`1` 跟随、`2` 等待对齐并保持）和 `leader`（逐帧、逐关节读数，
不可用时为 `NaN`）。元数据中的 `takeover_count` 记录请求次数，`takeover_seconds` 记录实际跟随秒数。
试验尚未保存或丢弃时，按 **M** 或点击 **标记手进入画面** 会给 `hand_marks` 增加一个时间戳，即使未开启
接管也可使用。回放在时间轴和曲线上标出对齐、跟随区间及手进入画面的时刻。

### 执行方式（`--mode`）

| 模式 | 做法 | 默认用于 |
|---|---|---|
| `chunk` | 推理一次，执行前 `--k` 步，再推理；推理期间机械臂等待 | |
| `ensemble` | 持续推理，每个时刻的目标是覆盖当前时刻的所有动作块按 exp(m · age) 加权平均（ACT 的时间集成） | ACT |
| `async` | 当前块继续执行的同时在后台推理，新块与旧块在时间上对齐 | |
| `rtc` | 在 `async` 基础上加 Real-Time Chunking：用旧块里还没执行的部分引导新块的去噪，前后两块衔接平滑 | π0.5 |

所有执行方式都会在策略步之间插值，供 60 Hz 控制环使用。

- **速度：** `--speed` 按比例缩放时间。0.5 倍速时，30 fps 的策略按每秒 15 步执行。
- **安全限制：** 臂关节速度不超过 `run.max_vel × speed` 度每秒，并始终在关节限位之内。**回起始姿态**按 `run.home_vel` 移动（默认最远的关节每秒 25 度，不受
  `speed` 影响），起步有短暂加速，各臂关节同时到位。
- **断流暂停：** 任一相机断流 0.5 秒就自动暂停。

### 加入自己的执行方式

执行器在 `macbot/policy/executors.py` 里，接口只有四个方法：

```python
class MyExecutor(Executor):
    def start(self): ...                  # 如果有线程，在这里启动
    def reset(self, active: bool): ...    # 清空动作块；active=True 时从当前观测重新规划
    def target(self, now: float): ...     # 返回 (关节目标, 块编号)，或 (None, 编号)
    def describe(self) -> str: ...
```

`AsyncChunk.accept()` 是一个钩子，决定是否切到新推理出的块、从哪一步开始切。继承它就能试验新的切换规则，不用改控制环。

## 6. 回看

```bash
macbot replay                                   # 打开终端显示的地址
macbot video ~/macbot-data/rollouts/cubes_to_box/rollout_003.npz --height 720
macbot health --data ~/macbot-data/rollouts/cubes_to_box
```

独立回放和运行页共用同一个播放器与评审记录。`health --data` 指向试验目录时，报告会列出每条试验评审后的
结果、备注和空抓次数。回放会标出判为无物体的闭合起点。

检测使用夹爪在收到闭合指令后、重新张开前的稳定读数，对照持物与空夹的校准值。跳过录制开始时已经闭合的
区间，以及结尾没有再次张开的闭合。部分夹取或没有稳定下来的闭合记为无法判定。内置校准适用于二值指令
端点约为 −120° 和 0° 的 reBot 真机软夹爪。其他配置可设置 `robot.grasp_miss: {held: ..., empty: ..., tolerance: ...}`，
单位与关节读数相同；设为 `false` 可关闭检测。闭合、张开指令阈值若与导出配置不同，可在该配置中指定
`close_above` 和 `open_below`。没有校准或两组容差区间重叠时，结果显示为未知。

### 导出已评审的试验供微调

```bash
macbot export --rollouts --data ~/macbot-data/rollouts/cubes_to_box --out ~/data/cubes_rollouts
macbot export --with-rollout-labels --data ~/macbot-data/raw/cubes_to_box --out ~/data/cubes_demos
```

试验导出沿用示范数据的 `observation.state`、`action`、相机键名、画面尺寸和构图元数据。臂关节动作取已保存
的从臂指令。二值策略在 `gp ≥ 0.5` 时夹爪标为 `1`，否则为 `0`。人工接管帧用领臂读数和示范导出相同的
`--close-above` / `--open-below` 滞回阈值（默认来自 `export` 配置）。保持帧和没有有效策略输出的帧使用
保存的夹爪指令；输出原始角度的策略也用指令做滞回二值化。有 `gp` 时，标签反映策略意图；执行端的张开
防抖可能让实际指令短暂不同。

逐帧增加 `takeover`、`grasp_miss`（在判定为空抓的区间内）和 `hand_mark`（标记前后各 0.5 秒）。逐试验
Parquet 元数据写入评审后的 `success`、`source="rollout"` 和空抓统计。帧标签为 0 代表未标出空抓；
`grasp_miss_count=null` 表示没有可用校准。排除的试验会跳过，其余成功、失败和人工修正均可导出。

示范导出加 `--with-rollout-labels`，可得到相同字段，帧标签填 0，`source="demonstration"`，`success=true`；
常规示范导出不受影响。用 LeRobot 合并前检查帧率、机器人、关节顺序、相机键名、尺寸和
`meta/camera_framing.json`。旧试验可能缺少构图字段，请确认采集设置后再合并。保留核对过的构图文件和
`gripper.json`。优势值在训练机计算。

`macbot bench --policy DIR` 用空白图像测这台 Mac 上的推理耗时。

## 假硬件模式

`collect`、`run` 和所有网页都支持 `--fake`：机械臂用一阶滞后模型模拟，相机用合成画面代替。`run --fake` 时策略是真的在推理，所以能在碰真机之前完整检查加载、编译和执行器。`--autostart 秒数` 会自动运行一段时间、保存一次试验，然后退出。

假硬件数据写入 `<任务名>_fake`。通过 `export --data`、`health --data` 或 `replay --dir` 指定对应的目录。未内置关节列表的从臂可通过 `robot.joints` 按顺序给出关节名。

## 物理仿真

安装 [MuJoCo 依赖](setup_zh.md#本地-mujoco-仿真)，各步骤使用同一个配置：

```bash
macbot --config configs/so101_sim.yaml collect
macbot --config configs/so101_sim.yaml health
macbot --config configs/so101_sim.yaml export
macbot --config configs/so101_sim.yaml run --policy path/to/pretrained_model
macbot --config configs/so101_sim.yaml replay
```

按 **S** 重置场景并录制脚本主臂的动作。这些数据用于测试流程，脚本不会完成任务。保存后可以在采集控制台回放。运行策略时，checkpoint 的状态和动作必须是同顺序、同单位的六维关节值，相机名称和尺寸也要匹配（示例中 `front` 和 `wrist` 均为 320×240）。**开始**会为新试验重置场景，**暂停**会冻结场景；保存或丢弃当前试验后可点击**重置场景**。仿真也支持 `--autostart 秒数`。仿真与真机数据请存放在不同的任务目录。
