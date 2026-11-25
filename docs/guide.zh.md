# 工作流程

[English](guide.md)

本页以一个任务为例，从录制示范一直走到回看策略试验。默认你已完成[快速开始](quickstart.zh.md)，`data_root` 为默认的 `~/macbot-data`，任务名为 `cubes_to_box`。

```
~/macbot-data/
  raw/cubes_to_box/episode_000.npz ...      录制的示范
  datasets/cubes_to_box_bin/                导出的 LeRobot 数据集（含 gripper.json）
  rollouts/cubes_to_box/rollout_000.npz ... 策略试验记录
  logs/                                     每次运行一个日志文件
```

!!! note "关于 sudo"

    本页命令都以普通用户运行，适用于 UVC 相机。Orbbec SDK 可能需要 root 权限，这时按[快速开始](quickstart.zh.md#特殊硬件)里的写法用 `sudo -E` 启动 `collect` 和 `run`。使用非默认配置时写成 `macbot --config my.yaml <command>`，见[环境配置](setup.zh.md#lerobot-机械臂与-uvc-相机)。

## 1. 采集

```bash
macbot collect
```

控制台会在浏览器中打开（端口空闲时为 8766）。

| 按键 | 作用 |
|---|---|
| ++s++ | 开始录制 |
| ++enter++ 或 ++arrow-right++ | 保存这条示范 |
| ++arrow-left++ | 丢弃这条示范 |

- **指令。** 空闲时可以修改指令框。每条示范保存自己的指令，作为 π0.5 的语言提示。
- **回放。** 点击任意一行的**回放**，可以拖动查看所有相机画面和关节曲线。
- **删除。** **删除**会把文件移到 `deleted/`，编号不会被复用。
- **退出。** **结束并回零**（或 ++ctrl+c++）会让参考 reBot 分两段回到零位，通用 LeRobot 机械臂回到起始姿态，然后断开连接。

每条示范保存后 macbot 都会检查。短于 `record.min_episode_s` 的示范会被拒绝。以下问题会标为警告：

- 画面延迟的 p95 超过 `checks.max_cam_age_s`
- 重复帧比例超过 `checks.max_repeat_ratio`
- 某个关节一帧内跳变超过 `checks.max_joint_step_deg`
- 迟到采样比例超过 `checks.max_late_ratio`

??? note "示范文件格式"

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
macbot health                  # 每条示范的报告
macbot health --split 27       # 第 27 条之前和之后的中位数，比如修过机械臂之后
```

每条示范的报告包括：

- 让指令和实测最对齐的滞后，以及这个滞后下跟踪误差的 p95
- 主臂静止时的偏差，标定漂移会体现在这里
- 夹爪的活动范围
- 最长的帧间隔和重复帧数

`macbot health` 还会列出各关节电机的最大值和带时间戳的看门狗事件。用 `--data` 可以改为读取某个运行目录。

### 电机监控与看门狗

`collect` 和 `run` 的关节面板会显示每个电机的温度，并标出最热的一个。

| 条件 | 默认值 | 结果 |
|---|---|---|
| 温度超过 `safety.temp_warn_c` | 60 °C | 高亮显示；电机降到 `safety.temp_resume_c`（55 °C）以下之前，**开始**保持锁定 |
| 在起始姿态以外静止保持超过 `safety.hold_warn_s` | 30 s | 面板建议回到起始姿态，让电机休息 |
| 电机报告故障位，或温度超过 `safety.temp_alarm_c` | 70 °C | 看门狗停止策略或遥操作 |
| 臂关节误差持续超过 `safety.track_deg` 的时间长于 `safety.track_s` | 8°，0.3 s | 看门狗停止策略或遥操作 |

看门狗触发后，会把实测姿态作为指令发出，并在本次会话中锁定运动指令。跟踪检查与实际发出的最后一条指令比较，并且不包括夹爪，因为夹住物体时夹爪本来就会堵转。

!!! warning "触发后请托住机械臂"

    失去力矩的电机可能保持不住姿态。必要时托住机械臂，查明原因后再重新启动。退出时仍会慢慢回零，电机断使能时机械臂不会掉落。

??? note "阈值、遥测和记录字段"

    三个温度阈值是 macbot 设得比较保守的值。[RS-06 手册](https://files.seeedstudio.com/products/RobStride/Product%20Literature/RS06/RS06User%20Manual251112.pdf)里的 145 °C 是热敏电阻故障阈值，不是机械臂建议的工作温度。

    B601-RS 复用已有的位置反馈：一个以 °C 为单位的温度、以 N·m 为单位的力矩（电机方向）、状态位，以及缓存的故障字和警告字。这些反馈里没有独立的转子温度或电流。其他硬件可能没有电机遥测。假硬件和仿真里的数值是合成的。通用从臂的跟踪误差使用机械臂自己的关节单位。

    被看门狗中断的录制保存在数据目录下的 `watchdog/` 中，`health` 会单独列出，它们不进入普通结果和导出。NPZ 文件按 `joint_names` 顺序保存 `motor_temp_c`、`motor_rotor_c`、`motor_torque_nm`、`motor_status`、`motor_fault` 和 `motor_warning`（不可用时为 `NaN` 或 `-1`），`meta` 中保存 `watchdog_events`。

    `safety.enabled: false` 关闭看门狗的判断，但保留遥测和温度显示。

## 3. 导出

```bash
macbot export                              # 保留设备单位下的原始动作
macbot export --gripper binary             # 夹爪动作为 0（张开）或 1（闭合），并写出 gripper.json
```

未通过检查的示范会被跳过。加 `--skip-warnings` 会把带警告的示范也跳过。

抓取放置任务用二值夹爪比原始角度效果好。

- **标签。** 主臂读数升到 `--close-above`（−40°）以上时开始记为"闭合"，直到降到 `--open-below`（−80°）以下。
- **部署。** "张开"发送 `--open-deg`（−150°），"闭合"发送 `--close-deg`（0°）。夹爪力限制决定夹得多紧。

## 4. 训练

训练在带 NVIDIA GPU 的 Linux 机器上进行。把数据集目录拷过去，然后运行：

```bash
DATASET=~/data/cubes_to_box_bin GPUS=0,1,2,3 ./train/train_pi05.sh cubes_pi05
DATASET=~/data/cubes_to_box_bin ./train/train_act.sh cubes_act
```

- `train_pi05.sh` 通过 `train/train_augmented.py` 微调 `lerobot/pi05_base`。这个脚本是 `lerobot-train` 加上在 GPU 上做的 openpi 风格图像增强：场景相机做裁剪和旋转，所有相机做颜色抖动。
- `WRIST_SHIFT=0.03` 给腕部相机加一点平移，适合两次录制之间腕部支架移动过的情况。

把 `checkpoints/<step>/pretrained_model` 拷回 Mac。二值夹爪数据集还要把 `gripper.json` 放在它旁边，或者上面一两层目录里。

!!! tip "用 `--grip-lead` 提前闭合夹爪"

    二值标签要等主臂越过 `--close-above` 才切换为"闭合"，这时手指已经基本合上。用它训练的策略也会这么晚才下达闭合指令，而真实夹爪还要走完整个行程（reBot 软夹爪约 0.5 s）。`run --grip-lead 0.35` 会在策略计划在当前动作块的 0.35 s 内闭合时提前闭合。张开不受影响。每次试验在 `gp` 旁边以 `gp_ahead` 记录这个前瞻值。

## 5. 运行策略

```bash
macbot run --policy ~/policies/cubes_pi05/004000 --speed 0.3
```

运行控制台会在浏览器中打开（端口空闲时为 8767），启动时处于**暂停**状态。

| 按键 | 作用 |
|---|---|
| ++s++ | 开始 |
| ++space++ | 暂停并保持当前指令 |
| ++h++ | 慢速回到起始姿态 |
| ++y++ / ++n++ | 把这次试验存为成功或失败 |
| ++t++ | 手动接管，或交还给策略（需要 `--takeover`） |
| ++m++ | 标记手进入画面的时刻，控制方式不变 |

每次试验保存到 `rollouts/<task>/`，包含所有帧、指令与实测关节、各动作块的开始时间和推理耗时。只想测这台 Mac 上的推理耗时，运行 `macbot bench --policy DIR`，它使用空白图像。

**安全限制。** 臂关节每秒最多移动 `run.max_vel × speed` 度，并保持在关节限位内。`--speed` 按比例缩放时间：0.5 时，30 fps 的策略按每秒 15 步执行。**回起始姿态**按 `run.home_vel` 移动（最远的关节默认每秒 25 度，与 `--speed` 无关），短暂加速后所有臂关节同时到位。任何相机停顿 0.5 s，运行会暂停。

### 在运行控制台里回看试验

暂停时，点击**已保存的试验**下的一行即可回放。列表先显示这个任务最新的五次试验（包括之前的会话），**显示全部**打开完整列表。可以拖动或播放同步的相机画面和关节曲线，位置读数显示当前动作块及其推理耗时。**回到实时画面**回到相机。开始或回起始姿态会关闭播放器。

在回放面板中可以修改**结果**、填写一行**备注**，或者勾选**排除此试验**并填写原因（例如 GPU 被占用），然后点击**保存**。被排除的试验仍可查看，但不计入成功率统计、健康汇总和导出。修改保存在录制文件旁边的 `rollout_XXX.review.json` 中，原始录制不变。拷贝评审过的试验时，请把这两个文件放在一起。

### 执行方式

`--mode` 决定动作块如何执行。所有方式都会在策略步之间插值，供 60 Hz 控制循环使用。

| 方式 | 做法 | 默认用于 |
|---|---|---|
| `chunk` | 推理一次，执行前 `--k` 步，再推理。推理期间机械臂等待。 | |
| `ensemble` | 持续推理。每个时刻的目标是覆盖当前时刻的所有动作块按 exp(m · age) 加权的平均（ACT 的时间集成）。 | ACT |
| `async` | 当前动作块继续执行的同时在后台推理。新块与旧块在时间上对齐。 | |
| `rtc` | 在 `async` 基础上加 Real-Time Chunking：用旧块里尚未执行的部分引导新块去噪，前后两块衔接平滑。 | π0.5 |

`async` 和 `rtc` 在上一次推理结束后立即再次推理，所以每个动作块只执行几步。`--query-every K` 改为每 `K` 个策略步推理一次，对应 RTC 论文中的执行步长。`rtc` 对推理期间已执行的动作用权重 1 引导新块，之后权重衰减，到 `--rtc-horizon` 时为零。`--rtc-schedule exp` 改用论文中的指数衰减，默认是线性衰减。动作块长 50 步时，`--query-every 15 --rtc-horizon 35 --rtc-schedule exp` 与论文设置一致。

### 手动接管

接管让你在试验进行中用主臂纠正策略。用 `--takeover` 或 `run.takeover: true` 开启，程序会连接 `collect` 使用的已标定主臂。

1. 策略运行时按 ++t++ 或点击**手动接管**。从臂保持当前指令。
2. 用手移动主臂，直到它和从臂对齐。对齐面板会显示还有哪些需要调整。
3. 之后从臂跟随主臂，开始时是软启动。
4. 再按 ++t++ 交还控制。策略从当前观测重新规划。

暂停会取消手动控制，之后按 ++s++ 恢复策略。回起始姿态、保存和退出仍然可用。`--fake` 和仿真使用脚本控制的主臂，也要通过同样的对齐检查。

??? note "对齐和跟随的细节"

    **臂关节**都要在 `run.takeover_align_deg`（默认 30°）以内。

    **夹爪。** 使用二值夹爪策略时，如果从臂当前指令为闭合，主臂读数必须高于 `export.gripper_close_above`；如果为张开，则必须低于 `export.gripper_open_below`。读数在两条阈值之间时，从臂继续保持。使用原始角度时，主臂夹爪与从臂夹爪指令之差要在 `run.takeover_gripper_align_deg`（默认 15°）以内。

    **软启动。** 从臂先以 `run.home_vel` 沿直线移动，一秒内加速到正常上限（`run.max_vel × speed`），各关节差都在 2° 以内时提前结束。录制中软启动和对齐一样标为 `2`，之后才标为人工操作（`1`）。

    **跟随时**，机械臂沿用策略的速度和关节限位，夹爪指令每秒最多变化 `300 × speed` 度，reBot 夹爪继续使用采集时的力限制。舵机库保持住的读数（0° 附近，比如夹爪合拢时）与 `collect` 一样照常使用。

    **主臂信号中断。** 读不到主臂时从臂保持不动。如果 0.5 s 内恢复，从臂重新软启动后继续跟随；中断更久则回到对齐。

??? note "接管试验记录了什么"

    试验会增加 `takeover`（`0` 策略、`1` 跟随、`2` 对齐并保持）和 `leader`（逐帧、逐关节的主臂读数，不可用时为 `NaN`）。元数据中 `takeover_count` 记录请求次数，`takeover_seconds` 记录实际跟随的秒数。试验进行中按 ++m++ 或点击**标记手进入画面**，会给 `hand_marks` 增加一个时间戳，不开启接管也可以用。回放在时间轴和曲线上标出对齐和跟随的区间，以及每次手进入画面的时刻。

## 6. 回看

```bash
macbot replay                                   # 打开终端里打印的地址
macbot video ~/macbot-data/rollouts/cubes_to_box/rollout_003.npz --height 720
macbot health --data ~/macbot-data/rollouts/cubes_to_box
```

单独的回放页面和运行控制台使用同一个播放器和标注。`health --data` 指向试验目录时，会列出每次试验的评审结果、备注和空抓次数。回放会标出每次检测到的空抓的起点。

??? note "空抓是怎么检测的"

    检测器在一次闭合指令快结束、夹爪重新张开之前，把实测的夹爪位置与两个标定好的水平比较。录制开始时已经在进行的闭合，以及最后一次没有再张开的闭合，都会被跳过。闭合不完整或没有稳定下来时，结果记为无法判断。

    内置配置适用于二值端点约为 −120° 和 0° 的真实 reBot 录制（参考软夹爪）。其他设备请按夹爪自己的单位设置 `robot.grasp_miss: {held: ..., empty: ..., tolerance: ...}`，设为 `false` 则关闭检测。如果指令阈值与导出配置不同，在这个配置里写上 `close_above` 和 `open_below`。没有标定，或者容差区间互相重叠时，结果为未知。

### 导出评审过的试验用于微调

```bash
macbot export --rollouts --data ~/macbot-data/rollouts/cubes_to_box --out ~/data/cubes_rollouts
macbot export --with-rollout-labels --data ~/macbot-data/raw/cubes_to_box --out ~/data/cubes_demos
```

试验导出与示范使用相同的 `observation.state`、`action`、相机键、图像尺寸和取景元数据，因此两者可以合并。导出保留成功、失败和纠正，跳过被排除的试验。

| 列或字段 | 含义 |
|---|---|
| `takeover` | 逐帧：`0` 策略、`1` 人工、`2` 对齐 |
| `grasp_miss` | 逐帧：处在检测到的空抓之内 |
| `hand_mark` | 逐帧：在手部标记前后 0.5 s 以内 |
| `success` | 这次试验的评审结果（片段元数据） |
| `source` | `"rollout"`；用 `--with-rollout-labels` 导出示范时为 `"demonstration"` |
| 空抓计数 | `grasp_miss_count=null` 表示这次试验没有可用的标定 |

逐帧标签为零表示没有检测到空抓。`--with-rollout-labels` 让二值示范导出带上相同的列，逐帧标签全为零，`source="demonstration"`，`success=true`；普通的示范导出不变。

??? note "试验导出中的动作标签"

    臂关节动作是保存下来的从臂指令。二值策略的夹爪在 `gp ≥ 0.5` 时为 `1`，否则为 `0`。人工控制期间，主臂读数经过与示范导出相同的 `--close-above` 和 `--open-below` 迟滞（默认值取自 `export`）。保持帧和没有有效策略输出的帧使用保存的夹爪指令；原始角度策略也使用指令迟滞。有 `gp` 时，标签表达的是策略的意图，所以在部署时的张开防抖生效期间，它可能短暂地与实际指令不同。

!!! warning "与示范合并之前"

    请核对 fps、机器人和关节顺序、相机键、图像尺寸和 `meta/camera_framing.json`。较早的试验可能缺少取景字段，合并前请确认当时的设置。合并后的数据集要带上核对过的取景信息和 `gripper.json`。优势值请在训练机器上计算。

### 导出用于 DAgger 和优势条件训练的片段

```bash
macbot export --rollouts --segments --min-segment-s 1.0 --pre-takeover-s 1.7
```

不加 `--segments` 时，每次试验导出为一个片段。加上后，导出按三条规则处理。

1. **去掉对齐帧**（`takeover=2`）。对齐、保持、软启动和主臂信号中断记录的是控制系统交接时的行为，不应成为训练目标。
2. **在交接处切分。** 每段连续的策略控制（`0`）或人工控制（`1`）各成为一个片段，这样动作块不会跨过交接，人工交还策略后立即接管也一样。每个片段的帧序号和时间戳从零开始。LeRobot 会在片段末尾补齐动作块，训练时用 `action_is_pad` 屏蔽补齐部分，或者只选完整的动作块。
3. **标记干预之前的行为。** `pre_takeover`（`uint8`）标出原试验中距离下一次接管请求不到 `--pre-takeover-s` 的策略帧（默认 1.7 s，30 fps 下约 50 帧）。录制里没有按键时间戳，所以窗口在第一次采样到从策略切换到对齐或人工控制时结束。人工控制期间的主臂信号中断不会开启新的窗口。窗口设为 `0` 则不标记。

短于 `--min-segment-s`（默认 1.0 s，按 `帧数 / fps` 计算）的片段会被丢弃。

??? note "片段元数据与合并"

    保留下来的每个片段都带有 `source="rollout_policy"` 或 `"rollout_human"`、`trial_name`（NPZ 文件名）、`original_start_time`（距试验开始的秒数）和 `segment_index`（在策略段和人工段中从零开始编号，在丢弃短片段之前计数）。评审的 `success` 和空抓计数针对整次试验。被排除的试验仍然排除，`grasp_miss` 和 `hand_mark` 与保留下来的帧对齐。汇总会报告纳入的试验数、各来源的片段数、去掉的对齐帧和短片段，以及保留的人工控制秒数和干预前帧数。

    片段导出比 `--with-rollout-labels` 多一列。要与示范按相同的结构合并，请给示范补一列全零、形状为 `(1,)` 的 `uint8` 列 `pre_takeover`。

## 没有硬件时

**假硬件。** `collect`、`run` 和所有控制台都支持 `--fake`：机械臂用一阶滞后模型模拟，相机画面是合成的。`run --fake` 加载的是真实策略，所以能在接触机器人之前检查加载、编译和执行器。`--autostart SECONDS` 会自动运行、保存一次试验并退出。假硬件的录制保存在 `<task>_fake` 下；读取时给 `export` 或 `health` 传 `--data`，给 `replay` 传 `--dir`。macbot 未列出的从臂在使用 `--fake` 时，请把 `robot.joints` 设为按顺序排列的关节名。

**仿真。** 安装 [MuJoCo 依赖](setup.zh.md#本地-mujoco-仿真)，每一步使用同一个配置：

```bash
macbot --config configs/so101_sim.yaml collect
macbot --config configs/so101_sim.yaml health
macbot --config configs/so101_sim.yaml export
macbot --config configs/so101_sim.yaml run --policy path/to/pretrained_model
macbot --config configs/so101_sim.yaml replay
```

按 ++s++ 会重置场景并录制脚本主臂的动作。这是用于测试流程的数据，脚本并不完成任务。保存的示范可以在采集控制台里回放。运行策略时，权重的关节状态和动作要有六个值，顺序和单位一致，相机键和尺寸也要匹配（示例中是 `front` 和 `wrist`，都是 320×240）。**开始**会为新试验重置场景，**暂停**会冻结场景；使用**重置场景**之前，请先保存或丢弃当前试验。`--autostart SECONDS` 在仿真中同样可用。仿真数据和真机数据请放在不同的任务目录。

[仿真](simulation.zh.md)页介绍笛卡尔空间键盘控制、脚本示范、在 CUDA 机器上训练，以及在 Mac 上按种子评测策略。

## 开发者参考

### 加入新的执行方式

执行器在 `macbot/policy/executors.py` 中，共用一个四个方法的接口：

```python
class MyExecutor(Executor):
    def start(self): ...                  # start threads, if any
    def reset(self, active: bool): ...    # drop chunks; active=True: plan from the current observation
    def target(self, now: float): ...     # -> (joint target, chunk id) or (None, id)
    def describe(self) -> str: ...
```

`AsyncChunk.accept()` 决定是否切换到新推理出的动作块，以及从哪一步切换。继承它就可以尝试新的切换规则，不用改控制循环。

### 测量换块时的进度倒退

机械臂比指令落后几个控制步，所以根据实测姿态规划的策略，可能返回一个起点落后于当前指令的动作块。为了测量这一点，`chunk`、`async` 和 `rtc` 会在试验中保存每一次换块：

| 字段 | 内容 |
|---|---|
| `ex_chunk` | 新动作块 |
| `ex_prev` | 被替换的动作块（第一个块为 `NaN`） |
| `ex_p` | 新块第一个动作替换的是旧块中的哪个位置 |
| `ex_entry` | 新块从哪个位置开始执行 |
| `ex_q_obs` | 策略看到的实测状态 |

每分钟试验大约多占 1 MB。然后运行：

```bash
python tools/chunk_shift.py ~/macbot-data/rollouts/<task>
```

它按试验和执行方式打印让新块与旧块最对齐的偏移 `delta_hat`（新块起点落后于指令时为正），分别在前 10 个动作上和 `rtc` 引导到旧块的前缀之后计算，同时给出指令与实测姿态之间的跟踪滞后。
