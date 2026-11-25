# 仿真

[English](simulation.md)

仿真场景在 MuJoCo 里放了一只 SO-101、四个方块和一个盒子。你可以用键盘操作机械臂、生成脚本示范、在 CUDA 机器上训练策略，再回到 Mac 上评测，用的命令和真机完全一样。

!!! note "目前验证到哪一步"

    整个流程已经端到端跑通，但还没有训练出能完成四方块任务的策略。仿真结果也不代表真机表现。仿真数据和真机数据请分开存放。

## 1. 安装

需要 Apple 芯片的 Mac 和 Python 3.12 或更高版本。

```bash
pip install 'macbot[sim]'      # 源码目录中用 pip install -e '.[sim]'
cp configs/so101_sim.yaml ./tabletop.yaml
macbot --config tabletop.yaml doctor
```

示例配置是 [configs/so101_sim.yaml](../configs/so101_sim.yaml)。仿真场景完全由这个文件配置，`macbot init` 向导只用于实体机械臂。

!!! warning "场景会运行下载的代码"

    首次运行会下载固定版本的 EnvHub 场景，包括 Python 代码和 MuJoCo 资源，并以 `trust_remote_code` 方式运行。开启之前请先审阅这些代码。缓存之后，场景加载不再需要连接 Hub。

??? note "代理和渲染"

    如果场景还需要通过 SOCKS 代理下载，请在同一环境中运行 `pip install 'httpx[socks]'`。`macbot doctor` 会检查这一项。

    在 macOS 上，仿真器会打印它使用的离屏渲染器。默认使用硬件 CGL；创建失败时会给出警告并改用软件渲染，这时遥操作可能明显慢于场景的 30 Hz 时钟。

## 2. 用键盘操作机械臂

示例配置用正弦波驱动机械臂，适合快速检查流程。要用键盘控制，在你的配置副本中设置 `teleop.type: keyboard`，保留 `teleop.hz: 30`，然后运行：

```bash
macbot --config tabletop.yaml collect
```

点击视频画面，按 ++enter++ 开始。视频下方的图例会高亮你按住的键。

| 按键 | 动作 |
|---|---|
| ++w++ / ++s++ | 按第一个画面的方向沿桌面前进、后退 |
| ++a++ / ++d++ | 在第一个画面中向左、向右 |
| ++e++ / ++q++ | 抬高、降低夹爪 |
| ++z++ / ++x++ | 从上方看顺时针、逆时针转腕 |
| ++space++ | 张开或闭合夹爪；闭合后移动时保持闭合 |
| ++shift++ | 按住时平移和转腕变慢，便于微调 |
| ++enter++ | 开始新的一条，或保存当前这条 |
| ++backspace++ | 丢弃这条 |
| ++escape++ | 停止这条，之后再选择保存或丢弃 |

夹爪默认朝下。按仿真时间计，平移速度为 6 cm/s，转腕为 45°/s，有 0.10 s 的加速过程；按住 ++shift++ 时两者都降到四分之一。松开按键或者点到视频以外，运动就会停止。预览画面会显示夹爪尖的高度及其在桌面上的投影，运动被限位挡住时会有提示。用键盘控制时，`cameras` 里的第一台相机请用固定视角的相机。

!!! tip "怎么抓起一个方块"

    先抬高再横跨桌面。把夹爪对准方块，下降套住它，闭合，抬起，移到盒子上方再张开。如果俯视工作空间够不到，可以降低夹爪，或者往机械臂方向退一点。

任务成功或超时时，控制台会自动保存。用脚本发送的按键序列通过采集接口完成过四方块任务；在一次简短的真人试用中，有人手动把一个方块放进了盒子。

仿真数据默认保存在 `~/macbot-data/sim/raw/<task>`，控制台启动时会打印实际目录。显式设置的 `data_root` 或 `collect --out` 优先。

??? note "用真实的 SO-101 主臂操作仿真从臂"

    标定过的 SO-101 主臂可以驱动仿真中的从臂：

    ```yaml
    teleop: {type: so101_leader, port: /dev/tty.usbmodemYOUR_LEADER, id: leader, hz: 30}
    ```

    需要标定时，用相同的端口和 ID 运行 `lerobot-calibrate --teleop.type=so101_leader --teleop.port=... --teleop.id=leader`。macbot 通过保存的 LeRobot 标定读取绝对关节角度。这条路径目前只用模拟的主臂测试过。用主臂和用脚本输入时，录制按键的用法相同。

## 3. 生成脚本示范

```bash
macbot --config tabletop.yaml sim demos --episodes 50 --seed 0 --out ./demos
macbot --config tabletop.yaml health --data ./demos
macbot --config tabletop.yaml export --data ./demos --out ./dataset
```

脚本从场景中读取方块位置，用逆运动学规划动作。`--episodes` 统计的是尝试次数，方块位置刁钻时可能失败。每次尝试都记录在 `demos_*.json` 中；默认只保存成功的示范，加 `--keep-failures` 会把失败的也保存下来。

## 4. 在 CUDA 机器上训练

把导出的整个 `dataset` 目录拷到 CUDA 机器上，用 LeRobot 训练 ACT。如果想保留解码后的原始像素、避免再做一次有损视频编码，导出时加 `--images`。

```bash
pip install 'lerobot[training]==0.7.0'
lerobot-train --dataset.repo_id=local/so101-tabletop --dataset.root=./dataset --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --steps=5000 --save_freq=1000 --batch_size=8 --num_workers=0 \
  --output_dir=./act-tabletop
```

把 `act-tabletop/checkpoints/last/pretrained_model` 拷回 Mac，放在 `./pretrained_model`。

## 5. 在 Mac 上评测

请用示范没有用过的种子评测：

```bash
macbot --config tabletop.yaml eval \
  --policy ./pretrained_model \
  --backend torch --mode chunk --k 30 --episodes 50 --seed 1000 --trace-only --out ./trials
macbot --config tabletop.yaml health --data ./trials
macbot --config tabletop.yaml replay --dir ./trials
```

`eval_*.json` 列出每个种子的结果、平均回合长度，以及成功率的 95% Wilson 区间。四个方块的中心都通过场景的盒子检测才算成功，默认 3,600 步超时。

- `--policy` 接受本地权重或 `owner/model@revision`。
- `--videos` 输出并排的 MP4。`--trace-only` 只在策略推理时渲染，保存关节轨迹但不保存相机回放，大批量评测时快得多；`health` 和 `replay` 的轨迹视图照样可用。
- 两种方式都会运行每一个仿真步。策略给出的目标直接发给仿真器，不经过实体机械臂的速度斜坡。
- 使用 `async` 和 `rtc` 时，`--delay-steps` 可以模拟推理延迟，默认为零。

比较不同的运行时，请记下场景配置、种子范围、策略版本和执行参数。
