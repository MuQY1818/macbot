# SO-101 仿真

[English](simulation.md)

使用 Apple 芯片 Mac 和 Python 3.12 或更新版本。从
[SO-101 示例配置](../configs/so101_sim.yaml) 开始。首次运行会下载并执行固定版本的
EnvHub Python 代码并加载 MuJoCo 资源；启用 `trust_remote_code` 前请先审阅。
场景包含一条 SO-101 仿真机械臂、四个方块和一个盒子。
macOS 会打印选用的离屏渲染器。默认使用硬件 CGL；如果创建失败，会告警并尝试软件渲染。
软件渲染时，遥操作可能显著慢于场景的 30 Hz 时钟。

```bash
pip install 'macbot[sim]'               # 或在源码目录运行 pip install -e '.[sim]'
cp configs/so101_sim.yaml ./tabletop.yaml
macbot --config tabletop.yaml doctor
```

仿真直接由 YAML 配置；USB 设置向导用于实体机械臂。
示例保留原有的正弦输入，用于快速检查流程。在配置副本中设置 `teleop.type: keyboard`，
保留 `teleop.hz: 30`。运行 `macbot --config tabletop.yaml collect`，按 **S** 开始。
让采集页面获得焦点，展开**键盘控制**可查看按键：

| 按键 | 动作 |
| --- | --- |
| I / K | 远移 / 近移 |
| J / L | 左移 / 右移 |
| U / O | 上升 / 下降 |
| N / M | 转动手腕 |
| F / G | 张开 / 闭合 |
| Shift | 按住时微调平移 |

横移前先抬起夹爪。将夹爪对准方块，下降、夹紧、抬升，再到盒子上方松开。
回车保存，左方向键丢弃。任务成功或到达时间上限时自动保存。
已通过采集控制台输入合成按键完成一次完整任务；仍待真人试用。

也可以用已标定的实体 SO-101 领臂操作仿真从臂。设置
`teleop: {type: so101_leader, port: /dev/tty.usbmodemYOUR_LEADER, id: leader, hz: 30}`。
需要标定时，以相同串口和 ID 运行 `lerobot-calibrate --teleop.type=so101_leader
--teleop.port=... --teleop.id=leader`。macbot 读绝对关节角并复用 LeRobot 标定。
领臂路径目前仅做了 mock 测试。

可按种子生成脚本示教，并导出为 LeRobot v3.0：

```bash
macbot --config tabletop.yaml sim demos --episodes 50 --seed 0 --out ./demos
macbot --config tabletop.yaml health --data ./demos
macbot --config tabletop.yaml export --data ./demos --out ./dataset
```

`--episodes` 指尝试次数。脚本读取场景位姿并用逆运动学抓放，困难的初始位置可能失败。
`demos_*.json` 会记录每次尝试；默认只保存成功的示教，指定 `--keep-failures` 可保留失败回合。

把完整的 `dataset` 目录传到 CUDA 机器，在那里用 LeRobot 训练 ACT。
如需保留录制画面的解码像素，导出时加 `--images`，避免再次进行有损编码。

```bash
pip install 'lerobot[training]==0.7.0'
lerobot-train --dataset.repo_id=local/so101-tabletop --dataset.root=./dataset --dataset.video_backend=pyav \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --steps=5000 --save_freq=1000 --batch_size=8 --num_workers=0 \
  --output_dir=./act-tabletop
```

将 `act-tabletop/checkpoints/last/pretrained_model` 带回 Mac，放在 `./pretrained_model`，
使用未出现在示教中的种子评估。

```bash
macbot --config tabletop.yaml eval \
  --policy ./pretrained_model \
  --backend torch --mode chunk --k 30 --episodes 50 --seed 1000 --trace-only --out ./trials
macbot --config tabletop.yaml health --data ./trials
macbot --config tabletop.yaml replay --dir ./trials
```

`eval` 支持本地 checkpoint 或 `owner/model@revision`。加 `--videos` 可另存双路并排 MP4。
批量评估 SO-101 桌面场景时，可用 `--trace-only` 只在策略查询时渲染，只保存关节轨迹；
`health` 和 `replay` 的轨迹视图仍可使用。两种方式都会执行完整的仿真步。
策略按仿真步推进并直接下发关节目标，不经过实体机械臂的速度斜坡。异步和 RTC 模式可用
`--delay-steps` 模拟推理延迟，默认零延迟。对比结果时应记录场景配置、种子范围、
策略版本及执行选项。

`eval_*.json` 保存每个种子的结果、平均回合长度和成功率的 95% Wilson 区间。
当四个方块的中心都通过盒内判定时，场景记为成功；默认上限为 3,600 步。
仿真成功不代表真机成功，仿真与真机数据应分开存放。
流程已跑通；这并不代表训练后的策略已经能完成四方块任务。
