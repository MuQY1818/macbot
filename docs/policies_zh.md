# macOS 上的 LeRobot 策略

[English](policies.md)

`macbot policies` 列出 LeRobot 0.7.0 注册的策略及实测状态。ACT 和 π0.5 保留原有封装和运行默认值。
其他类型走 PyTorch 通用适配器；也可以用 `--policy-impl generic` 显式让 ACT 或 π0.5 使用通用适配器。

## 使用自己的 checkpoint

使用本地 LeRobot `pretrained_model` 目录，包含 `config.json`、模型权重、
`policy_preprocessor.json`、`policy_postprocessor.json` 及处理器引用的状态文件。
按需安装 LeRobot 的相应扩展，例如 `pip install 'lerobot[diffusion]==0.7.0'` 或
`pip install 'lerobot[smolvla]==0.7.0'`。一些策略另需原始骨干模型和分词器。请使用可信来源的权重和依赖。

下面是六关节、两相机、有语言指令的配置写法，字段须按 checkpoint 修改。`save_size` 是 `[宽, 高]`，
而 checkpoint 的图像形状是 `[通道, 高, 宽]`。相机名与特征后缀相同可省略 `camera_map`；
`observation.image` 等非标准特征名需写全。

```yaml
robot: {type: lerobot, follower: {type: so101_follower}}
task: {name: demo, instruction: "put the cube in the box"}
cameras:
  top: {type: uvc, save_size: [256, 256]}
  wrist: {type: uvc, save_size: [256, 256]}
policy:
  camera_map: {top: camera1, wrist: camera2}
```

```bash
macbot policies
macbot --config robot.yaml bench --policy path/to/pretrained_model --profile
macbot --config robot.yaml run --policy path/to/pretrained_model --fake --autostart 3
```

`bench` 测量从转换观测到输出动作的耗时，`--device cpu` 可用于 CPU 对照。`--profile` 另做一轮同步计时，
分别统计观测转换、预处理、模型推理、后处理和结果复制。`run --mode auto` 对声明支持直接预测动作块的策略
使用同步分块（`--k` 限制播放步数）；需要维护历史或队列的策略按 `record.fps × speed` 调用 `select_action`。
每次开始都会清空策略和处理器的运行状态。通用推理不支持异步、时间集成或 RTC。

后处理后的动作必须是**绝对关节位置**，且顺序和单位与机械臂一致。维度相同不能证明这些语义正确。
在真机上使用通用策略前，核对后设置 `policy.action_space: joint_position`。macbot 不做逆运动学或关节坐标转换。
它会拒绝已知的末端执行器动作、缺少必要相机、额外传感器模态及非有限输出。checkpoint 声明的占位相机由模型
自行填充并屏蔽。原有 ACT／π0.5 路径不受影响。

## MPS 实测

MacBook Pro，Apple M5 Pro（48 GiB），macOS 27.0.1，PyTorch 2.11.0，LeRobot 0.7.0，Python 3.12。
通用路径计时包含处理器与复制动作到 CPU，未连接真机。对固定生成的 RGB 和关节状态预热 5 次、计时 30 次。
CPU 使用 fp32、4 线程。误差测试让两端共用 CPU 正态随机数和类别采样；计时使用各设备原生采样。
内存为进程 RSS 峰值。逐步执行的统计混合了队列取值和较慢的重新生成。

| 策略 / checkpoint | 执行方式 | MPS 中位数 / p90（ms） | CPU 中位数 / p90（ms） | 动作最大绝对差 | MPS 峰值 RSS（GiB） |
|---|---|---:|---:|---:|---:|
| ACT，小模型 | 通用，3 步动作块 | 7.21 / 7.93 | 6.02 / 6.16 | 1.31e-6 | 0.66 |
| Diffusion，小模型 DDIM | 逐步，每 3 次重新预测 | 1.44 / 22.24 | 0.11 / 12.43 | 3.10e-6 | 0.80 |
| VQ-BeT，小模型 | 逐步，每 3 次重新预测 | 1.18 / 7.26 | 0.07 / 4.71 | 2.38e-7 | 0.69 |
| TDMPC，小模型 | 逐步，每 2 次重新预测 | 5.30 / 9.42 | 1.06 / 2.23 | 2.38e-7 | 0.64 |

随机权重仅验证运行链路和数值差异，不用于评价任务表现。VQ-BeT 和 TDMPC 各只接受一台相机；
TDMPC 的特征键须为 `observation.image`。VQ-BeT 需要经过训练的 Residual VQ 才能产生有意义的动作。
原有 π0.5 的性能数据见[推理测速](acceleration_zh.md)。

从源码目录复现小模型结果：

```bash
python -m tools.tiny_policies diffusion /tmp/diffusion
# 在 robot.yaml 里配置 top 和 wrist 两台 64 × 64 相机。
for device in cpu mps; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 python -m tools.policy_sweep /tmp/diffusion \
    --config robot.yaml --device "$device" --out "/tmp/diffusion-$device.npy"
done
python -c 'import numpy as n; print(abs(n.load("/tmp/diffusion-cpu.npy")-n.load("/tmp/diffusion-mps.npy")).max())'
```
