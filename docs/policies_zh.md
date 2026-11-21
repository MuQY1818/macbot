# macOS 上的 LeRobot 策略

[English](policies.md)

`macbot policies` 列出 LeRobot 0.7.0 注册的策略及实测状态。ACT 和 π0.5 保留原有封装和运行默认值。
其他类型走 PyTorch 通用适配器；也可以用 `--policy-impl generic` 显式让 ACT 或 π0.5 使用通用适配器。

`run` 和 `bench` 可使用 `--backend auto`，也可在 YAML 中设置 `policy: {backend: auto}`。
`policies` 列出已实现的后端，附加依赖的安装状态和[保真验证](acceleration_zh.md#自动选择后端)另计。

## 使用自己的 checkpoint

实验性的 π0 `--backend mlx` 路径接受标准 224 像素 PaliGemma 2B checkpoint，使用
`policy.camera_map` 和随权重保存的处理器。提供 `--mode chunk` 和 RTC 引导，默认分块。
尚未验证训练权重的保真度和速度，请用离线检查及 `--fake` 评估。

使用本地 LeRobot `pretrained_model` 目录，包含 `config.json`、模型权重、
`policy_preprocessor.json`、`policy_postprocessor.json` 及处理器引用的状态文件。
如果只保存了 PEFT adapter，先合并到底座模型再导出。
按需安装 LeRobot 的相应扩展，例如 `pip install 'lerobot[diffusion]==0.7.0'`、
`pip install 'lerobot[smolvla]==0.7.0'` 或 `pip install 'lerobot[xvla]==0.7.0'`。
一些策略另需原始骨干模型和分词器。请使用可信来源的权重和依赖。

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

`lerobot/smolvla_base` 还需要第三台 256 × 256 相机，对应 `camera3`。对本文实测的
两相机 X-VLA checkpoint，把 `top` 和 `wrist` 分别映射到 `image`、`image2`。

`bench` 测量从转换观测到输出动作的耗时，`--device cpu` 可用于 CPU 对照。`--profile` 另做一轮同步计时，
分别统计观测转换、预处理、模型推理、后处理和结果复制。`run --mode auto` 对声明支持直接预测动作块的策略
使用同步分块（`--k` 限制播放步数）；需要维护历史或队列的策略按 `record.fps × speed` 调用 `select_action`。
每次开始都会清空策略和处理器的运行状态。通用推理不支持异步、时间集成或 RTC。

后处理后的动作必须是**绝对关节位置**，且顺序和单位与机械臂一致。维度相同不能证明这些语义正确。
在真机上使用通用策略前，核对后设置 `policy.action_space: joint_position`。macbot 不做逆运动学或关节坐标转换。
它会拒绝已知的末端执行器动作、缺少必要相机、额外传感器模态及非有限输出。checkpoint 声明的占位相机由模型
自行填充并屏蔽。原有 ACT／π0.5 路径不受影响。

## 支持范围

状态分三档。“已验证”指公开的训练好的 checkpoint 在 MPS 上完成推理并保存假机 rollout，或已有单独验证的专用封装。“能运行”指用小模型或随机权重在 MPS 上跑通了数据链路，同类训练好的模型的延迟和精度还没有测。“未验证”指没有实际运行。
同一策略的其他 checkpoint 仍须核对。此环境中的 LeRobot 工厂成功解析 29 个注册类型的模型类；
导入成功不表示已加载权重或运行推理。

| 策略类型 | 状态与范围 |
|---|---|
| `act`、`pi05` | 原有封装已验证；通用 ACT 也已测试。 |
| `diffusion`、`vqbet`、`tdmpc`、`gaussian_actor` | 能运行，用下表中的离线小模型跑通。GaussianActor 测试的是连续动作。 |
| `pi0` | 能运行，用 2B PaliGemma 前缀、六关节的随机权重 checkpoint 跑通。300M 前缀的视觉和文本嵌入维度不同，两端都会报错。 |
| `pi0_fast` | 用本地分词器在 CPU 和 MPS 加载并生成 token，两端的随机输出都未通过 FAST 动作解码，未验证匹配机械臂的训练权重。 |
| `smolvla` | 用 `lerobot/smolvla_base` 验证：三路 RGB、六维状态与动作，50 步动作块。 |
| `xvla` | 用 `whosricky/xvla-so101-pick-red-cube-2cam` 验证。六维状态由模型自行补齐，第三个屏蔽相机位留空。`ee6d` 基础模型不支持。 |
| `groot` | 未验证。GR00T N1.7 需要原生底座、对应机械臂本体及保存的处理器。 |
| `wall_x`、`wall_oss_05` | 未验证。属于不同模型系列，需要分别配套 checkpoint、骨干和处理器。 |
| `cosmos3_edge`、`cosmos3_nano`、`dm05`、`eo1`、`evo1`、`fineart_vla`、`g05`、`lawam`、`lingbot_vla_v2`、`molmoact2`、`vla_jepa` | 只验证工厂导入，未运行对应机械臂的 checkpoint 和骨干。 |
| `fastwam`、`flux3`、`lingbot_va`、`longwam` | 只验证工厂导入，视频或动作历史以及各自的处理器行为有待验证。 |
| `multi_task_dit` | 能运行，用 DDIM 小模型、本地随机 CLIP 模型及分词器跑通。 |

下表实测无需 `PYTORCH_ENABLE_MPS_FALLBACK`。如果未测 checkpoint 遇到 MPS 不支持的算子，可先试
`--device cpu`。要诊断 CPU 回退，请在启动 macbot **之前**设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，
并查看算子警告。回退无法解决缺少模型文件、维度不符或模型内部使用 float64。
通用适配器会覆盖标准设备处理器的目标设备，并在 MPS 上把其显式 float64 设置改为 float32。

## MPS 实测

MacBook Pro，Apple M5 Pro（48 GiB），macOS 27.0.1，PyTorch 2.11.0，LeRobot 0.7.0，Python 3.12。
通用路径计时包含处理器与复制动作到 CPU，未连接真机。使用固定生成的 RGB 和关节状态，预热 5 次，
小模型计时 30 次，SmolVLA 和 X-VLA 计时 15 次，π0 计时 5 次。
CPU 使用 fp32、4 线程。误差测试让两端共用 CPU 正态随机数和类别采样；计时使用各设备原生采样。
内存为进程 RSS 峰值。逐步执行的统计混合了队列取值和较慢的重新生成。

| 策略 / checkpoint | 执行方式 | MPS 中位数 / p90（ms） | CPU 中位数 / p90（ms） | 动作最大绝对差 | MPS 峰值 RSS（GiB） |
|---|---|---:|---:|---:|---:|
| ACT，小模型 | 通用，3 步动作块 | 7.21 / 7.93 | 6.02 / 6.16 | 1.31e-6 | 0.66 |
| Diffusion，小模型 DDIM | 逐步，每 3 次重新预测 | 1.44 / 22.24 | 0.11 / 12.43 | 3.10e-6 | 0.80 |
| VQ-BeT，小模型 | 逐步，每 3 次重新预测 | 1.18 / 7.26 | 0.07 / 4.71 | 2.38e-7 | 0.69 |
| TDMPC，小模型 | 逐步，每 2 次重新预测 | 5.30 / 9.42 | 1.06 / 2.23 | 2.38e-7 | 0.64 |
| GaussianActor，小模型 | 逐步，连续动作 | 3.70 / 3.97 | 0.60 / 0.67 | 4.77e-7 | 0.62 |
| Multi-Task DiT，小模型 DDIM | 逐步，每 3 次重新预测 | 1.74 / 7.63 | 0.12 / 1.47 | 4.77e-7 | 0.64 |
| π0，随机权重，2B 前缀，fp32 去噪 2 步 | 3 步动作块 | 490.01 / 492.80 | 2150.15 / 2219.61 | 2.68e-6 | 16.74 |
| SmolVLA，官方基础模型 | 50 步动作块 | 297.80 / 309.43 | 1126.30 / 1164.04 | 0.0316 | 3.38 |
| X-VLA，SO-101 两相机微调模型 | 30 步动作块 | 181.89 / 205.54 | 6226.21 / 7688.11 | 1.22 | 4.93 |

随机权重仅验证运行链路和数值差异，不用于评价任务表现。VQ-BeT 和 TDMPC 各只接受一台相机；
TDMPC 的特征键须为 `observation.image`。VQ-BeT 需要经过训练的 Residual VQ 才能产生有意义的动作。
原有 π0.5 的性能数据见[推理测速](acceleration_zh.md)。
SmolVLA 的视觉语言骨干使用 bf16，X-VLA 的 MPS 端沿用 checkpoint 保存的 bf16 设置。和 CPU fp32 的
配对差异以各 checkpoint 的输出单位计。真机使用前，请按任务能接受的误差评估。

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
