# Apple 芯片上的 π0.5 推理

[English](acceleration.md)

macbot 默认使用 PyTorch。可选的 MLX 后端直接读入标准 LeRobot 0.7.0 `pi05` safetensors，复用 checkpoint 的 LeRobot tokenizer、状态归一化和动作后处理。它支持单条观测和一或多个已配置的 RGB 视角，暂不支持历史记忆变体、checkpoint 中配置的 RTC 或适配器。

## 自动选择后端

**保真**指在已测试的配置上，使用相同权重和初始噪声或种子时，动作误差在 FP32 CPU
基线或注明的现有 BF16/FP16 基线的容差内。关节误差（度）和夹爪误差的均值及 p99，
均不得大于两个不同种子的基线结果之间对应的差异。确定性策略的跨种子差异为零时，
改用绝对容差：关节误差均值和 p99 ≤ 0.01°，夹爪误差均值和 p99 ≤ 已注明量程的 0.1%。
权重量化、减少去噪步数、丢弃视觉 token 属于有损选项。

`--backend auto` 需要显式开启。选择器检查模型结构、保存的处理器布局、执行方式、编译设置
和受支持的 MLX 运行时。启动时显示资格证据来自哪个公开 checkpoint，或说明回退到 PyTorch
的原因。兼容的 SmolVLA checkpoint 在同步分块时使用 MLX。下文记录的机型和版本是实验条件，
不限制其他兼容的 Apple 芯片 Mac。
兼容的 X-VLA 和标准 ALOHA ACT 布局在同步分块时也使用编译后的 MLX。
`run --backend auto` 保留 ACT 的默认 ensemble 模式；要使用合格的 MLX 路径，请指定 `--mode chunk`。
仍可用 `--backend mlx` 显式选择。既有默认后端与显式后端的默认执行行为不变。
`bench --mode rtc` 使用固定的剩余动作块测量有引导的推理，不连接机器人。

SmolVLA 验证：`lerobot/smolvla_base` revision
`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`，Mac17,9（Apple M5 Pro）、macOS 27.0.1、
PyTorch 2.11.0、MLX 0.32.4、LeRobot 0.7.0。每块 50 步动作、去噪 10 步、三路
256 × 256 RGB，复用已保存的处理器，预热 5 次后计时 50 次。使用合成观测，未连接机器人。
CPU FP32 基线严格加载 checkpoint 的完整权重。除跨种子差异外，两项统计的绝对上限是
关节 0.01°、夹爪 0.01 原输出单位。

| 后端 | 中位数 / p90 ms | 关节均值 / p99 ° | 夹爪均值 / p99 |
|---|---:|---:|---:|
| Torch CPU FP32 | 1195.22 / 1411.72 | 基线 | 基线 |
| Torch CPU FP32，另一种子 | 1067.91 / 1095.52 | 0.31330 / 1.47535 | 0.41570 / 2.23537 |
| Torch MPS，BF16 底座 | 295.70 / 307.56 | 0.00387 / 0.01744 | 0.00478 / 0.03623 |
| MLX FP16，编译 | 52.76 / 53.24 | 0.00062 / 0.00258 | 0.00059 / 0.00315 |

### π0.5 录制观测实测

使用 `zz4321/so101_pi05@dbd6d5754bbf78318808aea1706ff23beeb8b663`，
回放 `zz4321/so101_cube@a34feb2f19718483fd6db7fac1d47927c84a5b4e` 中 5 个
episode（0、12、24、36、49）的 25 帧。各后端读取相同的记录状态、图像和任务原文，
并使用原始处理器。同 SmolVLA 环境，预热 5 次后计时 50 次，共回放这批帧两轮。
RTC 共用 FP32 在 4 帧前生成并移位 4 步的动作块，20 步引导区间和线性引导上限 10。

参考为 MPS FP32 eager，种子 941。中位数 / p90：分块 1057.14 / 1068.51 ms，
RTC 1093.42 / 1111.33 ms。BF16 使用 macbot 的编译路径，仅关闭 checkpoint 自带且
不兼容的 `max-autotune` 编译；MLX 使用编译后的 FP16。关节和夹爪单位见下方合成观测对比。

| 模式 | 后端 | 中位数 / p90 ms | 关节均值 / p99 ° | 夹爪均值 / p99 |
|---|---|---:|---:|---:|
| 分块 | Torch BF16，编译 | 330.30 / 335.94 | 0.401 / 3.961 | 0.111 / 0.790 |
| 分块 | MLX FP16，编译 | 222.81 / 229.14 | 0.074 / 0.886 | 0.017 / 0.158 |
| 分块 | Torch FP32，种子 942 | 1072.99 / 1087.34 | 11.425 / 104.201 | 4.034 / 23.821 |
| RTC | Torch BF16，编译 | 346.87 / 367.02 | 0.342 / 4.532 | 0.069 / 0.637 |
| RTC | MLX FP16，编译 | 226.67 / 277.24 | 0.056 / 0.841 | 0.009 / 0.113 |
| RTC | Torch FP32，种子 942 | 1087.41 / 1102.96 | 8.798 / 100.508 | 2.552 / 22.653 |

两个模式的关节 p99 都超过 0.50°；但 MLX 相对于同一 FP32 参考的四项误差都小于
BF16，且运行更快，因此兼容的 π0.5 checkpoint 在分块和 RTC 中由 `auto` 选编译后的 MLX。
下文的 0.082° / 0.416° 来自自用权重、不同录制片段和 BF16 参考。本次公开权重直接比较
MLX 与 BF16，分块为 0.393° / 3.965°，RTC 为 0.330° / 4.257°；以 FP32 为共同参考，
MLX 的四项误差都更小。两次实测的权重和参考口径不同。

### 公开 π0.5 checkpoint 的合成观测对照

`zz4321/so101_pi05` 固定 revision `dbd6d5754bbf78318808aea1706ff23beeb8b663`，
在 `zz4321/so101_cube` 上训练了 5,000 步。机器和软件版本与 SmolVLA 测试相同。
每行用三路 640 × 480 图像、50 步动作块、10 步去噪，预热 5 次后计时 50 次。
两条路径使用原始处理器和相同的合成观测。RTC 共享模型空间的剩余动作块、4 步延迟和
20 步引导区间，线性引导权重上限为 10。

checkpoint 保存的 Torch 采样器请求 `max-autotune`，在该 MPS 环境选择 `bias_addmm` 时触发
`NoValidChoicesError`。BF16 基线使用原有封装，仅在测量时用 `TORCHDYNAMO_DISABLE=1`
关闭编译。941 是同噪声基线种子，942 用于测量 Torch 跨种子差异。将反归一化后的关节位置误差
按各关节的 `跨度 × 360 / (4095 × 200)` 换算为 SO-101 连续目标的度数。从记录的从臂状态中
推得各关节编码器跨度为 2337、2431、2273、2182、3887 ticks。夹爪误差使用 0–100 原始单位。

| 模式 | 后端 | 中位数 / p90 ms | 关节均值 / p99 ° | 夹爪均值 / p99 |
|---|---|---:|---:|---:|
| 分块 | Torch MPS BF16，eager | 460.22 / 466.64 | 基线 | 基线 |
| 分块 | Torch MPS BF16，另一种子 | 460.32 / 467.00 | 20.926 / 102.412 | 3.268 / 10.543 |
| 分块 | MLX FP16，编译 | 207.25 / 210.26 | 0.878 / 4.550 | 0.150 / 0.520 |
| RTC | Torch MPS BF16，eager | 475.17 / 487.15 | 基线 | 基线 |
| RTC | Torch MPS BF16，另一种子 | 483.18 / 492.75 | 15.690 / 100.621 | 3.275 / 13.470 |
| RTC | MLX FP16，编译 | 225.43 / 231.51 | 0.604 / 4.315 | 0.109 / 0.511 |

两个模式的 MLX 延迟更低，四项误差也都低于对应的跨种子差异，但均超过本测试预设的
绝对上限：关节均值 / p99 为 0.10° / 0.50°，夹爪均值 / p99 为 0.10 / 0.50。
这轮初步测试以 BF16 为参考，未达准入。当前自动选择以本节前面的录制观测 FP32 对照为准。

### π0 录制观测实测

使用 `zz4321/so101_pi0_cube@ec54d7a73d8959a7640edd7dad2690f460e64af2`，回放
`zz4321/so101_stick@06f6856ea2a7dd43bfb72696a99e1087649a268b` 的 25 帧，来自
episode 0、7、14、21、29。回放协议同上，使用两路录制图像。checkpoint 自带
`max-autotune`，两个模式在 MPS 上均报 `NoValidChoicesError`；BF16 对照保留原精度配置，
关闭该编译器。由于通用封装使用 inference mode，RTC 参考直接调用 LeRobot 的引导采样器。

FP32 参考种子为 941，中位数 / p90 为分块 692.58 / 718.63 ms，RTC 704.17 / 718.71 ms。

| 模式 | 后端 | 中位数 / p90 ms | 关节均值 / p99 ° | 夹爪均值 / p99 |
|---|---|---:|---:|---:|
| 分块 | Torch BF16，eager | 368.02 / 374.15 | 0.151 / 0.800 | 0.050 / 0.840 |
| 分块 | MLX FP16，编译 | 160.28 / 163.59 | 0.027 / 0.135 | 0.012 / 0.159 |
| 分块 | Torch FP32，种子 942 | 695.33 / 738.39 | 6.162 / 40.905 | 2.401 / 22.928 |
| RTC | Torch BF16，eager | 383.45 / 388.20 | 0.093 / 0.832 | 0.039 / 0.704 |
| RTC | MLX FP16，编译 | 169.65 / 175.58 | 0.015 / 0.101 | 0.007 / 0.114 |
| RTC | Torch FP32，种子 942 | 727.07 / 749.78 | 4.461 / 40.704 | 1.809 / 23.799 |

两个模式都通过绝对上限和 BF16 对照条款，兼容 checkpoint 可由 `auto` 选择编译后的 MLX。

用 `python -m tools.accel.recorded_observations --help` 从数据分片提取观测，
再用 `python -m tools.accel.compare_backend --help` 查看 `--observations` 等回放参数。
`--pi-fp32` 用于 Torch eager 参考，`--save-rtc-leftovers` 和 `--rtc-leftovers` 用于共用历史动作块。

### 公开 π0 checkpoint 的合成观测对照

`zz4321/so101_pi0_cube` 固定 revision `ec54d7a73d8959a7640edd7dad2690f460e64af2`，
在 `zz4321/so101_stick` 上训练了 5,000 步。使用两路 640 × 480 图像，计时、噪声、
RTC 和关节角换算方法同上。Torch 使用 checkpoint 的 BF16 配置并在测量时关闭 Dynamo。
由于 macbot 的通用封装使用 inference mode，RTC 参考直接调用 LeRobot 的引导采样器。
MLX 使用编译后的 FP16 路径。

| 模式 | 后端 | 中位数 / p90 ms | 关节均值 / p99 ° | 夹爪均值 / p99 |
|---|---|---:|---:|---:|
| 分块 | Torch MPS BF16，eager | 362.64 / 365.83 | 基线 | 基线 |
| 分块 | Torch MPS BF16，另一种子 | 362.03 / 366.54 | 29.285 / 152.684 | 8.909 / 37.207 |
| 分块 | MLX FP16，编译 | 156.08 / 158.34 | 0.378 / 2.795 | 0.082 / 0.554 |
| RTC | Torch MPS BF16，eager | 371.93 / 377.15 | 基线 | 基线 |
| RTC | Torch MPS BF16，另一种子 | 372.08 / 380.31 | 23.126 / 146.821 | 7.516 / 35.967 |
| RTC | MLX FP16，编译 | 156.74 / 161.46 | 0.270 / 2.283 | 0.068 / 0.466 |

两个模式的 MLX 延迟均更低，误差也都低于跨种子差异。两个模式的关节均值及 p99 超过了
绝对上限，分块的夹爪 p99 也超限。这轮测试先于录制观测 FP32 对照。

## X-VLA 和 ACT 录制观测实测

共同环境：Mac17,9（Apple M5 Pro）、macOS 27.0.1、PyTorch 2.11.0、MLX 0.32.4、
LeRobot 0.7.0。每行均使用保存的处理器、来自五个训练 episode 的 25 帧录制观测，
预热 5 次后测 50 次同步调用，从转换 RGB 和状态计到动作回读。CPU 基线用 4 线程。
三组均为训练权重离线回放。

### X-VLA

使用 `whosricky/xvla-so101-pick-red-cube-2cam@7a007f64dd5910dd5cde76c8d31d9d7f7e6c0778`，
回放 `whosricky/pick-red-cube-v3@f77f31614b10b9cf4ceb4c5f4c4edb503b12b74c`
的 episode 0、54、108、147、181。两路录制 RGB 缩放至 checkpoint 的 256 × 256 输入。
每块 30 步、去噪 10 步；MLX 批量处理 DaViT 视角，经 Florence-2 BART 编码器后，
把投影后的上下文复用于 soft prompt 动作头。动作头保留空视角槽位，采样保持原有噪声插值。
LeRobot 采样器没有 RTC 引导。五个从臂编码器跨度（2116、2343、2203、2294、4095
ticks）由同一 episode 内录制状态的增量网格恢复；乘 `360 / (4095 × 200)` 转成角度比例。
夹爪误差沿用 0–100 开合单位。

| 后端 | 证据 | 中位数 / p90 ms | 关节均值 / p99 ° | 夹爪均值 / p99 |
|---|---|---:|---:|---:|
| Torch CPU FP32，种子 941 | 录制回放基线 | 4875.08 / 5619.00 | 0 / 0 | 0 / 0 |
| Torch CPU FP32，种子 942 | 录制回放跨种子 | 4786.63 / 5063.67 | 0.13363 / 0.81821 | 0.34578 / 7.50583 |
| Torch MPS BF16，种子 941 | 已验证的默认路径 | 178.19 / 205.14 | 0.13311 / 0.65928 | 0.06051 / 0.32293 |
| MLX FP16，编译，种子 941 | 已验证，满足自动选择门槛 | 137.86 / 140.33 | 0.01014 / 0.04526 | 0.00556 / 0.03549 |

MLX 的四项误差均小于跨种子差异，满足绝对上限（关节 0.10° / 0.50°、夹爪
0.10 / 0.50），也比 MPS BF16 更准、更快。`auto` 限于已测的架构、处理器、
两相机输入和启用编译的分块配置。

### ACT

MLX 在 FP32 将冻结 BN 折叠进 ResNet-18 卷积，Transformer 使用保存的位置参数和
LeRobot 推理规定的零 latent。ACT 为确定性模型，输出完整 100 步动作块，无去噪。
完整 FP32 需要 `MLX_ENABLE_TF32=0`，macbot 只在策略进程加载 ACT MLX 时设置。
比较不同 MLX 精度时使用独立进程。

公开权重为 `lerobot/act_aloha_sim_transfer_cube_human@ba73b2766f1371cdc133ca4efb97eb090d744625`。
通过 `lerobot.processor.migrate_policy_normalization` 把保存的归一化统计迁移到 LeRobot 0.7
处理器，并确认统计与模型权重一致。回放
`lerobot/aloha_sim_transfer_cube_human@6a43d500f101255823a9d2b9dc244eeb01a2cd31`
的 episode 0、12、24、36、49。单路 640 × 480，14 维动作，12 个关节从弧度转成角度，
两个夹爪使用标称 0–1 的归一化位置单位。

| 权重与后端 | 证据 | 中位数 / p90 ms | 关节均值 / p99 ° | 夹爪均值 / p99 |
|---|---|---:|---:|---:|
| 公开 ACT，Torch CPU FP32 | 录制回放基线 | 56.25 / 63.35 | 0 / 0 | 0 / 0 |
| 公开 ACT，Torch MPS FP32 | 已验证的默认路径 | 16.42 / 17.20 | 3.39e-6 / 2.05e-5 | 1.00e-7 / 5.07e-7 |
| 公开 ACT，MLX FP32，编译，关闭 TF32 | 已验证，满足自动选择门槛 | 8.32 / 8.57 | 4.18e-6 / 2.73e-5 | 1.24e-7 / 9.54e-7 |
| 自训 ACT，Torch CPU FP32 | 录制回放基线 | 71.98 / 76.33 | 0 / 0 | 0 / 0 |
| 自训 ACT，Torch MPS FP32 | 已验证的默认路径 | 22.80 / 23.47 | 6.93e-6 / 4.58e-5 | 1.76e-7 / 1.43e-6 |
| 自训 ACT，MLX FP32，编译，关闭 TF32 | 已验证，通过保真门槛 | 12.28 / 12.58 | 7.29e-6 / 4.58e-5 | 1.71e-7 / 1.28e-6 |
| 自训 ACT，MLX FP32，编译，开启 TF32 | 已验证，有损实验 | 9.95 / 10.11 | 0.02227 / 0.08749 | 0.000399 / 0.004394 |
| 自训 ACT，MLX FP16，编译 | 已验证，有损实验 | 8.38 / 8.55 | 0.00706 / 0.03525 | 0.000153 / 0.001858 |

自训权重使用两路录制画面和 0–1 的二值闭合分数。两组关闭 TF32 的 FP32 结果均通过
关节 0.01° 和夹爪 0.001 的门槛。TF32 与 FP16 实验超限，可在 `tools.accel.compare_backend`
中用 `--act-mlx-tf32` 或 `--act-mlx-fp16` 复测。
公开 ALOHA 布局通过 `auto` 验证；自训布局仍可显式选择 `--backend mlx`。
这些数据不衡量任务成功率。

## PyTorch MPS 实验选项

ACT 和 Diffusion 可以用 `--torch-channels-last` 改变二维卷积权重的内存布局，
用 `--torch-autocast fp16` 或 `bf16` 改变推理精度（可能有损）。
`--torch-single-readback` 可省掉 ACT 显式的设备同步；`--torch-compile` 用于试验编译
Diffusion 的去噪网络。当前测试环境无法生成 ACT 的 Metal 内核，已禁用该组合。
这些选项必须显式开启，`auto` 不会使用。Diffusion 的动作队列和采样器保持不变。
用 `python -m tools.accel.compare_backend --help` 查看离线对比用法，无需打开机器人或相机。

## 离线试用

在 Apple 芯片上安装可选依赖，再用自己的 checkpoint 和相机尺寸配置跑基准：

```sh
python -m pip install '.[mlx]'
macbot --config "$CONFIG" bench --policy "$CHECKPOINT" --backend mlx -n 30
macbot --config "$CONFIG" bench --policy "$CHECKPOINT" --backend torch --profile -n 30
```

`bench` 使用生成的空白 RGB 数组，不连接相机或机器人。分段计时在独立的同步测量轮次进行，不能直接相加当作端到端延迟。`--no-compile` 关闭后端图编译。部署时，可在 `macbot run` 加 `--backend mlx`，或在 YAML 中设置 `policy: {backend: mlx}`。MLX 下 `--mode auto` 选择无引导的异步执行；显式 `--mode rtc` 使用所选后端执行有引导 RTC。

`--mlx-bits 8` 和 `--mlx-bits 4` 是实验性权重量化，可配合 `--mlx-quantize prefix`、`expert` 或 `both`。减少 `--steps` 会改变流匹配积分。使用这些选项前，请先在自己的 checkpoint 上评测。

## 复现延迟与保真度

在 macbot 源码目录中，使用 LeRobot 0.7.0 环境运行以下命令。将 `DATA` 指向原始 `episode_*.npz` 目录，`EVAL` 指向仓库外可写目录。观测归档、动作数组和中间层输出均应私下保存。

```sh
mkdir -p "$EVAL"
python -m tools.accel.eval_fidelity prepare --data "$DATA" --out "$EVAL/observations.npz"
python -m tools.accel.eval_fidelity run --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/bf16.json" --profile
python -m tools.accel.eval_fidelity run --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/seed2.json" \
  --reference "$EVAL/bf16.npy" --seed 942
python -m tools.accel.eval_fidelity run --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/fp32.json" \
  --reference "$EVAL/bf16.npy" --dtype fp32
python -m tools.accel.bench_mlx --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/torch_stages.npz" --capture-torch
python -m tools.accel.bench_mlx --checkpoint "$CHECKPOINT" \
  --observations "$EVAL/observations.npz" --out "$EVAL/mlx.json" \
  --trace "$EVAL/torch_stages.npz" --reference "$EVAL/bf16.npy" \
  --trim --compile --precompute-adarms --fused-rope --pack-linears --precision fp16 --profile
```

同一时间只运行一个基准进程。默认保真集包含来自 10 个通过质量检查的 episode 的 50 条观测。延迟测量预热五次后计时 30 次，使用常规随机噪声；保真对照则为每条观测在所有运行中复用相同的 FP32 初始噪声（种子 941，跨种子对照除外）。机械臂误差在反归一化后以度计，二值夹爪单独报告反归一化后的分数。两类指标都统计完整动作块和前 10 步的绝对误差均值与 p99。离线回放的一致性不能证明真实机器人任务成功率。

MLX 计时前，先比对 SigLIP 投影特征、各层有效位置的前缀 K/V、`t=1` 的速度和最终动作。下文的基线分段和对照结果都来自真实录制观测；命令行空白输入计时独立于这组数据。

## M5 Pro 实测

环境为 20 核 GPU、48 GiB 统一内存的 Apple M5 Pro，macOS 27.0.1、PyTorch 2.11.0 和 MLX 0.32.4。每次只运行一个 GPU 基准进程。桌面仍有后台活动，因此不把不同批次间的小幅变化全归因于配置差异。

同步计时中，MPS 的双路 SigLIP 合批耗时 36.45 ms，逐路测得 19.25 和 19.23 ms；Gemma 前缀和 KV 为 126.41 ms，每步专家去噪 13.54–13.87 ms。RGB 转换、主机到设备传输和 resize 分别为 0.67、1.27 和 0.56 ms，动作反归一化和原始动作读回分别为 0.54 和 0.22 ms。这些是独立测量轮次或各同步阶段的中位数。

删掉文本 padding 后，前缀的密集投影仍约有 2.2 TFLOP，其中仅视觉 token 就占约 2.03 TFLOP。双路 SigLIP 的投影约为 0.42 TFLOP；专家每步约 33 GFLOP 的投影，需要访问约 0.63 GB 密集投影权重。这些估算未计注意力、归一化和小投影。前缀与视觉算术强度较高；专家每字节的运算较少，对数据移动和调度更敏感。

FP32 MPS 对照用于校准数值误差，换初始噪声的对照用于校准策略自身的波动。本 checkpoint 的「实际保真」判据是：保留所有视觉 token 和 10 步去噪，不量化权重，且八项动作误差都低于各自跨种子差异的一半。这是离线实测判据，不保证任意轨迹完全等价。

| 配置 | 中位数 / p90（ms） | RSS 峰值 / GPU（GiB） | 臂全段（°） | 臂前 10 步（°） | 夹爪全段 | 夹爪前 10 步 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| MPS BF16，初次分段测量 | 297.71 / 311.91 | 12.38 / 9.16 | 基线 | 基线 | 基线 | 基线 |
| MPS BF16，同噪声重复测量 | 276.83 / 291.30 | 15.01 / 9.17 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| MPS BF16，换初始噪声 | 279.03 / 284.38 | 15.19 / 9.17 | 0.626 / 8.037 | 0.274 / 1.179 | 0.0060 / 0.0071 | 0.0057 / 0.0064 |
| MPS FP32，eager 对照 | 923.30 / 980.37 | 15.07 / 21.16 | 0.081 / 0.431 | 0.063 / 0.265 | 0.0007 / 0.0024 | 0.0007 / 0.0022 |
| MLX FP16，部署配置 | **168.53 / 188.62** | 8.44 / 6.89 | 0.082 / 0.416 | 0.061 / 0.263 | 0.0007 / 0.0023 | 0.0007 / 0.0022 |
| MLX FP16，全视觉 token 加 `--steps 2`，有损 | 122.40 / 131.40 | 8.49 / 6.89 | 0.447 / 2.466 | 0.316 / 1.373 | 0.0050 / 0.0273 | 0.0056 / 0.0122 |
| MLX FP16，视觉池化加两步，有损 | **70.96 / 71.81** | 8.42 / 6.69 | 4.173 / 26.491 | 1.807 / 7.609 | 0.0304 / 0.9806 | 0.0347 / 0.9848 |

误差栏均为均值 / p99。RSS 是含加载和编译在内的进程生命周期峰值；GPU 一列为 MPS 驱动分配的采样最大值或 MLX 在计时期间的分配峰值。统一内存下两者存在重叠。

**在保真的前提下，没有达到 100 ms。** 部署配置的八项误差分别不超过对应跨种子差异的 34%，中位数比重测的 MPS 基线低 39%。同步分段结果为视觉 24.32 ms、前缀与 KV 98.58 ms、专家每步 5.20–5.65 ms。少做去噪无法解决前缀耗时。删文本 padding、MLX 编译和预计算 AdaRMS 有效。8 位仿射量化降低了 GPU 分配量，但速度更慢；4 位前缀量化明显增大了 KV 和动作误差。表中低于 100 ms 的实验用 2×2 平均池化合并视觉投影 token，并把 Euler 去噪改成两步。此组合仅在离线工具中提供（`--pool-images --steps 2`）；以文档上方 FP16 命令为基础，去掉 `--pack-linears` 即可复现。

要在同等保真度下继续降低延迟，需要显著提高前缀投影和注意力的效率，或改用前缀更小的模型。若要进一步减少专家计算，也可以从设计或蒸馏时就针对较少的去噪步数。本次未训练或测试这些方案。

## 第二轮：延迟花在哪里

默认十步 MLX 在全部 50 条观测上的输出与 v0.3.0 逐位一致。本轮基线为 **164.00/167.32 ms**（中位数/p90），集成后的复测为 **160.74/162.85 ms**。接交流电，桌面有后台活动，一次只运行一个基准；至少预热 5 次，再计时 30 次。两次重测的差值不能归因于优化。

对真实尺寸的合成矩阵做 Metal 捕获，确认 MLX 0.32.4 的 gate/up 已分派到 `steel_gemm_fused_nax`，down 已分派到 `steel_gemm_splitk_nax`。回放中前缀长 553–558 个 token。以 557 行为例，FP16 gate/up（2048→32768）达到 **27.7 TFLOP/s**，down（16384→2048）达到 **21.1 TFLOP/s**，同机 4096 阶 FP16 方阵为 **30.0 TFLOP/s**。补到 576 行后，gate/up 从 2.696 变为 2.690 ms，down 从 1.771 变为 1.872 ms。填充后的吞吐更高，不一定节省时间。可用 `python -m tools.accel.bench_mlx_shapes --out "$EVAL/shapes.json"` 复测不同精度、布局和对齐尺寸。

初始同步分段为：预处理 1.93 ms、双相机视觉 23.60 ms、前缀与 KV 92.54 ms、专家每步 5.21–5.52 ms。另一次整机较慢时，输入已就绪，只执行双相机视觉直到全部 KV 就绪，中间无同步、不做去噪，实测为 **146.51/174.19 ms**。这些是当前实现的耗时，不是硬件理论下限。前缀补齐、改变权重布局、编译整个去噪循环、自定义 AdaRMS 内核和预切分调制向量均未取得可重复、足以替换默认配置的端到端收益。

| 同噪声回放配置 | 中位数/p90 ms | 机械臂均值/p99 ° | 八项误差相对种子波动的最大比例 |
| :-- | --: | --: | --: |
| 集成后的默认十步 | 160.74/162.85 | 0.082/0.416 | 34.0% |
| 均匀 Euler，三次求值 | 159.42/181.83 | 0.320/1.587 | 134.4% |
| 二次时间网格 Euler，三次求值 | 132.94/135.83 | 0.145/1.350 | 50.7% |
| 每路保留一半视觉块，均匀三次求值 | 95.17/99.24 | 0.818/5.788 | 933.6% |
| 每路保留四分之一，均匀三次求值 | 82.62/88.07 | 1.530/11.938 | 13847.9% |

短步数和视觉裁剪都属于**有损**配置。`bench` 和 `run` 可用 `--backend mlx --steps 3 --mlx-time-grid quadratic`，让求值时刻更集中在接近干净动作的一端。应在相同求值次数下比较误差；各轮桌面负载不同，延迟差不能单独归因于求解器。离线复现时，在上面的命令中加 `--steps 3 --solver quadratic`；还可比较 `ab2` 和 `midpoint`（后者要求偶数次求值）。视觉裁剪只在离线工具中提供：`--steps 3 --keep-visual-fraction 0.5` 在第二层之后按注意力为每路相机保留视觉块，保留全部文本和原始绝对位置。保留一半时，全块夹爪 p99 误差为 0.06643，种子波动为 0.00712。

显式 MLX RTC 在三条观测上单独对照，使用相同噪声和模型空间的剩余动作，延迟 4、引导范围 16、最大引导权重 10。相对 PyTorch guided 输出，机械臂均值/p99 为 **0.071/0.350°**；MLX guided 调用为 **163.17/175.12 ms**。LeRobot 0.7.0 在计算速度之后才打开输入梯度，因此参考实现的校正量等于加权残差。对整个去噪器求导得到 0.183/1.670°，部署采用前一种语义。未做真机回放。

## 大型 VLA 状态

| 策略 | checkpoint 与 revision | 后端 | 精度 / token | 中位数 / p90 ms | 保真 | 证据等级 |
|---|---|---|---|---:|---|---|
| π0 | `lerobot/pi0_libero_finetuned_v044@45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f` | MLX | fp16，流匹配 10 步 | 未测 | 已加载训练权重；tokenizer 与 fp32 对照待完成 | runs |
| π0-FAST | `lerobot/pi0fast-libero@840f4b503f4c09110421c33c810a85b6684fd658` | MLX | bf16，greedy，最多 256 token | 885.22 / 1281.60 | `Action:` 前缀后 token 序列与 PyTorch 分歧 | runs |
| GR00T | `nvidia/gr00t17-lerobot-libero_10-640@5ee08ab09fac5c5ef2388a14c882ea825ac861db` | MLX | 未运行 | 未测 | 未启动 fp16 内存与下载门槛 | 未验证 |

π0-FAST 行使用 Mac17,9、Apple M5 Pro、48 GB、macOS 27.0.1、PyTorch 2.15.0、MLX 0.32.4、LeRobot 0.7.0，预热 5 次后同步计时 50 次。根据这些结果，`auto` 不选择 π0、π0-FAST 或 GR00T。

## 实现参考

移植以 LeRobot 的 `pi05` 为准，使用 MLX 的融合注意力、归一化和量化矩阵乘 API。调研参考了 `tokimoa/pi0-mlx` 的模型卡和实现（Apache-2.0）、`mlx-smolvla`（Apache-2.0）、`mlx-lm`（MIT）和 `mlx-vlm` 的 PaliGemma/SigLIP（MIT）。未引入外部源码或转换后的权重。
