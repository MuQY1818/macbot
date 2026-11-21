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

## 实现参考

移植以 LeRobot 的 `pi05` 为准，使用 MLX 的融合注意力、归一化和量化矩阵乘 API。调研参考了 `tokimoa/pi0-mlx` 的模型卡和实现（Apache-2.0）、`mlx-smolvla`（Apache-2.0）、`mlx-lm`（MIT）和 `mlx-vlm` 的 PaliGemma/SigLIP（MIT）。未引入外部源码或转换后的权重。
