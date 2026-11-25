# 策略

[English](policies.md)

macbot 运行 LeRobot 0.7.0 的权重。所有策略都能用 PyTorch 在 Apple GPU（MPS）上跑，其中一部分还有 MLX 后端，通常更快。本页列出目前能用的策略，以及怎么接入你自己的权重。MLX 后端怎样验证、完整的测量数据见[加速](acceleration.zh.md)。

## 支持一览

延迟是在 Apple M5 Pro 上调用一次策略的中位数，从观测到动作块。两个数字分别是 PyTorch MPS 和 MLX。

| 策略 | MLX | `auto` 用 MLX | MPS → MLX（ms） | 验证所用权重 |
|---|---|---|---:|---|
| ACT | 有 | 分块 | 16 → 8 | ALOHA，公开 |
| π0.5 | 有 | 分块、RTC | 270 → 165 | SO-101，公开 |
| π0 | 有 | 分块、RTC | 368 → 160 | SO-101，公开 |
| SmolVLA | 有 | 分块 | 296 → 53 | 官方基础模型 |
| X-VLA | 有 | 分块 | 178 → 138 | SO-101，公开 |
| VQ-BeT | 有 | 否 | 7.7 → 4.2 | PushT，公开 |
| Diffusion | 有，但更慢 | 否 | 819 → 1960 | PushT，公开 |
| π0-FAST | 有 | 否 | — → 885 | LIBERO，离线 |
| TD-MPC | 有 | 否 | — | 随机小权重 |

只有当 MLX 在这种架构的一个训练过的权重上和 PyTorch 对上了，`auto` 才会在表中列出的模式下选 MLX。其他情况都留在 PyTorch，你仍然可以用 `--backend mlx` 手动指定 MLX。

- **VQ-BeT** 的 MLX 版和 PyTorch 都在 CPU 上跑时，动作只差 0.0002 像素。在 GPU 上，卷积的 float32 小差异会让 512 像素量程的动作平均差 0.3 像素左右，所以 `auto` 不选它。
- **Diffusion** 的 MLX 版结果正确，但在 100 步的公开权重上比 PyTorch 慢，请用 PyTorch。
- **π0-FAST** 的 MLX 版能解码出动作，但动作 token 和 PyTorch 的不一样。测试用的权重输出末端位姿动作，所以只在离线脚本里跑过，没有上机械臂。PyTorch 的延迟没有测。
- **TD-MPC** 没有 LeRobot 0.7 格式的公开权重，只测了数据通路。

Gaussian actor 和 Multi-Task DiT 只能用 PyTorch，用随机小权重测过。GR00T、WALL-X 和 WALL-OSS 需要的权重和主干模型文件都还没有试过。其余 LeRobot 类型（`cosmos3_edge`、`cosmos3_nano`、`dm05`、`eo1`、`evo1`、`fastwam`、`fineart_vla`、`flux3`、`g05`、`lawam`、`lingbot_va`、`lingbot_vla_v2`、`longwam`、`molmoact2`、`vla_jepa`）能通过 LeRobot 的工厂函数加载，但还没有跑过它们的任何权重。`macbot policies` 会按你本机的安装情况打印同样的表。

一个权重上的结果不能直接推到同类型的所有权重。正式使用前，先用 `macbot bench` 测一下你自己的权重。

## 接入你的权重

你需要一个 LeRobot 的 `pretrained_model` 目录：`config.json`、权重文件、`policy_preprocessor.json`、`policy_postprocessor.json`，以及它们引用的状态文件。PEFT 适配器要先合并进基础策略再导出。按模型需要安装 LeRobot 的附加依赖，比如 `pip install 'lerobot[smolvla]==0.7.0'`。只使用你信任的权重和依赖。

1. **确认动作。** 权重输出的必须是绝对关节位置，顺序和单位都要和你的机械臂一致。macbot 会拒绝已知的末端位姿格式，也会拒绝和机械臂对不上的状态或动作维度，但维度一致并不代表含义一致。确认无误后，在配置里写上 `policy.action_space: joint_position`。

2. **对应相机。** 在配置里给每台相机起名，并对应到权重里的图像键。`save_size` 的顺序是 `[宽, 高]`。名字本来就一致时可以不写 `camera_map`。

    ```yaml
    robot: {type: lerobot, follower: {type: so101_follower}}
    task: {name: demo, instruction: "put the cube in the box"}
    cameras:
      top: {type: uvc, save_size: [256, 256]}
      wrist: {type: uvc, save_size: [256, 256]}
    policy:
      camera_map: {top: camera1, wrist: camera2}
    ```

3. **先测速，再不接机械臂试跑。**

    ```bash
    CKPT=path/to/pretrained_model
    macbot --config robot.yaml bench --policy "$CKPT" --backend auto --profile
    macbot --config robot.yaml run --policy "$CKPT" --backend auto --fake --autostart 3
    ```

    `bench` 输入的是空白图像，不会打开相机或机械臂。`--profile` 会多跑一遍，分别计时预处理、推理和后处理。`--fake` 用假硬件把整个控制循环跑一遍。

`--policy` 也可以写 Hub 上的模型 ID，比如 `owner/model@revision`。

### 部分策略的注意事项

| 策略 | 需要知道的 |
|---|---|
| ACT | 默认用时间集成执行，这时留在 PyTorch。加 `--mode chunk` 后 `auto` 才会用 MLX。 |
| π0.5 | 至少需要 32 GB 内存。两个后端都支持 `--mode rtc`。 |
| π0 | 只有 MLX 后端支持 `--mode rtc`。 |
| π0、π0-FAST | 使用需要申请权限的 PaliGemma 分词器。如果你的账号没有权限，把 `MACBOT_PALIGEMMA_TOKENIZER` 设为本地 PaliGemma 的 `tokenizer.model`。 |
| SmolVLA | `lerobot/smolvla_base` 需要三台 256 × 256 的相机，分别对应 `camera1`、`camera2`、`camera3`。 |
| X-VLA | 测过的权重用两台相机，对应 `image` 和 `image2`。不支持 RTC 和 `ee6d` 基础模型。 |
| VQ-BeT、TD-MPC | 只接一台相机。TD-MPC 的图像键叫 `observation.image`，要写完整的键名。VQ-BeT 需要训练好的残差 VQ 才能输出有意义的动作。 |

ACT 和 π0.5 用的是 LeRobot 原有的封装。其他类型走一个通用适配器，调用 LeRobot 的 `select_action` 或分块推理；加 `--policy-impl generic` 时，ACT 和 π0.5 也走这个适配器。通用适配器不支持异步执行、时间集成和 RTC。

## 模型在 MPS 上出错时

先试 `--device cpu`，看问题是不是 MPS 特有的。如果想让 PyTorch 把不支持的算子放到 CPU 上跑，在启动 macbot **之前**设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，并留意算子警告。这个回退解决不了缺模型文件、形状不匹配，或者模型内部的 float64 运算。本文档里的测量都没有用到这个回退。

??? note "通用适配器：MPS 与 CPU 对比"

    MacBook Pro，Apple M5 Pro（48 GiB），macOS 27.0.1，PyTorch 2.11.0，LeRobot 0.7.0，Python 3.12。每次计时都包括保存的处理器和拷回 CPU。预热 5 次，之后小策略调用 30 次，SmolVLA 和 X-VLA 15 次，π0 5 次，输入是固定的合成图像和状态。CPU 用 float32、4 个线程。计算动作差异时两边用相同的随机数。逐步执行的计时既包括直接从动作队列取的调用，也包括重新规划的调用。

    | 策略 / 权重 | 执行方式 | MPS 中位数 / p90（ms） | CPU 中位数 / p90（ms） | 最大动作差 | 峰值 RSS（GiB） |
    |---|---|---:|---:|---:|---:|
    | ACT，小模型 | 3 步分块 | 7.21 / 7.93 | 6.02 / 6.16 | 1.31e-6 | 0.66 |
    | Diffusion，小模型 DDIM | 每 3 次调用重新规划 | 1.44 / 22.24 | 0.11 / 12.43 | 3.10e-6 | 0.80 |
    | VQ-BeT，小模型 | 每 3 次调用重新规划 | 1.18 / 7.26 | 0.07 / 4.71 | 2.38e-7 | 0.69 |
    | TD-MPC，小模型 | 每 2 次调用重新规划 | 5.30 / 9.42 | 1.06 / 2.23 | 2.38e-7 | 0.64 |
    | Gaussian actor，小模型 | 连续动作 | 3.70 / 3.97 | 0.60 / 0.67 | 4.77e-7 | 0.62 |
    | Multi-Task DiT，小模型 DDIM | 每 3 次调用重新规划 | 1.74 / 7.63 | 0.12 / 1.47 | 4.77e-7 | 0.64 |
    | π0，随机 2B 前缀，去噪 2 步 | 3 步分块 | 490.01 / 492.80 | 2150.15 / 2219.61 | 2.68e-6 | 16.74 |
    | SmolVLA，官方基础权重 | 50 步分块 | 297.80 / 309.43 | 1126.30 / 1164.04 | 0.0316 | 3.38 |
    | X-VLA，SO-101 双相机 | 30 步分块 | 181.89 / 205.54 | 6226.21 / 7688.11 | 1.22 | 4.93 |

    小模型和随机权重只用来检查数据通路，不代表任务表现。SmolVLA 和 X-VLA 在 MPS 上保留了权重自带的 bf16 设置，所以和 float32 的差异更大。差异的单位是各权重自己的动作单位。

    在源码目录里重复小模型对比：

    ```bash
    python -m tools.tiny_policies diffusion /tmp/diffusion
    # robot.yaml 里配置两台 64 × 64 的相机，命名为 top 和 wrist。
    for device in cpu mps; do
      PYTORCH_ENABLE_MPS_FALLBACK=0 python -m tools.policy_sweep /tmp/diffusion \
        --config robot.yaml --device "$device" --out "/tmp/diffusion-$device.npy"
    done
    python -c 'import numpy as n; print(abs(n.load("/tmp/diffusion-cpu.npy")-n.load("/tmp/diffusion-mps.npy")).max())'
    ```
