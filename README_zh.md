<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/logo.svg" alt="macbot" height="72">
  </picture>
</p>

<p align="center"><b>在 MacBook 上跑 LeRobot。</b><br>
把已有的 LeRobot 机械臂、校准、数据集和 checkpoint 搬到 MacBook，在 Apple 芯片上加速推理。</p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20silicon-18171F">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-F2701A">
  <img alt="LeRobot" src="https://img.shields.io/badge/LeRobot-0.7.0-1F9D63">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-6B6A75">
  <a href="https://github.com/MuQY1818/macbot/actions/workflows/test.yml"><img alt="tests" src="https://github.com/MuQY1818/macbot/actions/workflows/test.yml/badge.svg"></a>
</p>

<p align="center"><a href="https://github.com/MuQY1818/macbot/blob/main/README.md">English</a> · <b>中文</b></p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/scene-dark.svg">
    <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/scene.svg" alt="在机械臂上采集，导出 LeRobot 数据集，在 GPU 上训练，把权重拿回 MacBook 上部署" width="100%">
  </picture>
</p>

## 从这里开始

已经在 Linux 上用 LeRobot？按[迁移指南](docs/migrate_zh.md)复用校准并导入原命令。
从空白环境开始，见[快速开始](docs/quickstart_zh.md)。
使用 Apple 芯片 Mac 和 Python 3.12 或更高版本。数据集和 checkpoint 保持 LeRobot 格式，可在两台机器间直接使用。

```bash
pip install macbot
macbot doctor
macbot init
# 已有 LeRobot 命令？改用 macbot init --from-lerobot "$LEROBOT_COMMAND"。
macbot collect
macbot health
macbot export --out ./dataset
# 在 GPU 机器上用 lerobot-train 训练，再把 checkpoint 拿回来。
macbot run --policy ./pretrained_model       # 也可用 owner/model@revision
macbot replay
```

reBot B601-RS 已做真机验证。[SO-101 桌面流程](docs/simulation_zh.md)已在仿真验证：
训练 5,000 步的 ACT 完整任务成功 **0/50**（PyTorch MPS FP32，30 步动作块）。
各配置的验证范围见[机械臂支持](docs/robots_zh.md)。运行 π0.5 建议使用至少 32 GB 内存的 Apple 芯片 Mac。
已测 checkpoint 见[策略支持](docs/policies_zh.md)。

以下数据在 MacBook Pro（M5 Pro，48 GB）上实测，为每个动作块的大致耗时：

| 策略 | 后端 | 执行 | 延迟 |
|---|---|---|---:|
| π0.5 | PyTorch MPS | 分块 50 | 270 ms |
| π0.5 | MLX | 分块 50 | **165 ms** |
| π0.5 | PyTorch MPS | RTC 50 | 350 ms |
| π0.5 | MLX | RTC 50 | **170 ms** |
| SmolVLA | PyTorch MPS | 分块 50 | 296 ms |
| SmolVLA | MLX | 分块 50 | **53 ms** |
| X-VLA | PyTorch MPS | 分块 30 | 235 ms |
| ACT | PyTorch MPS | 分块 100 | 25 ms |

π0.5 使用我们在 SO-101 同类机械臂上训练的微调权重和录制观测，去噪 10 步（编译后的 PyTorch BF16 或 MLX FP16）。RTC 指 Real-Time Chunking 引导。SmolVLA 使用公开的 `lerobot/smolvla_base`，去噪 10 步（MPS BF16 底座或编译后的 MLX FP16）。X-VLA 使用公开的 SO-101 BF16 微调权重；ACT 使用我们训练的权重，在 MPS 上以 FP32 运行。测法和更多结果见[推理基准](https://github.com/MuQY1818/macbot/blob/main/docs/acceleration_zh.md)和[策略支持](https://github.com/MuQY1818/macbot/blob/main/docs/policies_zh.md)。

| 数据采集 | 实测 |
|---|---:|
| 遥操作控制频率 | 60 Hz |
| 录制采样频率 | 30 fps |
| 采样时刻的画面延迟 p95 | ≤ 40 ms |

<p align="center"><img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/collect_zh.jpg" alt="macbot 采集控制台" width="100%"></p>

| 跑策略 | 回放试验 |
|---|---|
| <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/run_zh.jpg" alt="跑策略"> | <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/replay_zh.jpg" alt="回放"> |

- `init` 引导选择机械臂和相机，并检查已有的 LeRobot 校准。
- `collect` 提供相机预览和逐条检查；没有机械臂可以试 `collect --fake`。
- `health --dataset` 和 `replay --dataset` 只读查看本地或 Hub 上的 LeRobot v3.0 数据集。
- `run` 加载 LeRobot checkpoint，并记录每次试验的画面和时序。
- `replay` 同步回放试验画面、关节曲线和动作块，详见[工作流程](docs/guide_zh.md)。

## 在 macOS 上跑起来要解决的问题

- **sudo 下的 CAN：**macbot 预载 PCBUSB，并把生成的录制文件交回调用用户。
- **相机识别：**用 AVFoundation unique ID 固定 UVC 相机角色，避开 iPhone 连续互通相机。
- **π0.5 推理：**macbot 支持 PyTorch MPS 编译和可选的 MLX 后端，数据见上面的基准。

## 参与贡献

开发与真机报告见[贡献指南](CONTRIBUTING_zh.md)，代码布局见[架构](docs/architecture_zh.md)，
后续工作见[路线图](ROADMAP_zh.md)。

## 安全

macbot 会驱动真实的机械臂。所有控制台启动时都处于暂停或待机状态；会限制关节速度和范围、限制夹爪夹紧力；相机断流时自动暂停；退出时先慢速回零再让电机失能。请保持工作区域无人、手边留着电源开关；新策略先用 `--speed 0.3` 跑。

## 致谢

基于 [LeRobot](https://github.com/huggingface/lerobot)。π0.5 和图像增强配方来自 [openpi](https://github.com/Physical-Intelligence/openpi)，Real-Time Chunking 来自 Black 等人（2025）。reBot 机械臂由 [Seeed Studio](https://github.com/Seeed-Projects) 设计。JetBrains Mono 字体采用 SIL Open Font License。

## 许可证

Apache-2.0，见 [LICENSE](https://github.com/MuQY1818/macbot/blob/main/LICENSE)。
