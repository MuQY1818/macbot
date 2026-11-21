<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/logo.svg" alt="macbot" height="72">
  </picture>
</p>

<p align="center"><b>在 MacBook 上跑 LeRobot。</b><br>
把已有的 LeRobot 机械臂、校准、数据集和 checkpoint 搬到 MacBook。</p>

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
macbot 处理 Mac 设备配置，数据集和 checkpoint 保持 LeRobot 格式，可在两台机器间直接使用。

```bash
macbot doctor
macbot init --from-lerobot "$LEROBOT_COMMAND"   # 原 lerobot-record 或 lerobot-teleoperate 命令
macbot collect
macbot health
macbot export --out ./dataset
# 在 GPU 机器上用 lerobot-train 训练，再把 checkpoint 拿回来。
macbot run --policy ./pretrained_model       # 也可用 owner/model@revision
macbot replay
```

真机和仿真验证范围见[机械臂支持](docs/robots_zh.md)，已测 checkpoint 见[策略支持](docs/policies_zh.md)。
没有机械臂也可以运行 `collect --fake`。

以下数据在 MacBook Pro（M5 Pro，48 GB）上实测，为每个动作块的大致耗时：

| 策略 | 动作块步数 | PyTorch MPS | MLX |
|---|---:|---:|---:|
| π0.5 | 50 | 270 ms | **165 ms** |
| π0.5 + RTC | 50 | 350 ms | **170 ms** |
| SmolVLA | 50 | 300 ms | — |
| X-VLA | 30 | 235 ms | — |
| ACT | 100 | 25 ms | — |

π0.5 去噪 10 步，PyTorch 用 bf16 加 `torch.compile`，MLX 用 fp16（`--backend mlx`）。MLX 输出的动作与 PyTorch 平均相差不到 0.1°。RTC 指 Real-Time Chunking 引导。SmolVLA 为官方基础模型，X-VLA 为 SO-101 微调版。测法和完整结果见[推理基准](https://github.com/MuQY1818/macbot/blob/main/docs/acceleration_zh.md)，其他策略见[策略支持](https://github.com/MuQY1818/macbot/blob/main/docs/policies_zh.md)。

| 数据采集 | 实测 |
|---|---:|
| 遥操作控制频率 | 60 Hz |
| 录制采样频率 | 30 fps |
| 采样时刻的画面延迟 p95 | ≤ 40 ms |

<p align="center"><img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/collect_zh.jpg" alt="macbot 采集控制台" width="100%"></p>

通过 `health --dataset` 和 `replay --dataset` 只读查看本地或 Hub 上的 LeRobot v3.0 数据集。
采集、执行策略和试验回放见[工作流程](docs/guide_zh.md)；相机权限和 reBot CAN 驱动见[安装指南](docs/setup_zh.md)。

## 参与贡献

开发与真机报告见[贡献指南](CONTRIBUTING_zh.md)，代码布局见[架构](docs/architecture_zh.md)，
后续工作见[路线图](ROADMAP_zh.md)。

## 安全

macbot 会驱动真实的机械臂。所有控制台启动时都处于暂停或待机状态；会限制关节速度和范围、限制夹爪夹紧力；相机断流时自动暂停；退出时先慢速回零再让电机失能。请保持工作区域无人、手边留着电源开关；新策略先用 `--speed 0.3` 跑。

## 致谢

基于 [LeRobot](https://github.com/huggingface/lerobot)。π0.5 和图像增强配方来自 [openpi](https://github.com/Physical-Intelligence/openpi)，Real-Time Chunking 来自 Black 等人（2025）。reBot 机械臂由 [Seeed Studio](https://github.com/Seeed-Projects) 设计。JetBrains Mono 字体采用 SIL Open Font License。

## 许可证

Apache-2.0，见 [LICENSE](https://github.com/MuQY1818/macbot/blob/main/LICENSE)。
