# 变更日志

[English](CHANGELOG.md)

macbot 的主要变更记录在这里。本项目遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 和
[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.3.0] - 2026-11-20

### 新增

- π0.5 的可选 MLX 后端（`--backend mlx` 或 `policy.backend: mlx`，安装 `macbot[mlx]`）。直接加载
  LeRobot 的 pi05 checkpoint，fp16 运行，带编译、文本补齐裁剪、AdaRMS 预计算、融合 RoPE 和投影合并：
  M5 Pro 上每个 50 步动作块约 165 ms，PyTorch MPS 约 270 ms；动作与 bf16 参照的误差平均 0.08°、
  p99 0.42°，远小于换随机种子造成的差异。带引导的 RTC 仍走 PyTorch；用 MLX 时 `--mode auto` 选异步分块。
- `macbot bench --profile` 对两种后端都给出同步计时的分阶段耗时（视觉、前缀、每一步去噪），
  `--steps` 可设更短的去噪步数。
- `tools/accel/`：离线保真度评测工具，从采集的数据里抽固定观测集，用相同噪声对比任意配置，
  报告延迟和动作误差，并以换种子的差异为标尺。
- 推理基准文档（`docs/acceleration.md`、`docs/acceleration_zh.md`）。

### 修复

- π0.5 checkpoint 存的是补齐到 32 维的状态；启动时的维度检查改为从归一化参数读取真实维度，
  `macbot run` 重新可以加载 π0.5。

## [0.2.0] - 2026-11-19

### 新增

- 统一的机械臂接口，包含 reBot B601 专用配置和通用 LeRobot 主从臂配置，面向兼容的关节位置控制
  机械臂，包括 SO-100/101、Koch、OMX、OpenArm 和 YAM。
- SO-101 UVC 配置、`macbot robots` 命令和可配置的 fake 关节布局。
- 本地 SO-101 MuJoCo 模拟器，提供脚本主臂、场景重置和机械臂观测相机（`type: robot`）。
- 进程内采集驱动脚本，以及覆盖通用机械臂、模拟相机、导出和诊断的测试。
- 双语贡献指南、架构、机械臂支持和路线图，社区政策，GitHub 议题和 PR 模板，以及每月更新
  GitHub Actions 依赖的配置。
- Ruff 和 pre-commit 开发工具；CI 覆盖 macOS 和 Ubuntu 上的 Python 3.12 和 3.13。

### 变更

- 采集、检查、试验控制和导出使用配置的关节布局和机械臂元信息。策略启动时检查检查点的状态与动作维度；
  导出保留机械臂类型并支持保存完整画面。
- `macbot doctor` 按所选配置检查环境，包括通用机械臂的驱动、串口和标定，以及内置模拟器的本地依赖。
- 补充 PyPI 软件包元数据和使用 Trusted Publishing 的发布工作流。

### 修复

- `macbot bench` 使用检查点中的状态维度，支持关节数少于或多于七个的机械臂策略测速。

## [0.1.0] - 2026-11-19

### 新增

- Apple 芯片上使用 reBot B601-RS 和 Arm 102 的首套参考流程，配备 PCAN-USB、UVC 和 Orbbec
  相机，以及用于无硬件测试的 fake 机械臂和相机。
- 浏览器采集，支持语言指令、逐条检查和回放；导出 LeRobot v3 数据集，可选二值夹爪标签，提供训练脚本示例。
- 在 MPS 上运行 ACT 和 π0.5，支持同步分块、时间集成、异步分块和 Real-Time Chunking，
  以及试验回放、MP4 导出和推理测速。
- 设备诊断、安装和使用指南，以及英文与简体中文控制台。

[Unreleased]: https://github.com/MuQY1818/macbot/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/MuQY1818/macbot/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MuQY1818/macbot/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MuQY1818/macbot/releases/tag/v0.1.0
