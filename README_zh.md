<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/assets/logo.svg" alt="macbot" height="72">
  </picture>
</p>

<p align="center"><b>在 MacBook 上跑 LeRobot。</b><br>
在 Apple 芯片的 Mac 上录制示范，把 LeRobot 策略直接跑在真实机械臂上。</p>

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

## 上手

```bash
pip install macbot
macbot init        # 在浏览器里选机械臂和相机
macbot collect     # 录制示范
```

第一次录制的完整步骤见[快速开始](docs/quickstart.zh.md)。如果你已经在 Linux 上用 LeRobot，`macbot init --from-lerobot` 可以直接导入你的 `lerobot-record` 命令，其余的看[迁移指南](docs/migrate.zh.md)。需要 Python 3.12 或更高版本。

macbot 读写的都是标准 LeRobot 格式，随时可以换回自己的工具。完整走一遍是这样的：

```bash
macbot collect                               # 录制
macbot health                                # 检查示范
macbot export --out ./dataset                # 导出 LeRobot v3.0 数据集
# 在 GPU 机器上用 lerobot-train 训练，再把权重拷回来
macbot run --policy ./pretrained_model       # 也可以用 Hub 上的 owner/model@revision
macbot replay                                # 回看试验
```

## 速度

MacBook Pro（M5 Pro，48 GB）上每个动作块的耗时：

| 策略 | 执行方式 | PyTorch MPS | MLX |
|---|---|---:|---:|
| π0.5 | 分块 50 | 270 ms | **165 ms** |
| π0.5 | RTC 50 | 350 ms | **170 ms** |
| SmolVLA | 分块 50 | 296 ms | **53 ms** |
| X-VLA | 分块 30 | 178 ms | **138 ms** |
| ACT | 分块 100 | 16 ms | **8 ms** |

给 `macbot run` 加上 `--backend auto`，在已经和 PyTorch 对照验证过的配置上就会用 MLX。测量用的是录制的真实观测，π0.5 是一个 SO-101 微调权重，SmolVLA、X-VLA 和 ACT 用公开权重，VLA 都去噪 10 步。误差数字和具体设置见[加速](docs/acceleration.zh.md)，支持的全部策略见[策略](docs/policies.zh.md)。π0.5 需要至少 32 GB 内存。

遥操作以 60 Hz 运行，录制为 30 fps。采样时相机画面最多延迟 40 ms（p95）。

<p align="center"><img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/collect_zh.jpg" alt="macbot 采集控制台" width="100%"></p>

| 运行策略 | 回放试验 |
|---|---|
| <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/run_zh.jpg" alt="运行策略"> | <img src="https://raw.githubusercontent.com/MuQY1818/macbot/main/docs/img/replay_zh.jpg" alt="回放"> |

## 硬件

我们在 Seeed reBot B601-RS 上做实机测试。SO-100/101、Koch 和 OMX 通过各自的 LeRobot 驱动运行，但目前只在假硬件上验证过，非常欢迎用真机的朋友反馈结果。SO-101 [仿真](docs/simulation.zh.md)能跑通整个流程，不过还没有训练出能完成任务的策略。各配置测过哪些见[机械臂](docs/robots.zh.md)。

有几处在 macOS 上要额外处理。参考配置因为 Orbbec 相机要用 sudo 运行，而 sudo 会丢掉库路径，所以 macbot 自己加载 PCBUSB CAN 驱动，并把 sudo 下录制的文件交还给你的用户。macOS 还喜欢把附近的 iPhone 当成相机，所以 macbot 按唯一 ID 选相机，不按名称或序号。

## 参与贡献

欢迎提交问题、硬件测试报告和 PR。开发环境见[贡献指南](CONTRIBUTING_zh.md)，代码结构见[架构](docs/architecture.zh.md)，接下来的计划见[路线图](ROADMAP_zh.md)。

## 安全

macbot 会驱动真实的机械臂。所有控制台启动时都是暂停或待机状态，会限制关节速度和范围、限制夹爪力度，相机断流时自动暂停，退出时先慢速回零再让电机断使能。请清空工作区，手放在电源开关附近，新策略先用 `--speed 0.3` 跑。

## 致谢

基于 [LeRobot](https://github.com/huggingface/lerobot)。π0.5 和图像增强配方来自 [openpi](https://github.com/Physical-Intelligence/openpi)，Real-Time Chunking 来自 Black 等人（2025）。reBot 机械臂由 [Seeed Studio](https://github.com/Seeed-Projects) 设计。JetBrains Mono 字体采用 SIL Open Font License。

## 许可证

Apache-2.0，见 [LICENSE](https://github.com/MuQY1818/macbot/blob/main/LICENSE)。
