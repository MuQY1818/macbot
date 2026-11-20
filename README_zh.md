<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="macbot" height="72">
  </picture>
</p>

<p align="center"><b>一台 MacBook 搞定真机学习。</b><br>
遥操作采数据、导出 LeRobot 数据集、在真机上跑 ACT 和 π0.5，全部在 Apple 芯片的 Mac 上完成。</p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20silicon-18171F">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-F2701A">
  <img alt="LeRobot" src="https://img.shields.io/badge/LeRobot-0.7.0-1F9D63">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-6B6A75">
</p>

<p align="center"><a href="README.md">English</a> · <b>中文</b></p>

<p align="center"><img src="docs/img/collect_zh.jpg" alt="macbot 采集控制台" width="100%"></p>

## 为什么做这个

做真机实验时，大家通常默认机械臂旁边要放一台装 Ubuntu、插 NVIDIA 显卡的工作站。我们发现，机械臂旁边的工作一台 MacBook 就够了：遥操作采数据、检查数据、导出 LeRobot 数据集，以及闭环跑策略，包括 π0.5。训练仍然放在 GPU 服务器上。macbot 就是我们日常用的这套工具，整理之后开源出来，方便大家复用。

以下数据在 MacBook Pro（M5 Pro，48 GB）上用 PyTorch MPS 实测：

| | |
|---|---|
| π0.5 推理（动作块 50 步，去噪 10 步） | **约 280 ms**（bf16 + `torch.compile`） |
| π0.5 加 Real-Time Chunking 引导 | **约 350 ms** |
| ACT 推理（动作块 100 步） | **约 25 ms** |
| 遥操作控制频率 / 录制采样频率 | 60 Hz / 30 fps |
| 采样时刻的画面延迟 p95 | **≤ 40 ms**（俯视 UVC + 腕部 Orbbec） |

## 功能

- **采集控制台**（网页）：实时画面；一键录制、保存、丢弃；每条数据单独记录语言指令；保存时自动检查画面延迟、重复帧、关节跳变和采样时序；任何一条都能立即回放，带关节曲线。
- **导出**为 LeRobot v3 数据集，夹爪动作可以选择二值化（张开或闭合）。
- **跑策略控制台**：支持 ACT 和 π0.5，提供四种执行方式（同步分块、时间集成、异步分块、Real-Time Chunking）。带速度和关节限位；可以暂停、回起始姿态、结束时分两段慢速回零；每次试验连同画面和时序一起保存。
- **试验回放**：多路画面同步播放，显示指令与实测曲线和换块位置，也能导出左右并排的 **MP4**。
- **`macbot doctor`**：开工前检查依赖、MPS、CAN 驱动、相机和标定文件。
- 所有涉及机械臂的命令都支持 **`--fake`**，用模拟的机械臂和相机跑通整条流程，任何一台 Mac 都能试。
- 每个网页都有中英文（顶栏切换），支持浅色和深色模式。

| 跑策略 | 回放试验 |
|---|---|
| <img src="docs/img/run_zh.jpg" alt="跑策略"> | <img src="docs/img/replay_zh.jpg" alt="回放"> |

## 硬件

| | 我们用的 | 说明 |
|---|---|---|
| 从臂 | Seeed reBot B601-RS | 通过 LeRobot 的 `rebot_b601_follower` 驱动 |
| 主臂 | Seeed reBot Arm 102 | USB 串口 |
| CAN 适配器 | PEAK PCAN-USB | macOS 驱动用 PCBUSB |
| 俯视相机 | 任意 UVC 摄像头（1080p） | 按 AVFoundation uniqueID 打开 |
| 腕部相机 | Orbbec Gemini 2 | 彩色流，经 `pyorbbecsdk2` 读取 |
| 电脑 | Apple 芯片的 MacBook Pro | 跑 π0.5 建议 32 GB 以上内存 |

LeRobot 支持的其他机械臂可以加到 `macbot/robot/`。控制台只需要 `connect`、`get_observation`、`send_action`、`disconnect` 四个接口。

## 快速开始

```bash
git clone https://github.com/MuQY1818/macbot && cd macbot
conda create -n macbot python=3.12 -y && conda activate macbot
pip install -e ".[orbbec,pi]"          # 只用 ACT 可以去掉 pi

macbot doctor                          # 看看哪些就绪、缺什么
macbot collect --fake                  # 用模拟机械臂和相机打开采集控制台
```

打开 <http://127.0.0.1:8766>。按 **S** 录制，**回车** 保存，**←** 丢弃。
连真机之前，先装 PCAN 驱动、填好相机 ID，见 [docs/setup_zh.md](docs/setup_zh.md)。

## 工作流程

```bash
sudo -E $(which python) -m macbot.cli collect          # 1. 遥操作录数据（Orbbec SDK 需要 sudo）
macbot health                                          # 2. 逐条看跟踪误差和采样时序
macbot export --gripper binary                         # 3. 导出到 ~/macbot-data/datasets/
DATASET=... ./train/train_pi05.sh my_run               # 4. 在 GPU 服务器上训练（ACT 用 train_act.sh）
sudo -E $(which python) -m macbot.cli run --policy runs/my_run/checkpoints/last/pretrained_model
macbot replay                                          # 5. 回放试验；macbot video <rollout.npz> 导出 MP4
```

完整说明（含数据格式和各执行方式）见 [docs/guide_zh.md](docs/guide_zh.md)。
自己的设置写在一个 YAML 里，用 `macbot --config my.yaml <命令>` 传入，没写的项沿用 [macbot/default.yaml](macbot/default.yaml)。可以从 [configs/example.yaml](configs/example.yaml) 开始改。

## 在 macOS 上跑起来要解决的问题

- **sudo 下的 CAN。** Orbbec SDK 要 root 权限，而 root 进程的 dyld 不搜索 `/usr/local/lib`，所以要在 CAN 后端加载之前手动预载 `libPCBUSB.dylib`。
- **打开正确的相机。** 按名字或编号打开 UVC 相机，可能会抓到通过“连续互通相机”连上的 iPhone。macbot 用一个 Swift 小程序按 AVFoundation uniqueID 打开相机，相机换了 USB 口也能找到。
- **π0.5 跑在 MPS 上。** 权重用 bf16 并开 autocast；跳过被遮掉的第三个相机槽（输出完全一致，省约 70 ms）；对去噪步和语言模型做 `torch.compile`，减少逐个 kernel 派发的开销（约 410 → 280 ms）；跳过反正会被权重覆盖的随机初始化，每次加载省约 80 秒。
- **root 写出的文件**会交还给运行 `sudo` 的用户。

## 目录结构

```
macbot/
  cli.py            命令入口：macbot <命令>
  default.yaml      全部设置及注释
  robot/            带前馈的 reBot 从臂、主臂、安全退出、模拟机械臂
  cameras/          UVC（Swift 助手）、Orbbec、模拟相机
  collect/          采集服务和逐条检查
  data/             导出 LeRobot、数据健康报告
  policy/           ACT / π0.5 封装、执行方式、二值夹爪、运行器、测速
  viz/              试验回放服务、导出 MP4
  web/              网页、主题、公共脚本（中英文）
train/              带 openpi 风格增强的 lerobot-train、示例启动脚本
scripts/            访达里双击即可运行的启动脚本
tools/              logo 生成、网页截图
tests/              不需要硬件的快速测试
```

## 安全

macbot 会驱动真实的机械臂。所有控制台启动时都处于暂停或待机状态；会限制关节速度和范围、限制夹爪夹紧力；相机断流时自动暂停；退出时先慢速回零再让电机失能。请保持工作区域无人、手边留着电源开关；新策略先用 `--speed 0.3` 跑。

## 致谢

基于 [LeRobot](https://github.com/huggingface/lerobot)。π0.5 和图像增强配方来自 [openpi](https://github.com/Physical-Intelligence/openpi)，Real-Time Chunking 来自 Black 等人（2025）。reBot 机械臂由 [Seeed Studio](https://github.com/Seeed-Projects) 设计。JetBrains Mono 字体采用 SIL Open Font License。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
