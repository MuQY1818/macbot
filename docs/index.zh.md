# macbot 文档

[English](index.zh.md)

macbot 让 LeRobot 在 Apple 芯片的 Mac 上运行：配置机械臂和相机、录制示范、导出为 LeRobot 数据集，再把训练好的策略
部署到机器人上。在 MLX 更快的模型上，它会用 MLX 加速推理。

<div class="grid cards" markdown>

-   **第一次使用**

    ---

    从组装好的主从臂开始，录下第一条示范。

    [快速开始](quickstart.zh.md)

-   **已经在 Linux 上用 LeRobot**

    ---

    沿用现有的标定文件、`lerobot-record` 命令、数据集和权重。

    [从 Linux 迁移](migrate.zh.md)

-   **让策略跑得更快**

    ---

    哪些策略有 MLX 后端、快多少、怎么开启。

    [加速](acceleration.zh.md)

-   **还没有硬件**

    ---

    在仿真场景里试用控制台、键盘遥操作和评测。

    [仿真](simulation.zh.md)

</div>

## 工作流程

每一步都读写 LeRobot 标准格式，任何一步都可以换回你自己的工具。

| 步骤 | 命令 | 说明 |
|---|---|---|
| 配置机械臂和相机 | `macbot init` | [快速开始](quickstart.zh.md) |
| 录制示范 | `macbot collect` | [采集](guide.zh.md#1-采集) |
| 检查并导出 | `macbot check`、`macbot export` | [检查](guide.zh.md#2-检查数据)、[导出](guide.zh.md#3-导出) |
| 在 GPU 机器上训练 | LeRobot 训练 | [训练](guide.zh.md#4-训练) |
| 运行策略 | `macbot run` | [运行策略](guide.zh.md#5-运行策略) |
| 回看试验 | `macbot replay` | [回看](guide.zh.md#6-回看) |

支持的机械臂见[机械臂](robots.zh.md)，支持的策略见[策略](policies.zh.md)。
