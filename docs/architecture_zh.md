# 架构

[English](architecture.md)

```text
机械臂 + 相机 -> 采集 -> 原始 NPZ -> 导出 -> LeRobot 数据集
                                                 |
                                                训练
                                                 |
机械臂 + 相机 <-> 运行 <- 策略检查点 <------------+
                   |
               试验 NPZ -> 回放 / 视频
```

`macbot/cli.py` 选择命令，`macbot/config.py` 将用户 YAML 配置叠加到 `macbot/default.yaml`。
`cameras` 映射作为整体替换。`macbot/web/` 为采集、运行和回放提供 HTTP 处理器、模板和公共资源。

| 模块 | 职责 |
|---|---|
| `robot/` | 打开 reBot 专用配置、通用 LeRobot 机械臂、模拟器或 fake 机械臂。 |
| `cameras/` | 提供 UVC（Swift/AVFoundation）、Orbbec、fake 和来自机械臂观测的相机数据源。 |
| `collect/` | 运行主从臂控制循环，按固定时间网格录制关节指令和观测，并检查每条数据。 |
| `data/`、`train/` | 检查原始数据并导出为 LeRobot v3；示例训练脚本产出策略检查点。 |
| `policy/`、`viz/` | 加载 LeRobot 策略，控制机械臂并记录试验；回放试验文件或导出 MP4。 |

[`Robot`](../macbot/robot/base.py) 接口提供有序的关节名、可选的夹爪和限位，以及
`connect(leader=...)`、`read()`、`read_leader()`、`send()`、`home_pose()` 和 `shutdown()`。
`send()` 返回执行设备限位后实际下发的值。每条数据的元信息记录关节布局和机器人类型。
关节的单位由驱动定义。

[相机接口](../macbot/cameras/__init__.py) 通过方法约定实现：`latest()` 返回 BGR 画面和时间戳
（首帧之前图像为 `None`），`age()` 提供画面时效，`close()` 释放资源。
`type: robot` 相机读取指定名称的模拟器观测，对其余流程来说与其他相机相同。

运行器通过 [`Executor`](../macbot/policy/executors.py) 将推理与控制循环分开。
同步分块按需预测，时间集成合并重叠预测，异步分块在后台线程预测。Real-Time Chunking 为异步执行
加入引导。`AsyncChunk.accept()` 决定新动作块在何时生效。运行器独立处理启动、暂停、速度限制和录制。

通用适配器通过 LeRobot 加载策略和随 checkpoint 保存的处理器。支持直接预测动作块的策略使用同步分块；
其余策略在每个策略时钟调用一次 `select_action`，由 LeRobot 管理观测历史和动作队列。
ACT 和 π0.5 保留原有封装和默认执行方式。
