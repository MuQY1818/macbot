# 架构

[English](architecture.md)

```text
机械臂 + 相机 -> 采集 -> 原始 NPZ -> 导出 -> LeRobot 数据集
                                                 |
                                                训练
                                                 |
机械臂 + 相机 <-> 运行 <- 策略权重 <--------------+
                   |
               试验 NPZ -> 回放 / 视频
```

## 命令与配置

`macbot/cli.py` 分发各个命令。`macbot/config.py` 把用户的 YAML 叠加到 `macbot/default.yaml` 上，其中 `cameras` 整体替换，不做合并。`macbot/web/` 存放四个控制台（配置、采集、运行、回放）的 HTTP 处理器、模板和公共资源。

配置向导（`macbot/init_wizard.py`）生成配置文件，借助 `robot/usb.py` 和 `cameras/discovery.py` 发现设备并显示预览。从向导启动采集时，它先释放预览，再交给独立的 `collect` 命令。`lerobot_command.py` 把导入的 LeRobot 命令当作纯数据解析，`hub.py` 把 Hub ID 解析为标准缓存中固定版本的文件。

## 模块

| 模块 | 作用 |
|---|---|
| `robot/` | 打开 reBot 配置、通用 LeRobot 机械臂、仿真器或假硬件 |
| `cameras/` | UVC 相机（Swift 和 AVFoundation）、Orbbec、假相机和仿真观测 |
| `collect/` | 主从臂遥操作循环、按固定时间网格录制、示范检查，以及仿真脚本示范 |
| `data/` | 健康报告、空抓检测、读取 LeRobot 数据集，以及导出为 LeRobot v3.0 |
| `policy/` | 加载策略、执行器、运行循环、评测，以及 MLX 后端（`mlx_*.py`） |
| `viz/` | 试验回放和 MP4 导出 |
| `safety.py` | 电机温度和跟踪误差看门狗 |
| `doctor.py` | 检查驱动、设备、标定和仿真配置 |
| `train/`（仓库顶层） | 在 GPU 机器上使用的训练脚本示例 |

## 扩展点

### 机械臂

[`Robot`](../macbot/robot/base.py) 接口提供有序的关节名、可选的夹爪和限位，以及 `connect(leader=...)`、`read()`、`read_leader()`、`send()`、`home_pose()` 和 `shutdown()`。`send()` 返回经过机械臂自身限制之后实际发出的指令。关节单位由驱动决定，每条示范都会记录关节布局和机械臂类型。新增机械臂的步骤见[机械臂](robots.zh.md#新增机械臂类)。

### 相机

[相机接口](../macbot/cameras/__init__.py)只约定结构：`latest()` 返回一帧 BGR 图像及其时间戳，第一帧到来之前返回 `None`；`age()` 报告画面已经过去多久；`close()` 释放设备。`type: robot` 相机读取仿真器中指定名称的观测，流程中的其他部分把它当作普通相机使用。

### 执行方式

运行器通过 [`Executor`](../macbot/policy/executors.py) 把推理和控制循环分开。同步分块按需推理，时间集成对重叠的预测加权平均，异步分块在后台线程推理。Real-Time Chunking 在异步路径上加入引导，`AsyncChunk.accept()` 决定新动作块何时接管。开始和暂停、速度限制以及录制都由运行器负责，与执行器无关。接口见[工作流程](guide.zh.md#加入新的执行方式)。

## 策略与后端

通用适配器加载 LeRobot 策略及其保存的处理器。如果策略提供 LeRobot 的分块接口，适配器同步执行这些动作块；否则在每个策略步调用 `select_action`，由 LeRobot 管理观测历史和动作队列。ACT 和 π0.5 保留各自的封装和默认执行方式。

`policy/backends.py` 区分策略能运行的后端和 `--backend auto` 认可的配置。判断只依据权重元数据和一张固定的表，不做任何计时，运行器会把结果记录到每次试验中。
