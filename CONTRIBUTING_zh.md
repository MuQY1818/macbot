# 参与 macbot 开发

[English](CONTRIBUTING.md)

## 开发环境

在独立环境中使用 Python 3.12 或 3.13。在源码目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pyflakes macbot tests tools
pytest -q
```

使用 Ruff 检查代码，行宽尽量保持在 120 列以内。风格调整仅限本次改动涉及的代码。
要在每次提交前运行检查，可执行 `pre-commit install`。

CI 在 macOS 和 Ubuntu 上运行快速的无硬件测试。要复现其最小环境，
请在装有 pip 25.1 或更新版本的新虚拟环境中运行：

```bash
python -m pip install --no-deps -e .
python -m pip install --group lint --group test
pytest -q -rs
```

需要 LeRobot 的测试使用 `pytest.importorskip`。网页服务应在进程内测试：
直接操作服务状态或请求处理器，无需通过回环 HTTP 发送请求。参见[采集驱动脚本](tools/drive_collect.py)。

## 无硬件试跑

```bash
macbot --config configs/so101.yaml collect --fake
macbot --config configs/so101.yaml health --data ~/macbot-data/raw/pick_cube_fake
```

在采集页面按 **S** 开始，按 **回车** 保存。要验证模拟器，先运行
`python -m pip install -e ".[dev,sim]"`，再运行 `macbot --config configs/so101_sim.yaml collect`，
不加 `--fake`。固定版本的场景会在首次使用时下载并执行 Python 代码；参见
[仿真安装](docs/setup_zh.md#本地-mujoco-模拟器)和[工作流程](docs/guide_zh.md#仿真)。
跑策略时使用与关节布局和相机匹配的检查点；`run --fake` 仍会运行真实策略。

## 扩展 macbot

- **机械臂：** 从[机械臂支持](docs/robots_zh.md)开始。优先使用兼容的 LeRobot 配置；只有设备需要不同
  行为时才新增 `Robot` 实现。无硬件测试应覆盖关节顺序、动作单位、标定保护和退出行为。
- **执行方式：** 在 [`macbot/policy/executors.py`](macbot/policy/executors.py) 实现 `Executor` 接口。
  要改变异步动作块的替换方式，可以覆写 `AsyncChunk.accept()`。在 `macbot/cli.py` 和运行器中
  接入新模式，并用 `FakePolicy` 测试时序、暂停和重置行为。参见[架构](docs/architecture_zh.md)。

## 提交和拉取请求

每次提交集中处理一件事，使用简洁的英文标题。每次提交前运行上述检查。提交 PR 时说明行为变化、
验证方法和结果。英文和简体中文文档应同步更新；用户可见的改动应记在 **Unreleased（未发布）** 下。

欢迎提交真机测试报告。请注明机械臂与相机配置、macOS 版本和芯片、成功与失败的操作，条件允许时提供
`macbot bench` 数字。请使用真机报告表单，并事先移除日志和截图中的个人信息。
