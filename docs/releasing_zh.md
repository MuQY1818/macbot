# 版本号与发版

[English](releasing.md)

macbot 遵循[语义化版本](https://semver.org/lang/zh-CN/)，0.x 阶段按通行做法处理。版本号写作 `MAJOR.MINOR.PATCH`（主版本.次版本.修订号）。

## 哪些算对外接口

如果用户什么都没改，升级之后原来能用的配置却用不了了，或者行为变了，这就是不兼容的改动。对外接口包括：

- 命令行的命令、参数和默认值（`macbot collect`、`run`、`export`、`bench` 等）；
- YAML 配置项及其含义（`macbot/default.yaml`）；
- 磁盘上的文件格式：原始数据、试验记录、`gripper.json`，以及导出的 LeRobot 数据集结构；
- Python 扩展接口：`Robot` 接口（`macbot/robot/base.py`）、相机类型、执行方式；
- 会改变机械臂动作的默认行为：默认后端、执行方式、速度和关节限位、安全停止；
- 支持的 Python 和 LeRobot 版本。

内部模块、`tools/` 下的脚本、日志文字和网页布局不算对外接口。

## 0.x 阶段（现在）

| 升哪一位 | 什么时候 | 例子 |
|---|---|---|
| `0.MINOR.0` | 对外接口有不兼容的改动，或者改变机械臂行为的默认值变了。 | 配置项改名或删除；数据格式变了；`--mode auto` 改选别的执行方式；LeRobot 依赖升到新的次版本。 |
| `0.x.PATCH` | 所有向后兼容的改动：修 bug、新增可选功能、支持新的机械臂或策略、新增默认安全的参数、不改变输出的加速、文档。 | MLX 后端支持 RTC；新增 `--mlx-time-grid` 参数；支持又一款 LeRobot 机械臂；修复崩溃。 |

所以在 0.x 阶段，新功能一般以修订版发布。次版本号一变，就是提醒用户升级前先看 CHANGELOG。每一项不兼容改动都要写在 **Changed** 或 **Removed** 下，并说明用户该怎么改。

## 1.0.0 之后

1.0.0 表示上面这些接口稳定了，条件是：至少两个系列的机械臂在真机上验证过，数据和试验记录格式冻结，配置项有完整文档。之后按标准语义化版本走：不兼容改动升 `MAJOR`，向后兼容的新功能升 `MINOR`，只修 bug 升 `PATCH`。

## 发版流程

1. 改动通过 PR 合进 `main`，CI 必须全绿，每个 PR 在两份 CHANGELOG 的 `## [Unreleased]` 下各加一行。
2. 攒到值得用户安装的程度再发版，不是合一次发一次。涉及安全或崩溃的修复要及时发修订版。
3. 需要先试用再依赖的改动（新后端、新默认值），可以先发预发布版，例如 `0.4.0rc1`。只有用户明确要求时 pip 才会装它（`pip install --pre macbot`）。
4. 发版步骤：按上表定版本号；把 `Unreleased` 下的内容移到带日期的新版本小节（`CHANGELOG.md` 和 `CHANGELOG_zh.md` 都要改）；改 `pyproject.toml` 和 `macbot/__init__.py` 里的版本号；合并后从 `main` 发布 GitHub Release `vX.Y.Z`。发布流程会通过 Trusted Publishing 自动上传 PyPI。
5. 发出去的版本有问题，就发一个新的修订版来修，不撤回重发。装了会有危险的版本，同时在 PyPI 上 yank。
