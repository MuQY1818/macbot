# 路线图

[English](ROADMAP.md)

## 第一阶段：便于上手

- 在 README 中加入演示视频和 GIF。
- 在 Hugging Face Hub 发布示例数据集和已训练策略，让没有机械臂的用户也能回放和运行 fake 模式。
- 维护 CI、PyPI 发布、贡献指南，以及带 `good first issue` 标签的议题。
- **进行中：** SO-100/101 已用 fake 机械臂试跑，尚未验证真机。SO-101 模拟器已在仿真中试跑，
  这不代表真机验证。

## 第二阶段：验证更多硬件和工作流程

- 完成 SO-100/101 真机验证。
- 编写并验证只用 UVC 相机、无需 `sudo` 的配置。
- 在 Apple 芯片上用 MPS 训练 ACT 和 SmolVLA。
- 汇总通过 PR 提交的 `macbot bench` 结果，建立覆盖 M1–M5 芯片的社区基准表。

## 第三阶段：扩展执行方式和数据工具

- 以 `AsyncChunk.accept()` 钩子为起点，以插件形式支持更多执行方式。
- 在 GPU 服务器上运行策略推理，由 Mac 操作机械臂和相机。
- 加入失败标注、一条命令上传到 Hugging Face Hub，以及数据集合并。
