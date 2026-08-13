# 云端训练工具索引

- 连接云服务器前读取本机`docs/CLOUD_SERVER_LOCAL.md`，只使用其中登记的当前实例。
- 训练面板从`runs/training_queue.json`读取队列；`src/daxigua/rl/training_queue.py`负责该文件的校验、合并与原子写入契约，实际启动任务的是云端接力或调度守护。
- 部署新训练前先查看现有训练进程、队列文件和`/root/autodl-tmp/training-relay/`，有可用接力时优先复用，避免误判为空闲或重复启动任务。
- 队列格式见`docs/PROJECT_PORTAL.md`；已有接力实例见`docs/codex/48_部署辅助旁路至128M基线自动接力_2026_08_10.md`和`docs/codex/60_部署128M至120FPS适应训练自动接力_2026_08_11.md`。

训练方案与部署方式仍在演进，具体检查项、恢复方式和完成条件以当次配置、代码及实验目标为准。
