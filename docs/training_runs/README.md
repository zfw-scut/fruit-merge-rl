# 训练实验目录

本目录保存从本地 `runs/` 提取的轻量训练摘要，目的是让迁移后的开发者和
Agent 在没有原机器全部训练文件的情况下，仍能了解历史实验参数、训练进度、
单局得分、reward breakdown、性能和大型产物位置。

## 阅读顺序

1. `INDEX.md`：查看所有训练实验和辅助输出。
2. `runs/<run_id>/summary.md`：查看某次训练的人类可读总结。
3. `runs/<run_id>/metrics_summary.json`：供程序或 Agent 做结构化比较。
4. `runs/<run_id>/config.json`：查看该次训练真正生效的参数。
5. `runs/<run_id>/artifacts.md`：决定还需要单独迁移哪些大文件。

## 更新方式

在原始训练数据仍位于项目 `runs/` 时运行：

```bash
python tools/export_training_catalog.py
```

训练数据在其他目录时：

```bash
python tools/export_training_catalog.py \
  --runs-dir /path/to/runs \
  --output-dir docs/training_runs
```

脚本会重建 `runs/` 下的生成摘要。不要在生成的单 Run 目录中手工记录长期
结论；需要人工补充的分析应另建文档，避免下次导出时丢失。

## Git 与大型文件

本目录只包含小型 Markdown 和 JSON，不包含：

- PyTorch checkpoint；
- ReplayBuffer 冷段；
- 完整训练 CSV；
- 资源监控原始日志。

这些原始文件仍位于被 `.gitignore` 忽略的 `runs/`。迁移模型时至少另行复制
所需 checkpoint；只有继续使用原经验池训练时才需要复制 ReplayBuffer。

## 数据解释限制

- 早期训练版本没有 `episode_metrics.csv` 或 reward breakdown 时，对应字段会显示“未记录”。
- 当前导出器同时识别 Reward V2 的 task/potential 指标与 Reward V1 的
  score/survival/height 指标；旧字段只用于解释历史实验，不代表仍在当前训练中启用。
- Reward V2 的 `StateAnalyzer` 性能、降级率和 shaping p95 只会出现在采用新版
  `metrics.csv` 的实验中。
- 历史 `config.json` 没有 Git commit 字段，因此不能可靠推断训练对应的源码提交。
- 指标摘要来自已经落盘的 CSV；突然断电前尚未 flush 的最后几轮不会出现。
