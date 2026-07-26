# Project Documentation

本目录用于存放项目文档，方便后续开发、复盘和让新的 agent 快速理解项目。

## 目录说明

- `codex/`: 记录 Codex 对项目做出的较大修改，以及修改记录的维护规则。
- `project_map/`: 记录项目文件构成、职责分工、可复用组件和阅读入口。
- `training_runs/`: 记录本地训练实验的轻量摘要、配置、指标统计和产物索引，供迁移后的开发者或 agent 分析。
- `CODING_STYLE.md`: 记录项目源码注释和编码风格，当前强调教学型详细注释。
- `rl/`: 存放后续强化学习模型、环境接口和算法方案的设计参考文档。

## 当前 RL 开发入口

- `rl/CAUSAL_ATTRIBUTION_V1.md`: 第一次大规模训练采用的 Reward V2、完整状态归因、
  因果 Q 排序和稀疏反事实实现规格。
- `rl/INTERFACE_V0.md`: 当前已经实现的 RL 环境、状态、经验和训练接口。
- `rl/TRAINING_SPEED_OPTIMIZATION_PLAN.md`: 当前训练吞吐优化和 fast 物理模式说明。

## 建议阅读顺序

1. 先读 `project_map/PROJECT_FILE_INDEX.md`，了解项目有哪些文件、各自负责什么。
2. 需要继续训练或分析模型时，读 `training_runs/INDEX.md` 和目标实验的 `summary.md`。
3. 再读 `CODING_STYLE.md`，了解源码注释和后续修改风格。
4. 再读 `codex/` 下最新的修改记录，了解近期发生过哪些较大的结构或逻辑变化。
5. 需要继续修改项目时，按对应目录下的 `RULES.md` 更新文档。
