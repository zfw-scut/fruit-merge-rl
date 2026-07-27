# Project Documentation

本目录用于存放项目文档，方便后续开发、复盘和让新的 agent 快速理解项目。

## 目录说明

- `codex/`: 记录 Codex 对项目做出的较大修改，以及修改记录的维护规则。
- `project_map/`: 记录项目文件构成、职责分工、可复用组件和阅读入口。
- `training_runs/`: 记录本地与云端训练实验的轻量摘要、配置、指标统计和产物索引，供迁移后的开发者或 agent 分析。
- `CODING_STYLE.md`: 记录项目源码注释和编码风格，当前强调教学型详细注释。
- `rl/`: 存放后续强化学习模型、环境接口和算法方案的设计参考文档。

## 当前 RL 开发入口

- `rl/STRUCTURE_AWARE_GNN_V2.md`: 当前结构感知模型和首次新架构长训的主规格；
  说明显式关系、连锁 motif、关系 GNN、六维结构监督、集中式 GPU actor、无未来
  泄漏证明以及旧 checkpoint/replay 边界。
- `rl/CAUSAL_ATTRIBUTION_V1.md`: 第一次大规模训练采用的 Reward V2、完整状态归因、
  因果 Q 排序、预算反事实和局部 Shapley 规格；它继续定义归因语义，模型输入与新
  训练重置以 V2 文档为准。
- `rl/INTERFACE_V0.md`: 当前已经实现的 RL 环境、状态、快照、经验、因果训练、
  checkpoint 与恢复接口。
- `rl/FIRST_500K_RUNBOOK.md`: 云端安装、全量测试、preflight、5k/10k/500k
  阶段门禁、监控、停止和 checkpoint 恢复手册。
- `rl/TRAINING_SPEED_OPTIMIZATION_PLAN.md`: 当前训练吞吐优化和 fast 物理模式说明。
- `training_runs/FIRST_500K_READINESS.md`: 第一次 500k 的唯一就绪证据表；区分
  已完成检查、运行中阶段和待运行阶段，不能用配置存在或中间 CSV 替代结论。

结构感知训练代码已经把完整状态分析投影到主模型图：水果获得可达、伙伴、支撑、
遮挡和空腔信息，`merge_pair` / `level_ladder` 作为 motif 节点连接成员水果、可见
队列和每个动作；关系感知 GNN 同时输出 dueling Q 与六维一步结构预测。结构 loss
只训练共享表示，不修改 Reward V2，也不把动作后状态泄漏到动作前输入。并行采集可把
greedy 请求集中到主进程 GPU actor 做微批推理。

完整状态归因的训练代码也继续贯通：环境使用 Reward V2；采集侧生成 3-step return、
规则因果样本和有界反事实 proposal；更新器把 TD、结构辅助、规则 Q 排序与
反事实/局部 Shapley 差值监督联合训练。`EngineSnapshot` 保存并校验完整 Pymunk、
队列、RNG 和 episode 状态，反事实只在原动作可复现后生成标签。
复现门禁区分严格一致、仅连续数值抖动和物理语义分歧；后两类均丢弃标签，
但只有语义分歧进入独立 1% 比率门禁。
`CounterfactualCoordinator` 与 `LocalShapleyCoordinator` 共用软预算和 10% 硬账本；
普通反事实最多借用到 9%，最后 1% 为已选中的局部 Shapley 保留。队列或预算不足时
不阻塞 rollout，并对 selected 后未执行的任务做独立终态记账。

训练 checkpoint 已版本化并采用原子替换，保存 online/target/optimizer、更新计数、
Python/PyTorch/CUDA RNG、主 replay 和 `CausalReplayBuffer` 状态及运行/配置指纹。
hybrid 主 replay 恢复时只恢复有界热层，并在 sidecar 中明确记录为 hot-resume，
不会把未保存的冷层宣称为精确恢复。

新架构第一次正式长训前的三个阶段分别由以下配置承载，全部从 update 0 开始：

1. `configs/train_dqn_causal_smoke_5k.toml`：5000 更新烟测；
2. `configs/train_dqn_causal_calibration_10k.toml`：10000 更新标定，必要时从
   update 0 另起独立 25000 更新校准；连续延长虽可保持冻结的 epsilon horizon，
   但必须作为不同证据单独标记；
3. `configs/train_dqn_causal_500k.toml`：500000 更新正式配置。

`tools/preflight_training.py` 是启动前门禁，覆盖结构感知正式配置、CUDA、依赖版本、
六维结构与完整因果 optimizer step、`EngineSnapshot` 重演、局部 Shapley 物理链路、
磁盘和 CPU 余量。配置和门禁已经存在不等于对应烟测/标定已经完成；实际结果应在
运行后写入 `training_runs/`，本文不预先声明结果。旧 V1 的 5k/10k 是历史证据，
不能批准结构感知 V2 的 500k。

## 建议阅读顺序

1. 先读 `project_map/PROJECT_FILE_INDEX.md`，了解项目有哪些文件、各自负责什么。
2. 再读 `rl/STRUCTURE_AWARE_GNN_V2.md`，确认当前模型、训练开关和兼容边界。
3. 需要继续训练或分析模型时，读 `training_runs/FIRST_500K_READINESS.md`、
   `training_runs/INDEX.md` 和目标实验的 `summary.md`。
4. 再读 `CODING_STYLE.md`，了解源码注释和后续修改风格。
5. 再读 `codex/` 下最新的修改记录，了解近期发生过哪些较大的结构或逻辑变化。
6. 需要继续修改项目时，按对应目录下的 `RULES.md` 更新文档。
