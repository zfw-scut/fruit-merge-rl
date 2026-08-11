# 模型文档时效性

本目录混合了已实现契约、已结束实验和未来设计。默认只选与任务直接对应的一份文档；参数和当前实现以代码、TOML、测试为准。

| 类别 | 文档 | 含义 |
| --- | --- | --- |
| 当前实现基础 | `ACTION_CONDITIONED_EFFECT_VIEW.md`、`AUXILIARY_ACTION_EFFECT_LEARNING.md`、`TRAINING_DATA_EXTENSION_ARCHITECTURE.md` | 部分或全部已实现；最新效果仍看评估报告 |
| 稳定基础规格 | `GNN_DQN_BASELINE.md`、`GNN_DQN_TRAINING_SYSTEM.md`、`LOCAL_MODEL_VIEWER.md` | 解释基础契约，不代表当前最佳模型、默认配置或训练规模 |
| 已结束实验 | `REWARD_V2_ACCESSIBLE_SPACE.md`、`FIVE_LAYER_EPSILON_ABLATION.md` | 保留实验结论，不作为当前默认方案 |
| 未来设计 | `STRATEGIC_VIEW_CONNECTIVITY.md`、`STRATEGIC_LEVEL_AGGREGATION.md`、`STRATEGIC_REGION_ANCHORS.md`、`HIGH_LEVEL_PLAN_LIFECYCLE.md` | 尚未实现，不得用于描述当前接口 |
| 历史总览 | `MODEL_DESIGN_STATUS_OVERVIEW.md` | 2026-08-09 状态快照；只用于导航，不作为最新进度或效果来源 |

若文档内出现“当前基准、推荐配置、等待下一轮”等说法，先核对 `../README.md`、目标配置和 `../model_evaluations/` 的最新同谱系报告。不要为修改一个现有参数阅读所有设计文档。
