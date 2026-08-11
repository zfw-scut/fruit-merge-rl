# RL 导航

按数据流定位，避免通读 `trainer.py`：

```text
config.py -> observations.py / model.py -> replay.py -> learner.py -> trainer.py -> evaluation.py
```

| 修改目标 | 文件 |
| --- | --- |
| TOML 类型与校验 | `config.py` |
| 图状态构建 | `observations.py` |
| GNN、Q 值、辅助头 | `model.py` |
| loss、优化器、目标网络更新 | `learner.py` |
| GPU Replay | `replay.py` |
| epsilon、采集、训练编排 | `trainer.py` |
| checkpoint、恢复、weights-only | `checkpoint.py` |
| 正式评估与关键局复跑 | `evaluation.py`、`event_analysis.py`、`key_decisions.py` |
| 辅助动作标签与结构几何 | `action_effects.py`、`contact_geometry.py` |
| 训练事实采集 | `decision_data.py` |
| 面板与曲线 | `monitoring.py`、`curves.py`、`training_queue.py` |
| 模型观看/场景推理 | `viewer.py`、`scenario_model_*` |

## 阅读边界

- 现有 TOML 参数调整遵循 `configs/AGENTS.md`，通常不读本目录其它文件。
- UI、曲线、日志字段修改不需要阅读模型或训练热路径。
- 模型输出变更要检查 learner、checkpoint、viewer 和相关测试；Replay 契约变更要检查 trainer 和 decision data。
- 新模型、新奖励、新训练方法、正式训练或有决策价值的评估才执行 `docs/AGENTS.md` 的完整证据流程。
- `runs/` 只在用户要求检查具体 run 时读取，并从 manifest、配置或摘要开始，不扫描全部产物。

## 最小测试

| 范围 | 测试 |
| --- | --- |
| 基线模型、配置、Replay、训练、面板 | `tests.test_rl_baseline` |
| 辅助动作学习 | `tests.test_action_effect_learning` |
| 训练事实采集 | `tests.test_decision_data` |
| 场景模型推理 | `tests.test_scenario_lab` |

先运行相关测试类或单个测试，再按改动风险扩大；不要默认启动训练或全量评估。
