# 训练配置导航

只读目标 TOML 和 `src/daxigua/rl/config.py` 中对应 dataclass；不要默认阅读训练器或历史报告。

| TOML 段 | 权威类型 |
| --- | --- |
| `training` | `TrainingConfig` |
| `model` | `ModelConfig` |
| `dqn` | `DqnConfig` |
| `reward` | `RewardConfig` |
| `replay` | `ReplayConfig` |
| `branch_learning` | `BranchLearningConfig` |
| `evaluation` | `EvaluationConfig` |
| `analysis` | `AnalysisExportConfig` |
| `decision_data` | `DecisionDataConfig` |
| `dashboard` | `DashboardConfig` |
| `autoscale` | `AutoScaleConfig` |

## 修改规则

- 命令行参数会在 `tools/train_gnn_dqn.py:resolve_config` 覆盖 TOML；只在怀疑覆盖关系时读取该函数。
- 改现有字段值：检查对应 dataclass 约束，并解析一次配置即可，不扩展阅读。
- 改 `[model]` 结构字段可能导致 checkpoint 不兼容；改奖励语义、训练算法或模型结构才属于研究方案变更。
- `run_dir`、预算、日志、面板、评估频率等运行参数不是策略提升证据。
- 不改用户未指定的其它实验配置，不为“保持一致”批量同步历史 TOML。

最小验证：

```powershell
$env:PYTHONPATH='src'
conda run -n python-torch python -c "from daxigua.rl.config import TrainingConfig; TrainingConfig.from_toml('configs/<目标>.toml')"
```

若仅改 TOML，通常无需启动训练。
