# 新增 RewardBreakdown 训练日志

## 修改原因

当前环境已经在每次 `DaxiguaEnv.step()` 中返回 `reward_breakdown`，但训练日志只记录总 reward，无法判断模型主要被合成得分、高度惩罚、危险线惩罚还是终局惩罚驱动。后续调 reward 权重和分析延迟奖励问题，需要把各奖励项写入长期训练指标。

## 主要修改

- 新增统一的 reward breakdown 字段顺序，避免环境、采集器和训练日志之间字段不一致。
- `RolloutCollector` 在采集经验时累计每个 reward breakdown 字段。
- `train_dqn.py` 的 `metrics.csv` 新增各奖励项的窗口均值字段。
- 训练曲线生成时额外输出 `plots/reward_breakdown_curves.png`，用于单独观察奖励组成和高度比例变化。
- `collect_*` 指标改为按“距离上一行日志以来”的窗口汇总，降低 `collect_per_update=1` 时的单步噪声。

## 涉及文件

- `src/daxigua_rl/reward.py`: 增加 `REWARD_BREAKDOWN_FIELDS`。
- `src/daxigua_rl/training/collector.py`: 累计采集窗口内的 reward breakdown。
- `src/daxigua_rl/scripts/train_dqn.py`: 写入 reward breakdown CSV 字段，并生成独立曲线图。
- `tests/test_training_metrics.py`: 验证指标行包含 reward breakdown 均值。

## 验证方式

- 运行训练指标相关单元测试，确认新字段可以正常写入。

## 备注

- 当前改动只记录和展示 reward 组成，不改变 reward 公式，也不改变模型训练目标。
