# 历史入口：第一次长训目标已冻结为 250k

2026-07-28 起，第一次结构感知 V2 大规模训练的最终目标由 500k 调整为
250k updates。当前正式运行手册见
[`FIRST_250K_RUNBOOK.md`](FIRST_250K_RUNBOOK.md)。

`configs/train_dqn_causal_500k.toml` 仅保留为未来可选延长实验，不得用于当前
250k 正式门禁或启动。
