# 新增 AttributionTracker 因果归因器

## 修改原因

第一次大规模训练计划直接启用完整状态归因。Reward V2 和 StateAnalyzer 已能描述
即时任务价值与局面质量，但还不能把后续合成、连锁、解救或长期封路回溯到历史投放。

## 主要修改

- 新增 worker-local `AttributionTracker`，按真实 drop 和 ordered merge 建立水果
  谱系、根材料权重与每个合成唯一的价值包。
- 新增统一 `AttributionEvent`、贡献者、证据、结构来源和 step result 契约；事件
  保存归因版本与配置指纹，并验证动作身份、episode/worker 隔离和价值包不重复。
- 实现直接/机械/连锁触发、后续真实合成兑现的邻接/阶梯/墙锚/支撑/伙伴/营救事件，
  以及渐进通道损失、出生即埋、封路、空腔、桥接和终局支撑等负事件。
- 实现 3 个稳定边界的瞬态窗口、12 个不可达边界确认、部分通道恢复、进入合成谱系
  撤销，以及 terminal/truncated/reset/shutdown 收口。
- 环境保留 Reward V2 的 terminal next potential 为 0，同时额外生成只供归因使用的
  post-action terminal analysis，避免按动作前状态误判；诊断无效的终局几何只会
  以 `terminal_geometry_untrusted` 撤销待定证据，不会参与确认。
- 单/多进程 collector 在主 replay 写入前调用 tracker；完整事件和分析保持在 worker，
  主进程只接收耗时、事件状态、pending、谱系/连锁和延迟等轻量统计。
- 训练入口新增 attribution CSV 列、warmup 独立汇总和 shutdown pending sidecar；
  清理阶段即使 replay flush 失败也会继续 finalize worker 和关闭日志。

## 涉及文件

- `src/daxigua_rl/attribution/schema.py`: 归因事件、谱系、贡献者和结果不变量。
- `src/daxigua_rl/attribution/tracker.py`: 历史归因状态机。
- `src/daxigua_rl/env.py`: post-action terminal analysis 与物理结果证据。
- `src/daxigua_rl/training/collector.py`: 单 worker tracker 生命周期和统计。
- `src/daxigua_rl/training/parallel_collector.py`: 多 worker 汇总、pending gauge 和关闭审计。
- `src/daxigua_rl/scripts/train_dqn.py`: 训练指标、warmup/shutdown JSON 和可靠清理。
- `tests/attribution_fixtures.py`、`tests/test_attribution_tracker.py`: 确定性归因场景。
- `tests/test_training_metrics.py`、`tests/test_reward_v2_integration.py`: 指标和环境集成回归。
- `docs/rl/CAUSAL_ATTRIBUTION_V1.md`、`docs/rl/INTERFACE_V0.md`: 规格状态和公开接口。

## 验证方式

- `python -m compileall -q src tools tests`
- `python -m unittest discover -s tests -v`：116 项通过。
- 完成 1 次 warmup + 1 次 update 的最小训练 smoke test；成功生成并检查
  `metrics.csv`、`attribution_warmup.json`、`attribution_shutdown.json`、checkpoint
  和训练曲线，实际合成产生唯一 `MERGE_LINEAGE` 与共享预算的 `DIRECT_TRIGGER`。

## 备注

- 只有显式 `ContactInfluenceEdge` 才允许机械触发或接触型破坏归因。当前物理引擎
  尚未生产逐帧接触影响边，因此这些事件不会用静态几何伪造。
- `CORRIDOR_OPENED_USED` 等需要“真实经过新通道”的事件在路径证据落地前保持禁用。
- 本次完成实现顺序第 6 步；下一步是 `CausalReplayBuffer` 与规则 Q 排序 loss。
