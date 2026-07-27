# 第一次 500k 完整因果训练就绪记录

状态：结构感知 V2 的云端 preflight 与一次用户授权的独立 10k 已通过；正式
5k→10k 顺序链仍不完整，500k 未获准

最后更新：2026-07-28

运行手册：`../rl/FIRST_500K_RUNBOOK.md`

## 1. 当前结论

2026-07-28 引入的结构感知 V2 改变了主模型训练语义：`StateAnalysis` 的可达、
支撑、遮挡、空腔和 C/R/K 被投影进图，`merge_pair` / `level_ladder` 成为显式
motif，GNN 使用 relation gate + attention 和 dueling 读出，同时新增六维一步
结构辅助监督与集中式 GPU actor。正式基线也从 H128/L3、3 rollout、batch 64
调整为 H256/L4、16 rollout、batch 128。

因此下文 2026-07-27 的 312 项测试、preflight、5k 和 10k 只保留为 **V1 历史
证据**。它们证明旧版完整归因/反事实/Shapley 链路曾闭合，但不能批准 V2 的
500k，也不能提供可续训的 checkpoint 或 replay。V2 必须在最终 commit 上重新执行
全量测试、preflight、全新 5k 和全新 10k，全部从 update 0 开始。

新方案的设计、无未来泄漏/非奖励边界、参数和兼容表见
[`../rl/STRUCTURE_AWARE_GNN_V2.md`](../rl/STRUCTURE_AWARE_GNN_V2.md)。

| 阶段 | 状态 | 结论 |
| --- | --- | --- |
| V2 源码与单元测试 | 本地及新服务器 364/364 通过 | 训练源码 commit `ddcb249`；云端 compileall 同时通过，运行前工作树 clean |
| V2 正式配置 preflight | 新服务器通过 | RTX 5090 / 25 核 / 90 GiB 容器上 `ready=true`、无 required failure/warning |
| V2 5k 冒烟 | 未启动 | 全新空目录、update 0；旧 V1 5k 不替代 |
| V2 独立 10k 测试 | 已通过自身 10k 门禁 | 用户明确授权直接从 update 0 运行；`ready=true`，但它没有发生在 V2 5k 之后，不能单独补齐手册的正式顺序链 |
| V2 可选独立 25k 校准 | 待定 | 只在 10k 稀疏证据不足时决定 |
| V2 500k 正式训练 | 未启动、未获准 | 至少仍需处理缺失的正式 V2 5k 顺序证据，并在当前最终提交上重跑最终 preflight |

## 2. V2 冻结源码身份

本节旧值已失效，必须由 V2 最终提交和新 preflight 重新填写：

| 项目 | V2 当前证据 |
| --- | --- |
| 分支 | `codex/work-1` |
| 目标 GPU | NVIDIA GeForce RTX 5090，32,607 MiB |
| 目标训练环境 | Python 3.11.15、PyTorch `2.12.1+cu130`、CUDA runtime 13.0 |
| 冻结训练 commit | `ddcb249d0b5510700c87c6eec670a188a8400e5a` |
| 工作树 clean | preflight 与 10k 启动时均为 clean |
| 全量测试 | 2026-07-28：本地及新服务器 364/364 通过；compileall、diff check 通过 |
| 正式配置 SHA-256 | `08c35479ddabd1cb383cd2287a94a457ae951b80acef483c3177e64b37ed2541` |
| 正式配置解析指纹 | `3530d4aea2873fa3eacbc1d9622de949a7069042c3b12d30057cc0ebb6a7bd913` |

以下是 V1 历史身份，仅用于解释旧短跑：

| 项目 | 当前证据 |
| --- | --- |
| 分支 | `codex/work-1` |
| 训练源码冻结 commit | `368cc9bd554e237fd976786180afc112ad6f816e` |
| 云端短跑源码身份 | 分支 `codex/work-1`，commit `368cc9b`，工作树 clean |
| 全量测试 | 2026-07-27：本地 312/312、云端 312/312 通过 |
| 正式配置 | `configs/train_dqn_causal_500k.toml` |
| 正式配置 SHA-256 | `08c35479ddabd1cb383cd2287a94a457ae951b80acef483c3177e64b37ed2541` |
| 正式配置解析指纹 | `6cceea5075e97f62938c8f4fa70690ebaf50253ea22b1f131271815dcbc95a64` |

V2 正是一次训练语义修改，因此已经触发证据重置。以后若 V2 5k/10k 后再次修改图
schema、模型、Reward、loss、采集或物理语义，仍必须再次更新本节并重跑门禁。

提交前的本地正式配置探针还得到以下功能证据：

- RTX 4070 Laptop GPU 上 H256/L4 模型 Q 输出为 15，结构输出为 15×6；
- 16 个 rollout worker 完成两次模型同步和 32 个 greedy actor 请求，合并成
  2 个 16 图 GPU batch，全部 worker/actor/queue 干净关闭；
- batch 128 完整因果 optimizer step 消费 768/768 个结构维度，结构与反事实 loss
  均非零，CUDA 峰值约 2.84 GiB；
- 局部 Shapley 与 EngineSnapshot 原动作复现通过。

该次本地报告的总 `ready=false` 是预期结果：执行时工作树尚未提交，且本机只有 16
个有效 CPU 核，无法同时为 16 个 rollout worker 和 6 个反事实 worker 保留资源。
它只证明功能链路，不批准云端短跑；最终 commit 后仍须在 RTX 5090 服务器重新运行
完整 preflight。

## 2.1 V2 新服务器 preflight 与独立 10k 记录

本次运行由用户明确授权直接执行 10k，用于验证结构感知新模型和新服务器性能。
它使用空目录、从 update 0 开始，没有加载旧 checkpoint/replay；但此前没有先运行
V2 5k，因此本节只登记为“独立 10k 技术证据”，不把它改写成已经完成正式
5k→10k 顺序门禁。

| 项目 | 记录 |
| --- | --- |
| 训练源码 | `ddcb249d0b5510700c87c6eec670a188a8400e5a`，分支 `codex/work-1`，启动时 clean |
| 云端正式 preflight | `runs/preflight/first_500k_pre_10k_ddcb249.json`，`ready=true`、`required_failures=[]`、`warnings=[]` |
| 配置 | `configs/train_dqn_causal_calibration_10k.toml`；H256/L4、batch 128、16 rollout、6 物理 worker、集中式 GPU actor |
| 输出目录 | `runs/dqn_structure_v2_calibration_10k/` |
| 运行时间 | 2026-07-28 05:10:31 → 06:05:56（55 分 25 秒） |
| control / monitor | `runs/stage_control/v2_10k_ddcb249_20260728T0510/` / `runs/resource_monitor/v2_10k_ddcb249_20260728T0510/` |
| 退出与进度 | 退出码 0；`10000 / 10000` updates，`162500` env steps |
| 阶段 readiness | `runs/preflight/dqn_structure_v2_calibration_10k_readiness.json`，`ready=true`、`required_failures=[]` |
| 本地轻量证据 | `runs/cloud_evidence/v2_calibration_10k/`；完整模式 9 个训练文件，另含 readiness 与资源监控 sidecar |
| 最终 checkpoint | `checkpoints/latest.pt`，3,395,338,159 bytes；模型、target、optimizer、RNG、hot replay 与 causal replay 恢复检查通过 |

核心训练结果：

| 校准项 | 结果 |
| --- | --- |
| episode / truncated | 1175 局；4/1175 = 0.3404%，低于 2% |
| 贪心评估 | update 5000：均分 560.2、最高 615；update 10000：均分 577.6、最高 646；`best_eval=646` |
| 得分趋势 | 训练局分平滑均值由约 350 上升到约 640；仍有较大单局方差，10k 只证明已出现学习趋势，不代表收敛 |
| 主 TD 数值 | 最终 loss/TD loss = 3.2224/3.1088；全程 `|Q|/|target|/TD MAE` 最大 49.548/49.280/5.122，无连续越过 100 |
| 结构辅助 | 51/51 个日志窗口进入 optimizer；每批 128 个样本、768/768 有效维；结构 loss 从约 0.028 降至最终 0.0119，MAE 从约 0.160 降至 0.0786 |
| 集中式 actor | 累计 105,682 请求、16,278 批；最大 batch 16、最终平均 batch 6.49，无异常重置 |
| 普通反事实 | 501 个 label-ready 结果、907 个样本；strict/numeric/semantic = 502/1/2，语义分歧率 0.3960% < 1% |
| 局部 Shapley | observed/selected/completed/strict/samples = 23729/11/11/11/32；全部选择均闭合并进入 optimizer |
| 因果 replay | 最终 20,000：rule 19,169、counterfactual 799、Shapley 32；正/负规则 9,585/9,584 |
| 共享预算 | 所有窗口 hard budget 为 true；实际/投影最大均 9.1178% < 10% |
| Reward / analyzer | shaping p95 最大 0.05488 < 0.707；StateAnalyzer degraded 最大 0.03102% < 1% |
| 吞吐 | 最终累计 3.047 updates/s、49.516 env steps/s；总墙钟 55 分 25 秒 |

资源 sidecar 共 1110 个 3 秒样本。容器 CPU 配额为 25 核，训练进程平均使用
13.39 核、峰值 17.27 核；GPU utilization 平均 37.15%、峰值 74%。显存峰值
31,417 MiB（96.35%）只出现单个样本，连续超过 95% 的最大长度为 1，随后恢复，
没有 OOM。cgroup 内存上限 90 GiB，working set 峰值 22.24 GiB，
`memory.current` 峰值 36.76 GiB，有效可用内存最低 67.76 GiB；swap 和
low/high/max/OOM/OOM-kill 事件增量全为 0。

readiness 的两个非 required warning 已解释：

- `counterfactual_candidate_close_dropped`：正常关闭时清空跨窗口 top-K 候选
  256/23718 = 1.0793%；queued/inflight/token reservation 最终均为 0。
- `replay_cold_storage_growth`：当前一代 89 个连续 cold segment 共 17.176 GiB，
  高于 16 GiB 提醒线但不破坏恢复语义；正式 500k 前必须单独核算磁盘增长与清理
  策略，不能把 warning 当成已解决。

结论：结构输入、六维辅助监督、集中 actor、完整归因、预算反事实、Shapley、
checkpoint 和资源稳定性均已在一次完整 10k 中闭合，且得分有明确上升趋势。该结果
足以证明 V2 技术链路可以继续扩展，但按当前运行手册，缺失的 V2 5k 顺序证据仍需
由用户明确决定是补跑，还是书面批准以本次更大规模独立 10k 替代；在此决定和最终
preflight 完成前，不自动批准 500k。

## 3. V1 历史本地证据

已完成的有效小规模运行：

| 项目 | 记录 |
| --- | --- |
| run | `runs/dqn_causal_perf_probe_60_v4/` |
| 规模 | 60 updates，256 warmup |
| 结果 | 完成；主 TD、规则与物理反事实闭环曾产生有效产物 |
| 用途 | 性能和完整链路探针 |
| 限制 | 只有 2 局 episode；产物生成于 readiness/replay/warmup 新 schema 之前，不能用当前门禁直接批准扩容 |

最终收尾另执行了 `runs/dqn_causal_release_cleanup_probe_4/`：4 updates、256
warmup、单 rollout worker、2 个物理 worker 的低内存诊断覆盖了最终 checkpoint、
hardlink 周期 checkpoint、资源清理和失败指针生命周期。它完成 1 个可复现反事实
结果、插入 1 个样本并实际进入 counterfactual loss；共享 token 比例约 0.77%，
`failure_latest.json` 不存在。用 `baseline_config=None` 按 4-update 诊断规模调用
readiness 后 `ready=true`，只有 episode 样本不足和关闭时清空 top-K 候选池两项预期
warning。该 run 使用诊断覆盖参数，仍然不替代正式 5k。

随后在干净 commit `474cf453c559336b2e2fd8c18c994a74cd016fb2` 上执行
`runs/dqn_causal_head474cf45_probe_20/`：20/20 updates、320 warmup、正常退出，
`failure_latest.json` 不存在；主 TD 在全部 5 个日志窗口更新，规则 batch 与物理
反事实 batch 各在 2 个窗口进入同一次 optimizer，最大 rule batch 为 31、最大
counterfactual batch 为 1。反事实累计完成 1、原动作复现通过 1、样本插入 1，
失败 0，实际 token 比例最高约 6.23%，没有突破 10% 硬上限。最终 update 20
checkpoint 的 online/target 模型、Adam 与 CPU/CUDA RNG 均通过真实恢复检查。

对应诊断 readiness 报告位于
`runs/preflight/dqn_causal_head474cf45_probe_20_readiness.json`。它只因本探针主动把
`num_envs`、物理 worker、采样率和训练长度缩小而触发
`baseline_training_semantics_match`，拒绝把诊断 run 冒充正式 5k；另有 episode
样本不足和关闭时丢弃未调度候选两项预期 warning。这一失败是门禁正确工作的证据，
不是训练链路失败。

旧的 245/286 项测试和早期 commit 只属于此前历史版本，不得把它们登记为本轮
结果。本轮训练语义提交的最终全量回归为 312 项全部通过，身份见第 2 节。

## 4. V1 历史 Preflight 状态

| 检查 | 当前结果 | 说明 |
| --- | --- | --- |
| 报告 | 已通过 | `runs/preflight/first_500k_pre_368cc9b.json` |
| 源码身份 | 已通过 | branch `codex/work-1`，commit `368cc9b`，`dirty=false` |
| 总结 | 已通过 | `ready=true`、`required_failures=[]`、`warnings=[]` |
| 正式 500k 配置契约 | 已通过 | 500000 updates、CUDA、3 env、9 steps/update、完整 CF/Shapley |
| CUDA 前后向 | 已通过 | Torch `2.12.1+cu126`，Tesla V100-PCIE-32GB |
| 完整因果 optimizer | 已通过 | TD、规则与物理反事实同一次真实 optimizer step |
| 局部 Shapley 物理实现 | 已通过 | 4 个 subset、8 个 permutation、效率残差在门内 |
| 快照复演 | 已通过 | 32/32，match rate 1.0 |
| CPU | 已通过 | cgroup/affinity 后有效 6 核；3 rollout + 2 物理 worker 配置闭合 |
| 内存 | 已通过 | cgroup 上限 25 GiB；reclaim-aware 有效可用 21.52 GiB |
| 磁盘 | 当时通过 | 80.608 GiB 可用，高于 80 GiB 门槛 |

上述报告批准了同提交的 5k 启动。5k 产物落盘后只剩约 70 GiB，因此它不能直接充当
500k 最终启动证据。10k 和文档提交完成后必须重新运行正式 preflight；在可用磁盘
恢复到至少 80 GiB 之前，`output_disk_free` 应按失败处理，不能通过降低正式阈值绕过。

## 5. V1 历史 5k 冒烟记录

状态：**已通过**

| 项目 | 当前记录 |
| --- | --- |
| 配置 | `configs/train_dqn_causal_smoke_5k.toml` |
| 输出目录 | `runs/dqn_causal_smoke_5k_v5/` |
| 启动/完成时间 | 2026-07-27 21:39:58 → 22:13:21（33 分 23 秒） |
| 源码 | `368cc9bd554e237fd976786180afc112ad6f816e`，云端工作树 clean |
| 退出码 | train / monitor / cgroup monitor / readiness = `0 / 0 / 0 / 0` |
| 最终 update/env steps | `5000 / 46000` |
| launcher log | `runs/launcher_logs/train_20260727_213959.log` |
| resource monitor | `runs/resource_monitor/dqn_causal_smoke_5k_v5_20260727T133958Z/` |
| readiness | `runs/preflight/dqn_causal_smoke_5k_v5_readiness.json`，schema 2，`ready=true` |
| 最终 checkpoint | `checkpoints/latest.pt`，2,013,651,214 bytes |
| 最终结论 | 5k 所有 required checks 通过；允许进入独立 10k |

仓库中的 `runs/dqn_causal_smoke_5k/` 是旧实现下中止的探针，只到 update 100，
没有完成 checkpoint，也没有可证明正常/异常收尾的 failure marker；现标记为废弃、
非门禁证据。有效证据仅来自全新目录 `dqn_causal_smoke_5k_v5`，没有续跑或覆盖。

| 门禁 | 通过标准 | 结果 |
| --- | --- | --- |
| 训练完成 | 5000 updates，正常退出 | 通过：5000/5000，env steps 46000 |
| 数值稳定 | 无 NaN/Inf/failure sidecar | 通过；`|Q|/|target|/TD MAE` 最大 22.683/23.880/6.758 |
| 主 TD | TD optimizer 持续更新 | 通过 |
| 规则监督 | 正负规则样本均存在，rule batch 实际进入 loss | 通过；最终正/负规则各 9882，最大 rule batch 24 |
| 反事实监督 | strict 结果生成样本并进入 loss | 通过；147 strict、255 samples、最大 CF batch 11 |
| 三态复现 | semantic < 1%，numeric 独立报告，unknown=0 | 通过；strict/numeric/semantic = 147/0/1，semantic 0.6757% |
| 共享预算 | hard budget 始终为 true，实际比例不超过 0.10 | 通过；max actual 8.9842%，max projected 9.0% |
| 快照 | snapshot failure 为 0 | 通过：0 |
| StateAnalyzer | degraded rate 与 shaping p95 符合阈值 | 通过：最大 0.1103% / 0.05882 |
| episode | truncated rate < 2% | 通过：2/376 = 0.5319% |
| 并行稳定性 | 无死锁、worker 异常、ID 串线 | 通过；退出后无残留训练/监控进程 |
| 产物完整 | checkpoint、CSV、曲线、shutdown sidecar 齐全 | 通过；checkpoint 真实恢复、RNG/manifest/replay 校验均通过 |
| Shapley | selected 全部产生 strict 结果、样本并进入 optimizer | 通过：observed/selected/completed/strict/samples = 6383/3/3/3/9 |

反事实唯一失败为 `original_reproduction_mismatch`，同时包含
`physics_result`、`final_state_semantics` 和 `fruit_semantics` 分叉，因此正确归入
`semantic_divergence_drop`，没有生成标签。numeric jitter 为 0，五类 numeric 最大
误差均为 0，`unknown_failed=0`、`non_gate_result_failed=0`。

readiness 唯一非 required warning 是正常关闭时清空跨窗口 top-K reservoir：
`candidate_close_dropped=256/6380=4.0125%`。关闭时 queued/inflight/reserved/
external-active/pending/candidate-pool 全为 0；因此没有已获批任务或 token 丢失。

资源监控共 656 个样本、1965 秒：CPU 平均 4.23/6 核、p95 5.32、峰值 5.84；
GPU utilization 平均 10.41%、p95 27%、峰值 32%，显存峰值 4166 MiB；target RSS
峰值 8381 MiB，cgroup working set 峰值 8.574 GiB、`memory.current` 峰值
16.273 GiB、有效可用内存最低 16.426 GiB。swap 与 low/high/max/OOM/OOM-kill
事件增量均为 0。训练瓶颈主要在 CPU 物理模拟，不在 GPU。

## 6. V1 历史 10k 校准记录

状态：**已通过**

前置条件：5k 冒烟全部硬门禁通过。

| 项目 | 记录 |
| --- | --- |
| 配置 | `configs/train_dqn_causal_calibration_10k.toml` |
| 输出目录 | `runs/dqn_causal_calibration_10k/` |
| 启动/完成时间 | 2026-07-27 22:21:56 → 23:28:08（66 分 12 秒） |
| commit | `368cc9bd554e237fd976786180afc112ad6f816e` |
| 启动方式 | 独立空目录，从 update 0 开始；没有 `--resume`/`--overwrite-run-dir` |
| control | `runs/cloud_stage_control/10k_368cc9b/` |
| resource monitor | `runs/resource_monitor/dqn_causal_calibration_10k_20260727T142156Z/` |
| 启动时磁盘 | 75,144,835,072 bytes = 69.984 GiB |
| 四个退出码 | train / monitor / cgroup / readiness = `0 / 0 / 0 / 0` |
| readiness | `runs/preflight/dqn_causal_calibration_10k_readiness.json`，schema 2，`ready=true` |
| 最终 update/env steps | `10000 / 92500` |
| 是否另起独立 25k | 不需要；正式配额已获得 6 个 strict Shapley 结果、18 个样本 |
| 最终 checkpoint | `checkpoints/latest.pt`，2,040,798,831 bytes |
| 最终结论 | 10k 所有 required checks 通过；训练链路允许进入最终 500k 启动准备 |

最终普通反事实三态为 strict / numeric jitter / semantic divergence =
283 / 1 / 1，总评估 285，semantic rate 0.3509%。`label_ready=283`，插入
522 个样本；`unknown_failed=0`、`non_gate_result_failed=0`。

numeric 记录的 merge-event position / fruit position / linear velocity /
orientation / angular velocity 最大误差为 0 / 0.0017626 / 0.0013093 /
0.0028120 / 0.0000536。该记录只新增 `fruit_angle` 与 `fruit_velocity` 诊断，
没有新增任何 `*_semantics` 或 `physics_result` 代码。semantic 记录则明确包含
`physics_result`、`final_state_semantics` 和 `fruit_semantics`。两类 drop
都没有生成标签，三态账本按设计闭合。

结束后补齐：

| 校准项 | 结果 | 结论 |
| --- | --- | --- |
| Reward task 与 shaping 尺度 | shaping p95 最大 0.05547 < 0.707 | 通过 |
| TD/rule/counterfactual loss 数量级 | 三者均进入 optimizer；最大 rule/CF batch 22/11；`|Q|/|target|/TD MAE` 最大 42.846/43.357/6.650 | 通过 |
| 正负因果样本 | 最终正/负 rule 各 9751；CF 480；Shapley 18；无效样本 0 | 通过 |
| 归因 worker 收尾 | 8 个 pending 全部记录为 `worker_shutdown`；n-step flush 6 | 通过 |
| 反事实 proposal/admit/result/sample | 12979 / 285 / 285 / 522；283 strict、两类 drop 各 1 | 通过 |
| 反事实 token 实际/投影比例 | 最终 8.9859%；全窗口最大均 9.0501% < 10% | 通过 |
| Shapley observed/selected/result/sample | 12985 / 6 / 6 strict / 18；failed/terminal drop 为 0 | 通过 |
| episode truncated ratio | 6/765 = 0.7843% < 2% | 通过 |
| 吞吐、CPU、内存、GPU | CPU 平均 4.38/6 核；working set 峰值 9.825 GiB；GPU 平均 10.34%、显存峰值 4118 MiB | 通过 |
| checkpoint 与 cold replay | latest 2.041 GB；模型/optimizer/RNG/replay 恢复通过；83 个连续 cold segment、11.186 GiB | 通过 |

本轮不另起 25k。正式 0.05% 配额已经选择并完成 6 个 Shapley 任务，18 个样本真实
进入 optimizer；普通反事实的 strict、numeric 与 semantic 三态也已全部出现。
增加 25k 不再用于补足缺失的链路证据，只会额外消耗当前紧张的磁盘空间。

## 7. V1 历史云服务器复验

状态：以下只记录旧 V100 服务器和 V1 训练，不代表当前 V2 新服务器门禁。

| 检查 | 云端结果 |
| --- | --- |
| GPU/Torch CUDA | Tesla V100-PCIE-32GB；Torch `2.12.1+cu126`，CUDA runtime 12.6 |
| Python/Pymunk/Chipmunk | Python 3.11.15；Pymunk 7.3.0；Chipmunk `2.0.1-ade7ed7...` |
| 当前提交全量 tests | 312/312 通过，随后 `compileall` 通过 |
| 500k config 32 次 preflight | commit `368cc9b`：`ready=true`、32/32、无 warning |
| affinity/cgroup 有效 CPU | 6 核；冻结配置为 3 rollout + 2 共享物理 worker |
| cgroup 内存 | 上限 25 GiB；5k working set 峰值 8.574 GiB，OOM 事件增量 0 |
| 磁盘 | preflight 时 80.608 GiB；5k 后约 70 GiB，最终 500k 前必须恢复到 ≥80 GiB |
| 5k launcher/resource monitor | `train_20260727_213959.log` / `dqn_causal_smoke_5k_v5_20260727T133958Z` |
| 10k launcher/resource monitor | `train_20260727_222157.log` / `dqn_causal_calibration_10k_20260727T142156Z` |
| 10k 资源结论 | CPU 平均 4.38/6 核；working set 峰值 9.825 GiB；显存峰值 4118 MiB；所有 memory event 增量为 0 |

如果云端依赖版本、GPU 数量或训练语义与本地不同，checkpoint 的严格恢复可能拒绝；
不要通过放宽指纹校验绕过差异。

## 8. 500k 启动批准清单

以下全部针对 V2，旧 V1 的勾选不继承。全部打勾前保持“未获准”：

- [x] V2 冻结训练 commit 已提交，启动时工作树 clean，全量测试和 compile 零失败。
- [x] 新服务器完成依赖、CUDA、32 次快照和正式配置 preflight。
- [ ] 全新 V2 5k 正常完成，图 schema、结构 target、checkpoint/replay 均闭合。
- [x] 一次从 update 0 开始的独立 V2 10k 正常完成并给出标定结论；但它未经
  V2 5k 前置，正式顺序是否由该 10k 替代仍待明确批准。
- [x] TD/结构/规则/反事实 loss 无 NaN/Inf，结构有效 target 持续进入 optimizer。
- [x] 集中 actor 无 timeout/失败，request/batch/sync/关闭统计闭合。
- [x] H256/L4、batch 128、16 rollout、6 物理 worker 的吞吐、内存和 GPU 可承受
  500k。
- [x] Reward shaping、StateAnalyzer degraded、truncated 和物理复现率符合手册。
- [x] 规则与物理反事实真实进入 optimizer，Shapley 稀疏状态已解释，10% 硬预算
  未超限。
- [x] 正式配置没有在 V2 独立 10k 后被无记录修改。
- [ ] 云端输出盘满足最终 preflight 门槛。
- [ ] 正式输出目录为空，不会覆盖或读取任何旧 run/replay。
- [ ] 旁路资源监控已经启动。
- [ ] 同一 V2 run 的 checkpoint 恢复命令和训练产物备份位置已确认。

批准状态：**未获准；等待处理缺失的 V2 5k 顺序证据、重跑最终 preflight，并完成
正式输出/监控/恢复与备份确认**

批准人/时间：待填

正式启动 commit/config/preflight：待填

## 9. 已知边界

- 正式 hybrid TD replay 的 checkpoint 只保存 hot layer；恢复时 cold layer 不会伪装成
  已恢复。因果 replay 则精确恢复。
- worker 的物理轨迹和在途反事实/Shapley 任务不会跨进程恢复；续跑从新 episode
  边界开始。
- 物理反事实和 Shapley 只对 `strict_match` 生成标签。`numeric_jitter_drop` 与
  `semantic_divergence_drop` 都只进入诊断；前者独立记录五类最大误差，后者才消耗
  1% 语义分歧门槛。unknown 或非门禁失败必须为 0。
- 异常会生成带时间戳的 `failure_<时间戳>.json` 历史记录和活动指针
  `failure_latest.json`。成功恢复并完整保存最终 checkpoint/曲线后，入口只自动删除
  活动指针，不删除历史记录；readiness 对历史记录告警，对仍存在的活动指针硬失败，
  不需要人工删文件。
- 500k 的正式目标是首次完整方案测试，不是长期维护基线。短跑发现明确问题时应积极
  修正，但修正后必须重置本文证据链。
