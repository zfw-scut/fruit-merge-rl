# 第一次 500k 完整因果训练就绪记录

状态：云端正式 5k/10k 均已通过；500k 未启动，磁盘门禁未通过

最后更新：2026-07-27

运行手册：`../rl/FIRST_500K_RUNBOOK.md`

## 1. 当前结论

训练前实现已经完成并通过本地、云端各 312 项全量回归。云端提交 `368cc9b` 的正式
500k 配置 preflight 返回 `ready=true`、无 warning，随后从 update 0 运行的全新
5k v5 已正常完成并通过 readiness schema v2。该运行同时覆盖 strict / numeric
jitter / semantic divergence 三态物理复现门、共享预算、普通反事实、局部 Shapley、
checkpoint 恢复和资源收尾。

独立 10k 于 2026-07-27 22:21:56（Asia/Shanghai）在同一提交上从 update 0
启动，并于 23:28:08 正常完成。它获得 6 个 strict Shapley 结果和 18 个样本，
同时真实覆盖 numeric jitter 与 semantic divergence，不需要另起 25k。

5k 后云盘剩余约 70 GiB；10k 采用 65 GiB 最低启动、45 GiB 优雅停止、40 GiB
硬停止的临时短跑门禁并最终剩余 53.018 GiB。短跑本身通过，但这不能替代 500k
启动前至少 80 GiB 的正式磁盘门槛。当前唯一已知启动阻塞是存储余量；未获得新的
历史产物删除/迁移授权前，不进行清理，也不启动 500k。

| 阶段 | 状态 | 结论 |
| --- | --- | --- |
| 源码与单元测试 | 已通过 | 本地与云端均为 312 项，0 failure / 0 error |
| 60-update 本地探针 | 已完成 | 仅作性能/闭环证据，不替代 5k |
| 干净 HEAD 20-update 探针 | 已完成 | 规则与反事实监督均实际进入 optimizer，不替代 5k |
| 正式配置 preflight | 云端已通过 | commit `368cc9b`，`ready=true`、32/32 快照、无 warning |
| 5k 冒烟 | 已通过 | 5000/5000；四个阶段退出码均为 0，readiness `ready=true` |
| 10k 校准 | 已通过 | 10000/10000；四个阶段退出码均为 0，readiness `ready=true` |
| 可选独立 25k 校准 | 不需要 | 10k 已有 6 个 strict Shapley 结果、18 个样本并进入 optimizer |
| 500k 正式训练 | 未启动、未获准 | 训练链路门禁通过；等待最终复验和磁盘恢复到 ≥80 GiB |

## 2. 冻结源码身份

| 项目 | 当前证据 |
| --- | --- |
| 分支 | `codex/work-1` |
| 训练源码冻结 commit | `368cc9bd554e237fd976786180afc112ad6f816e` |
| 云端短跑源码身份 | 分支 `codex/work-1`，commit `368cc9b`，工作树 clean |
| 全量测试 | 2026-07-27：本地 312/312、云端 312/312 通过 |
| 正式配置 | `configs/train_dqn_causal_500k.toml` |
| 正式配置 SHA-256 | `08c35479ddabd1cb383cd2287a94a457ae951b80acef483c3177e64b37ed2541` |
| 正式配置解析指纹 | `6cceea5075e97f62938c8f4fa70690ebaf50253ea22b1f131271815dcbc95a64` |

如果 5k/10k 后修改了任何训练语义，必须更新本节、重新跑全量测试和 32 次 preflight，
不能沿用旧证据。

## 3. 当前本地证据

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

## 4. Preflight 状态

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

## 5. 5k 冒烟记录

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

## 6. 10k 校准记录

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

## 7. 云服务器复验

状态：训练提交与 5k/10k 均已复验；最终证据提交不改变训练语义，500k
exact-commit preflight 仍受 80 GiB 磁盘门禁阻塞。

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

全部打勾前保持“未获准”：

- [x] 5k 正常完成，本文第 5 节所有硬门禁通过。
- [x] 10k 或获批的独立 25k 正常完成，第 6 节已给出结论。
- [x] 没有未解释的 NaN、死锁、snapshot/reproduction failure 或预算超限。
- [x] Reward shaping p95、StateAnalyzer degraded 和 truncated 比例符合运行手册。
- [x] 规则与物理反事实均真实进入 optimizer；Shapley 稀疏状态已解释。
- [x] 正式配置没有在短跑后被无记录修改。
- [ ] 最终 commit 已提交，工作树 clean。
- [ ] 云服务器重新完成依赖、CUDA、全量测试和 32 次 preflight。
- [ ] 云端正式输出盘可用空间恢复到至少 80 GiB。
- [x] 正式输出目录为空，不会覆盖历史 run。
- [ ] 旁路资源监控已经启动。
- [ ] checkpoint 恢复命令和训练产物备份位置已确认。

批准状态：**未获准；当前阻塞为云盘只剩 53.018 GiB**

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
