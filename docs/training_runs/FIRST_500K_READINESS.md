# 第一次 500k 完整因果训练就绪记录

状态：本地训练前实现与验证已完成；正式 5k/10k 尚未运行，500k 未获准

最后更新：2026-07-27

运行手册：`../rl/FIRST_500K_RUNBOOK.md`

## 1. 当前结论

本地训练前实现已经完成并通过最终全量回归。60-update 性能/闭环探针证明完整因果
链路能在小规模闭合，但不是正式 5k 证据；正式 5k、10k 和 500k 均未启动。

本机总内存约 15.2 GiB，本轮可用内存实测约 0.5–2 GiB，不满足新版 preflight 的
32 GiB 总内存与 8 GiB 可用内存硬门禁，因此不得在本机继续正式 5k/10k。下一份有效
preflight 和两个短跑必须迁移到云服务器执行。云服务器配置入口已经识别，但尚未获得
用户对 root SSH 只读检查的明确授权，所以当前也没有云端环境证据。

| 阶段 | 状态 | 结论 |
| --- | --- | --- |
| 源码与单元测试 | 已通过 | 286 项测试，0 failure / 0 error |
| 60-update 本地探针 | 已完成 | 仅作性能/闭环证据，不替代 5k |
| 正式配置 preflight | 本机硬门禁失败，云端待运行 | 本机内存不足；旧 `ready=true` 已失效 |
| 5k 冒烟 | 未运行 | 必须在通过新版 preflight 的云服务器启动 |
| 10k 校准 | 待运行、待填 | 5k 通过后启动 |
| 可选独立 25k 校准 | 未决定 | 从 update 0 新启动；只在 Shapley 样本不足且其它门禁正常时使用 |
| 500k 正式训练 | 未启动、未获准 | 等待 5k 与 10k/独立 25k 结论 |

## 2. 冻结源码身份

| 项目 | 当前证据 |
| --- | --- |
| 分支 | `codex/work-1` |
| 训练源码冻结 commit | `89d86a2d9a953c0e2b613aba7169909a870b0444` |
| 全量测试 | 2026-07-27：286/286 通过；测试后只更新本文证据字段 |
| 正式配置 | `configs/train_dqn_causal_500k.toml` |
| 正式配置 SHA-256 | `08c35479ddabd1cb383cd2287a94a457ae951b80acef483c3177e64b37ed2541` |
| 正式配置解析指纹 | `515be7448497ef25f45e74a01e539c65d5fbcbcdad4bc3be6a2f88ce6fcd77a8` |

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

旧的 245 项测试和 commit `7d7676f` 只属于此前历史版本，不得把它们登记为本轮
结果。本轮最终全量回归为 286 项全部通过；commit 在本轮提交后补入第 2 节。

## 4. Preflight 状态

| 检查 | 当前结果 | 说明 |
| --- | --- | --- |
| 本机真实门槛 preflight | 失败且符合预期 | commit `89d86a2`，32/32 快照；唯一 required failure 为 `host_physical_memory` |
| 本机总内存 | 失败 | 15.22 GiB，低于 32 GiB |
| 本机可用内存 | 失败 | 本次实测 0.69 GiB，低于 8 GiB |
| 本机磁盘 / 有效 CPU | 通过 | 221.05 GiB 可用；16 个有效 CPU |
| 本机功能性 preflight | 通过但无启动效力 | 只把内存/磁盘阈值临时置 0；`ready=true`、32/32 快照、无 warning，用于证明其余门禁闭合 |
| 旧 preflight | 已失效 | `runs/preflight/first_500k_pre_smoke.json` 来自旧 commit 和旧门禁，只作历史诊断 |
| 云端环境读取 | 未执行 | root SSH 只读检查仍需用户明确授权 |
| 云端全量 preflight | 待运行 | 必须包含 `formal_500k_config_contract`、CUDA、完整因果 optimizer、Shapley、32/32 快照、CPU/内存/磁盘 |

只有云端新版 preflight 返回 `ready=true`、`required_failures=[]`，且
`formal_500k_config_contract` 通过，才允许开始正式 5k。

## 5. 5k 冒烟记录

状态：**未运行**

| 项目 | 当前记录 |
| --- | --- |
| 配置 | `configs/train_dqn_causal_smoke_5k.toml` |
| 输出目录 | `runs/dqn_causal_smoke_5k/` |
| 启动日期 | 待填 |
| 完成时间 | 待填 |
| 退出码 | 待填 |
| 最终 update/env steps | 待填 |
| launcher log | 待填 |
| resource monitor | 待填 |
| 最终 checkpoint | 待填 |
| 最终结论 | 待填 |

仓库中的 `runs/dqn_causal_smoke_5k/` 是旧实现下中止的探针，只到 update 100，
没有完成 checkpoint，也没有可证明正常/异常收尾的 failure marker；现标记为废弃、
非门禁证据。它不代表正式 5k 已启动或通过，不能继续追加。云端新版 preflight
通过后必须从 update 0 使用全新空目录启动，并从完整 CSV、sidecar 和最终 checkpoint
汇总门禁：

| 门禁 | 通过标准 | 结果 |
| --- | --- | --- |
| 训练完成 | 5000 updates，正常退出 | 待填 |
| 数值稳定 | 无 NaN/Inf/failure sidecar | 待填 |
| 主 TD | TD optimizer 持续更新 | 待填 |
| 规则监督 | 正负规则样本均存在，rule batch 实际进入 loss | 待填 |
| 反事实监督 | 至少一个复现通过结果生成样本并进入 loss | 待填 |
| 共享预算 | hard budget 始终为 true，实际比例不超过 0.10 | 待填 |
| 快照与复现 | snapshot failure 为 0；失败率符合运行手册 | 待填 |
| StateAnalyzer | degraded rate 与 shaping p95 符合阈值 | 待填 |
| 并行稳定性 | 无死锁、worker 异常、ID 串线 | 待填 |
| 产物完整 | checkpoint、CSV、曲线、shutdown sidecar 齐全 | 待填 |
| Shapley | 记录 observed/selected/drop reason；5k 零选择不单独否决 | 待填 |

建议填入的最终轻量摘要：

```text
duration:
updates_per_second:
env_steps_per_second:
peak_host_memory:
peak_gpu_memory:
final_loss / td_loss / rule_loss / cf_loss:
causal_replay positive / negative / rule / cf / shapley:
counterfactual completed / failed / reproduction_failed / samples:
counterfactual actual_token_ratio_max:
shapley observed / selected / completed / samples:
shaping_p95_max:
state_analysis_degraded_rate_max:
episodes / truncated_ratio:
checkpoint_bytes:
```

## 6. 10k 校准记录

状态：**待运行、待填**

前置条件：5k 冒烟全部硬门禁通过。

| 项目 | 记录 |
| --- | --- |
| 配置 | `configs/train_dqn_causal_calibration_10k.toml` |
| 输出目录 | `runs/dqn_causal_calibration_10k/` |
| 启动/完成时间 | 待填 |
| commit / config fingerprint | 待填 |
| 是否另起独立 25k | 待决定 |
| 最终 checkpoint | 待填 |
| 最终结论 | 待填 |

结束后补齐：

| 校准项 | 结果 | 结论 |
| --- | --- | --- |
| Reward task 与 shaping 尺度 | 待填 | 待填 |
| TD/rule/counterfactual loss 数量级 | 待填 | 待填 |
| 正负因果样本与 cause type 分布 | 待填 | 待填 |
| 事件 confirmed/cancelled/interrupted/pending | 待填 | 待填 |
| 反事实 proposal/admit/complete/reproduce/sample | 待填 | 待填 |
| 反事实 token 实际/投影比例 | 待填 | 待填 |
| Shapley observed/eligible/selected/result/sample | 待填 | 待填 |
| episode score 与 truncated ratio | 待填 | 待填 |
| 吞吐、CPU、内存、GPU | 待填 | 待填 |
| checkpoint 保存时间和大小 | 待填 | 待填 |

若 10k 后决定另起 25k，必须从 update 0 使用独立输出目录，并记录原因。允许的主要
原因是正式 0.05% 配额下 Shapley 样本不足，但 selector、预算、复现和其它训练信号
都正常；不得用 25k 掩盖 NaN、死锁、预算超限或复现失败。独立从 update 0 启动仍是
首选，因为它提供统一的 25k 探索曲线和更容易比较的校准证据。技术上也允许把 10k
checkpoint 延长到 25k：入口会沿用 checkpoint 冻结的
`epsilon_schedule_total_updates`，不会重新放大探索率；必须由
`resume_<时间戳>.json` 中的 `epsilon_schedule_extended_without_reexpansion=true`
证明这一点，并把该 run 标为“10k 后连续延长”，不能和独立 25k 混为同一证据。

## 7. 云服务器复验

状态：待迁移后填写。

| 检查 | 云端结果 |
| --- | --- |
| GPU/驱动/Torch CUDA | 待填 |
| Python/Pymunk/Chipmunk | 待填 |
| 当前提交的全量 tests（数量以实际输出为准） | 待填 |
| 500k config 32 次 preflight | 待填 |
| 云端 free disk ≥ 80 GiB | 待填 |
| affinity/cgroup 有效 CPU ≥ 13；8 rollout + 4 物理 worker | 待填 |
| launcher/resource monitor 路径 | 待填 |

如果云端依赖版本、GPU 数量或训练语义与本地不同，checkpoint 的严格恢复可能拒绝；
不要通过放宽指纹校验绕过差异。

## 8. 500k 启动批准清单

全部打勾前保持“未获准”：

- [ ] 5k 正常完成，本文第 5 节所有硬门禁通过。
- [ ] 10k 或获批的独立 25k 正常完成，第 6 节已给出结论。
- [ ] 没有未解释的 NaN、死锁、snapshot/reproduction failure 或预算超限。
- [ ] Reward shaping p95、StateAnalyzer degraded 和 truncated 比例符合运行手册。
- [ ] 规则与物理反事实均真实进入 optimizer；Shapley 稀疏状态已解释。
- [ ] 正式配置没有在短跑后被无记录修改。
- [ ] 最终 commit 已提交，工作树 clean。
- [ ] 云服务器重新完成依赖、CUDA、全量测试和 32 次 preflight。
- [ ] 正式输出目录为空，不会覆盖历史 run。
- [ ] 旁路资源监控已经启动。
- [ ] checkpoint 恢复命令和训练产物备份位置已确认。

批准状态：**未获准**

批准人/时间：待填

正式启动 commit/config/preflight：待填

## 9. 已知边界

- 正式 hybrid TD replay 的 checkpoint 只保存 hot layer；恢复时 cold layer 不会伪装成
  已恢复。因果 replay 则精确恢复。
- worker 的物理轨迹和在途反事实/Shapley 任务不会跨进程恢复；续跑从新 episode
  边界开始。
- 物理反事实和 Shapley 只对通过原动作复现门的结果生成标签；失败结果是诊断，不是
  训练监督。
- 异常会生成带时间戳的 `failure_<时间戳>.json` 历史记录和活动指针
  `failure_latest.json`。成功恢复并完整保存最终 checkpoint/曲线后，入口只自动删除
  活动指针，不删除历史记录；readiness 对历史记录告警，对仍存在的活动指针硬失败，
  不需要人工删文件。
- 500k 的正式目标是首次完整方案测试，不是长期维护基线。短跑发现明确问题时应积极
  修正，但修正后必须重置本文证据链。
