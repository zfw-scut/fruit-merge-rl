# 旧项目经验索引

- 状态：只读历史导航，不是 accelerated-v1 的实现规格
- 建立日期：2026-08-01
- 历史分支：`codex/work-1`
- 索引时历史分支 tip：`c45dfcf718fe3053222e39abef5894b40f87b64b`
- 共同基线：`1a95a3d9a47bc6774144061f05ef067098b43544`

## 1. 用途与边界

本文帮助后续 agent 在设计模型、奖励、训练、模拟器或性能管线前，主动吸收旧项目的
实现经验、训练证据和失败教训。旧分支在共同基线后还有 41 个提交，保留了完整训练
体系、云端实验和 Android 成品；这些内容没有复制到 accelerated-v1。

必须同时遵守以下边界：

- `codex/work-1` 是只读证据库，不是依赖库；
- accelerated-v1 的源码、测试和运行流程不得依赖旧工作树路径或旧分支文件；
- 旧代码出现在本文中，不代表允许迁移；
- 不直接 cherry-pick 大型功能提交，不整目录复制旧训练模块；
- 需要复用时，先写清问题、证据和成本，再按当前接口重新实现最小部分；
- 历史训练目录中的旧哈希是产物身份，不能机械替换成重写后的哈希。

## 2. Agent 必读流程

涉及模型、奖励、训练、模拟器、状态表示、长期规划或 CUDA 性能的任务，按以下顺序
开展：

1. 先阅读本文，确定旧项目是否做过相同尝试；
2. 使用 `git show` 阅读相关旧文档和代码，不切换当前 worktree；
3. 使用 `git log`、`git diff` 查看设计引入时同时改变了什么；
4. 查找旧训练证据，区分“链路能运行”和“策略确实变好”；
5. 明确写出保留的概念、拒绝的旧实现和新的验证方式；
6. 只有获得当前任务授权后，才向 accelerated-v1 添加代码或依赖。

常用只读命令：

```powershell
# 查看旧分支时间线
git log --oneline codex/work-1

# 阅读旧分支中的文档或源码
git show codex/work-1:docs/rl/STRUCTURE_AWARE_GNN_V2.md
git show codex/work-1:src/daxigua_rl/reward.py

# 查看某次提交实际改变了什么
git show --stat 69744b90bac2b256a89280e4cab97f267c8eeb59
git diff 1a95a3d9a47bc6774144061f05ef067098b43544..codex/work-1 -- src/daxigua_rl

# 在旧分支中搜索符号，不检出旧分支
git grep -n StateAnalyzer codex/work-1 -- src
```

## 3. 关键演进节点

下列完整哈希及短前缀均指向 Git 历史中文化重写后的 canonical commit。

| 阶段 | 关键提交 | 得到的能力 | 应吸取的结论 |
| --- | --- | --- | --- |
| 无渲染接口 | `89f87857ea1fbc74d2ca3f0f281a6b8da5a476ee` | `HeadlessGame`、`DaxiguaEnv`、对象状态 | 渲染和训练状态应分离；但旧模拟器本身不直接迁入。 |
| 初代图模型 | `5b20135`、`d85c8a7` | GraphBuilder、H128/L3 GNN-Q | 图表达能处理可变水果数，但 Python 动态构图后来成为瓶颈。 |
| 张量批训练 | `a2d2fb5` | `GraphBatch`、`TensorTransition` | 将多个图合并成一次 GPU 前向是正确方向。 |
| 性能基线 | `7652538`、`81c511f813ed58a31815f674dfaff85feab622f8` | accurate/fast30 对比、并行/异步 rollout、profiling、TOML | 先测量、再优化；物理分布偏移必须与吞吐一起验证。 |
| 因果设计规格 | `2487a2425ee77265a3f7256d6764176ab2f7cfa9` | 完整状态归因 V1 规格 | 规格覆盖面很广，但设计成立不能替代有效性和成本证据。 |
| 正确性基线 | `1a95a3d9a47bc6774144061f05ef067098b43544` | 真实碰撞半径、连续稳定窗口、正确边界 bootstrap | 真实终止和技术截断仍需正确 bootstrap；物理等待上限现已单列为非边界 `settle_timeout`。 |
| 状态塑形 | `87fab99`、`bf6aebe`、`2d48c58` | `StateAnalysis`、`StateAnalyzer`、Reward V2 | 去除存活/最高高度奖励是进步；每步重型静态分析和代理 potential 不应照搬。 |
| 延迟归因 | `5772745`、`ca28394` | Tracker、谱系事件、确定性快照 | 事件谱系便于审计，但复杂状态机和逐步快照成本很高。 |
| 完整因果训练 | `0cb070b`、`356263d`、`7e6cade` | Double DQN、3-step、因果 replay、反事实、局部 Shapley、恢复门禁 | 链路完整不等于有效长期监督；物理复现和资源门禁必须独立验证。 |
| 结构感知 V2 | `69744b90bac2b256a89280e4cab97f267c8eeb59` | H256/L4、关系门控、motif、六维辅助头、集中 GPU actor | 模型增大约 8.9 倍，但局部一步目标没有带来可靠长期规划。 |
| 正式训练门禁 | `4023be59a7db90a400b88cc0b883ff77f58447d0`、`610e12daa381f0c9a7262f778726206cbae86fdf` | 500K/250K runbook、preflight、readiness | 门禁能证明配置和链路可运行，不能替代策略净增益对照。 |
| 运行可靠性修复 | `2e68172bd0b344fdaebc8856f61b4dcd3af0ef4b`、`0cc0e68` | 修复 Shapley FD 泄漏、观看稳定边界 | 多进程张量 IPC 和物理边界必须纳入故障测试。 |
| 场景尺寸迁移 | `604899b7440af6beeb70acfa33a11f8ed9855254` | 560×1120、21 动作、weights-only 初始化 | 动作共享读出有迁移价值；旧 replay/optimizer 不应跨尺寸续用。 |

表中短哈希均可在 `codex/work-1` 唯一定位；做迁移审计时应使用 `git rev-parse` 记录
完整哈希。

## 4. 相关旧文档

这些文件只存在于 `codex/work-1`，使用 `git show codex/work-1:<路径>` 阅读。

| 主题 | 历史路径 | 用途 |
| --- | --- | --- |
| 无渲染接口和状态契约 | `docs/rl/INTERFACE_V0.md` | 理解旧状态、动作和物理边界。 |
| 训练吞吐优化 | `docs/rl/TRAINING_SPEED_OPTIMIZATION_PLAN.md` | 查看 fast30、并行采集、缓存和 profiling 的演进。 |
| 因果归因规格 | `docs/rl/CAUSAL_ATTRIBUTION_V1.md` | 理解旧归因目标、预算和反事实边界，不作为新实现规格。 |
| 结构感知模型 | `docs/rl/STRUCTURE_AWARE_GNN_V2.md` | 查看关系边、motif、六维目标和集中 actor 的完整设计。 |
| 250K 正式训练 | `docs/rl/FIRST_250K_RUNBOOK.md` | 查看正式配置、恢复边界和训练门禁。 |
| 500K 旧方案 | `docs/rl/FIRST_500K_RUNBOOK.md` | 查看完整因果方案最初的大规模训练假设。 |
| 250K 证据 | `docs/training_runs/FIRST_250K_READINESS.md` | 查看门禁、故障、恢复和证据解释。 |
| 500K 证据 | `docs/training_runs/FIRST_500K_READINESS.md` | 查看旧因果链路的准备状态。 |
| 性能主链路记录 | `docs/codex/33_优化DQN训练性能主链路_2026_07_20.md` | 查看进入因果方案前的轻量性能基线。 |
| 物理与 TD 语义 | `docs/codex/37_修正状态归因基础语义_2026_07_27.md` | 查看共同基线保留的正确性修复。 |
| Reward V2 | `docs/codex/40_接入RewardV2状态塑形_2026_07_27.md` | 查看指数合成效用和 C/R/K potential。 |
| 因果主链路 | `docs/codex/42_完成首轮完整因果训练主链路_2026_07_27.md` | 查看 tracker、replay、反事实和联合损失的接线。 |
| 尺寸迁移 | `docs/codex/54_扩大场景并新增尺寸迁移训练_2026_07_29.md` | 查看大地图和 21 动作的迁移边界。 |

## 5. 已验证有价值的经验

这里保留的是原则，不是旧实现的复制许可。

### 5.1 规则、物理和训练语义必须分层

- 水果等级、半径、合成分值属于稳定领域规则；
- 显示半径和真实碰撞半径不能混用；
- `terminated` 与技术 `truncated` 不能都当成禁止 bootstrap；物理等待上限
  `settle_timeout` 不是回合边界，必须允许继续 bootstrap 和下一次投放；
- 稳定状态必须满足连续稳定窗口，不能依赖最后一帧瞬时速度；
- 游戏分值、环境 reward 和辅助监督必须分别记录。

### 5.2 性能必须分阶段量化

- 同时记录 `env_steps/s` 和 `updates/s`，不能只看 GPU 使用率；
- 将物理、状态构建、动作推理、采样、前向、反向、保存和评估分别计时；
- fast 物理模式必须与 accurate 模式比较分数、局长、合成率、截断率和状态分布；
- next-state 复用、多环境采集和异步重叠是值得保留的工程思想；
- 增大 learner batch 前，先确认采样端能持续供应数据。

### 5.3 训练产物必须可审计

- 配置、代码身份、模型结构、地图尺寸和动作数应写入 checkpoint 或伴随清单；
- best checkpoint 必须由独立 greedy eval 决定，不能只看训练 loss；
- 恢复训练要区分严格 resume 和 weights-only 初始化；
- checkpoint 应原子写入并验证 round-trip；
- 大规模训练前需要短 rollout、单批更新、保存恢复和资源边界门禁。

### 5.4 长期信号必须来自足够长且足够多的样本

- Double DQN、n-step 和轨迹片段训练值得重新评估；
- 合成谱系适合做离线审计或便宜的真实轨迹标签；
- 稀有高等级合成和长连锁需要分层采样，不能被即时合成样本淹没；
- 辅助目标必须检查有效样本数、类别分布和实际梯度量级，而不只是 loss 非零。

## 6. 失败原因与历史证据

以下数值来自旧工作树本地 `runs/cloud_evidence`，属于特定机器和配置下的历史证据，
不是 accelerated-v1 的性能承诺。该目录未提交，换机器后可能不存在。

| 证据 | 历史观察 | 说明 |
| --- | --- | --- |
| 250K 正式训练约 143K | 吞吐由早期约 `2.69 update/s` 降至约 `1.28 update/s`，约 `20.5 env steps/s` | 系统复杂度随训练阶段显著拖慢主链路。 |
| 集中 actor | 平均 batch 约 `5.4/16` | CPU 请求到达不整齐，GPU actor 长期吃不满；继续增大 learner batch 不能解决。 |
| StateAnalyzer | 约 8046 次调用累计约 427 秒，均值约 53 ms | 每步 Python 静态分析是明确 CPU 热点。 |
| 反事实 proposal | 累计约 247073 个，接纳约 5045 个，接纳率约 2%；序列化约 627 MB | 在线反事实产出/成本比过低。 |
| 结构辅助监督 | 140K 附近 TD loss 约 5.87，加权结构 loss 约 0.0019 | loss 量纲不能直接等同梯度贡献，但结构项明显弱于主 TD 信号。 |
| 大地图 25K | 约 `0.865 update/s`、`14.0 env steps/s` | 地图扩大后 CPU 物理、分析、构图和同步等待进一步放大。 |
| 长期正样本 | 同一历史窗口约 7622 个 `MERGE_LINEAGE`，但 `REALIZED_ADJACENCY=8`、`REALIZED_LADDER=1`、Shapley 样本约 62 | 真正长期铺垫标签极稀疏，无法支撑复杂在线归因体系。 |

本地证据路径：

```text
../fruit-merge-rl/runs/cloud_evidence/formal_250k_live_140k/metrics.csv
../fruit-merge-rl/runs/cloud_evidence/size_transfer_560x1120_stopped_25k/metrics.csv
```

由这些证据得到的主要教训：

1. 反事实预算只限制物理重演，不代表 tracker、分析、快照、proposal 和 IPC 没有成本；
2. 稀少的“因果样本”被高频 replay，不等于模型获得了可靠长期规划；
3. 一至两级局部 motif 和一步结构变化预测不能替代长轨迹信用传播；
4. 增大网络从约 40 万到约 358 万参数，只增加表达能力，不自动产生规划能力；
5. 动态稠密图、Python 逐边构造和 GraphTensor 进程间传输不适合大批 CUDA 管线；
6. eval 有明显方差，单个周期下降或上升都不能独立证明策略变化；
7. 代码、指标和门禁数量增加后，必须重新检查每项是否影响优化器，而不是只确认有日志。

### 6.1 多进程资源故障

旧 250K 训练曾在约 update 42348 因 Local Shapley 把 GraphTensor 经 PyTorch
multiprocessing 共享而耗尽文件描述符。历史修复改为 bytes 传输并提高 nofile 上限。
这说明：

- 进程间传输格式必须显式计算句柄、复制和序列化成本；
- 提高系统上限只能作为保险，不能替代有界资源设计；
- CUDA tensor、共享内存和 worker 生命周期都必须有压力测试和退出测试。

相关历史文档：

```powershell
git show codex/work-1:docs/codex/52_修复局部Shapley文件描述符泄漏_2026_07_28.md
git show codex/work-1:docs/training_runs/FIRST_250K_READINESS.md
```

## 7. 禁止直接照搬的模块

以下模块可以阅读和提取教训，但不得整体复制、直接依赖或通过大型 cherry-pick 引入：

### 7.1 完整状态分析与归因栈

```text
src/daxigua_rl/attribution/
src/daxigua_rl/training/counterfactual_coordinator.py
src/daxigua_rl/training/local_shapley_coordinator.py
src/daxigua_rl/attribution/state_analyzer.py
src/daxigua_rl/attribution/tracker.py
src/daxigua_rl/attribution/causal_replay.py
```

原因：热路径成本高、状态机耦合深、长期正样本稀疏，反事实有效产出低。

### 7.2 旧结构感知模型实现

```text
src/daxigua_rl/graph/builder.py
src/daxigua_rl/models/gnn_q.py
src/daxigua_rl/training/structural_targets.py
```

原因：动态稠密图、motif 虚拟节点、H256/L4 关系 GNN 和六维一步辅助头形成整体耦合，
不能满足当前定长张量、大批 CUDA 和多时间尺度监督方向。

### 7.3 旧训练总入口和 IPC 管线

```text
src/daxigua_rl/scripts/train_dqn.py
src/daxigua_rl/training/parallel_collector.py
src/daxigua_rl/training/replay_buffer.py
```

原因：入口已经承担过多配置、门禁、恢复、归因和 worker 协调职责；集中 actor 的同步
RPC 平均 batch 过小；GraphTensor 对象和冷磁盘 replay 也不适合作为新默认热路径。

### 7.4 旧奖励和正式训练配置

```text
src/daxigua_rl/reward.py
configs/train_dqn_causal_*.toml
docs/rl/FIRST_250K_RUNBOOK.md
docs/rl/FIRST_500K_RUNBOOK.md
```

原因：Reward V1 含存活、最高高度和危险高度代理；Reward V2 虽修正方向，但依赖昂贵
C/R/K potential。旧配置绑定旧地图、模型、replay 和归因预算，不能成为新训练默认值。

### 7.5 产品和运维层

Android 游戏、训练面板、云服务器脚本、密码和本地运行产物不进入训练核心分支。需要
部署或导出时，只根据当时稳定接口重新增加最小适配层。

## 8. 可以保留概念但必须重新验证的内容

| 概念 | 重新采用前的要求 |
| --- | --- |
| accurate/fast 物理双模式 | 重新建立分布对比，不沿用旧结论作为永久事实。 |
| Double DQN、n-step | 使用新 replay 和固定张量实现，并做独立消融。 |
| 合成谱系 | 优先作为离线/轨迹级标签，不阻塞环境 step。 |
| 支撑、遮挡、同级伙伴、投放路径 | 用批量几何或稀疏关系实现，禁止恢复逐状态 BFS/复杂 motif。 |
| 集中 GPU 推理 | 使用连续 chunk、共享内存和足够大的真实 batch；测量队列等待。 |
| 原子 checkpoint | 保留思想，但重新定义新模型、optimizer、replay 和地图指纹。 |
| 大地图 21 动作 | 单独迁移领域几何与动作契约，不迁移旧 replay 或训练状态。 |
| 模型导出 | 等模型结构稳定后再定义，不提前兼容旧 Android 权重格式。 |

## 9. 采用旧经验前的检查表

每次准备参考旧实现时回答：

1. 它解决的原始问题现在仍然存在吗？
2. 历史证据证明的是“能运行”，还是“策略显著变好”？
3. 它在每个 env step、每次 update、每局结束时分别花费多少？
4. 它产生多少有效训练样本，类别是否足够平衡？
5. 它能否使用定长张量、大批 CUDA 或离线处理实现？
6. 它是否迫使训练入口、replay、模型和模拟器互相依赖？
7. 最小可验证版本是什么，如何与基线做消融？
8. 如果结果无提升，能否一项配置关闭且不改变其他语义？

只有这些问题有明确答案，旧经验才可以从“只读参考”进入新设计。
