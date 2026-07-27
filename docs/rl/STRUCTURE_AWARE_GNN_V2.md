# 结构感知 GNN V2 与首次长训契约

状态：已实现，等待以新架构重新完成训练门禁

适用分支：`codex/work-1`

最后更新：2026-07-28

## 1. 为什么需要这一版

完整状态归因 V1 已经能在动作发生后回答“哪次铺垫后来兑现”“哪次投放造成封路或
埋死”，规则样本、反事实和局部 Shapley 也能把这些结果转成 Q 值监督。但旧 GNN
主要接收几何和等级信息，模型需要自己从多层圆形布局中重新推断支撑、遮挡、可达伙伴
和连锁结构。归因系统知道的结构，没有完整进入策略网络。

V2 的目标是闭合这条链路：

```text
当前稳定局面的可见结构
-> 关系图与连锁 motif
-> 关系感知 GNN
-> 动作 Q 值 + 一步结构结果预测
-> TD / 因果排序 / 结构辅助监督联合更新
```

这不是把规则动作写死，也不是增加一个规划器。策略仍由 DQN 的长期回报决定；V2
只是让模型能直接看见长期规划所需的中间结构，并用真实动作结果训练共享表示。

## 2. 三条不可跨越的边界

### 2.1 不使用未来信息作为模型输入

构图只读取动作前稳定边界的 `GameState`、15 个当前候选投放位置以及当时已经显示给
玩家的 q0～q3 队列。q1～q3 是游戏当前可见信息，不是未来泄漏。

`q0_landing_depth`、安全列、第一阻挡者和 motif 的触发动作来自当前几何的静态只读
分析，不推进一次隐藏物理模拟。动作后的 `StateAnalysis`、合成事件和 terminal 标志
只用于生成训练 target，绝不写回动作前图。

### 2.2 不增加或复制环境奖励

V2 不修改 Reward V2。指数合成效用、C/R/K potential 和 terminal 语义仍由
`reward.py` 唯一定义。motif 节点不会发放“连锁奖励”，六维结构 target 也不会写入
`reward`、episode score 或 n-step return。

总训练损失为：

```text
TD loss
+ lambda_rule * rule ranking loss
+ lambda_cf * counterfactual/Shapley loss
+ lambda_structural * structural auxiliary loss
```

结构项是共享 GNN 表示上的辅助学习任务，不是第四种环境奖励。

### 2.3 新架构必须从 update 0 开始

V2 改变了图特征维度、图拓扑、消息传播层、Q 读出和模型规模。旧 checkpoint 和旧
replay 只能作为历史证据保留，不能用于 V2 正式训练恢复。详细边界见第 9 节。

## 3. 模型现在能看见哪些长期结构

`GraphBuilder.build(state, action_candidates, state_analysis=...)` 将当前
`StateAnalysis` 投影进训练图。模型输入不包含任意 `fruit_id`、`region_id` 或
`motif_id` 数值，避免把单局身份误学成可泛化特征。

### 3.1 水果节点

每个场上水果除位置、速度、等级和真实碰撞半径外，还包含：

- 15 个规范投放列中的可达比例和顶部可见比例；
- 是否仍有可达同级伙伴、伙伴数和可达伙伴数；
- 支撑父节点数、被其支撑的子节点数；
- 埋藏深度、倒置 blocker 数和关键 blocker 数；
- 是否连接到顶部可用空间。

这使“小水果被大水果压到底部且失去合成入口”成为显式状态，而不必依赖水果年龄或
单纯的绝对高度。

### 3.2 显式关系边

水果之间不仅有距离边，还能表达：

- 当前接触；
- `supports`、`caps`、`bridges`；
- 可达同级伙伴；
- 关键 blocker；
- 等级倒置 blocker。

支撑、盖压和桥接使用 `supporter -> supported` 方向；关键阻挡和倒置关系使用
`blocker -> victim` 方向。反向边按图构建需要显式生成，模型不会把方向自动当作
对称关系。

动作到水果的边还标出：

- 该动作列是否仍能到达这个水果；
- 这个水果是否为 q0 在该列上的第一阻挡者。

因此动作节点可以区分“落在附近”和“真实投放路径会首先撞到谁”。

### 3.3 动作与全局节点

每个动作节点获得与自身列对齐的 q0 静态落点深度、安全标志和 blocker 数。全局节点
获得：

- `top_connected_capacity`；
- `recoverability`；
- `chain_readiness`；
- 顶部连通自由空间比例；
- 封闭空腔比例和数量；
- 分析是否有效、降级。

`StateAnalysis` 缺失时，旧的 `GraphBuilder.build(state, actions)` 调用仍可执行，
但新增结构列全部为零且不创建 motif 节点。这是接口兼容路径，不是正式 V2 训练模式。

## 4. 连锁关系的特殊机制

### 4.1 第一版 motif 的范围

当前实现只把已有静态分析能够可靠识别的两种局部结构变成虚拟节点：

- `merge_pair`：两颗同级水果构成可接近、存在实际触发列的合成材料；
- `level_ladder`：上述同级材料附近还有一颗下一级水果，前一级合成后具备继续合成的
  局部阶梯条件。

第一版不会声称已经穷举任意长度的未来连锁。`level_ladder` 当前描述两级局部链，
而真实同一步物理结果可以通过 fruit ID 谱系识别更深的连续合成。

### 4.2 motif 节点保存什么

每个 motif 节点保存类型、基础等级、成员数、深度、就绪度、可触发动作比例，以及
当前 q0 或未来可见 q1～q3 是否有兼容等级。它不保存成员 ID，也不保存一次假想动作
后的局面。

水果与 motif 的边区分：

- `pair_member`：当前阶梯的同级合成材料；
- `chain_target`：前一级合成后可能继续命中的高一级水果；
- `stage`：成员处于局部链的哪个等级阶段。

这比把所有水果压成一个 `chain_readiness` 标量更具体：GNN 能知道“哪两颗是材料、
哪一颗是下一阶段目标”。

### 4.3 当前动作、未来队列和保护动作

队列与 motif 只连接真正兼容的可见槽位：

- q0 边使用 `trigger_now`，表示当前水果具备立即触发该 motif 的等级条件；
- q1～q3 边使用 `future_queue` 和按队列距离衰减的关系强度，表示该结构值得保护到
  已知的未来槽位。

每个候选动作都与 motif 双向连接，并分别展开：

- `trigger_now`：该动作位是否在 motif 的实际触发 mask 中；
- `preserve`：当前不触发，但落点安全且第一 blocker 不包含 motif 成员；
- `break_risk`：当前不触发，却会把 q0 直接落到 motif 成员路径上，风险按受影响
  成员比例计算。

`preserve` 和 `break_risk` 只是动作前静态提示，不是动作好坏的最终判决。真实动作
是否保住或破坏连锁，仍由下一状态结构 target、延迟规则归因或极少量反事实结果确认。

### 4.4 一个局部阶梯示例

假设场上有两颗 L3 水果形成可达材料，附近有一颗 L4 水果，q0～q3 中存在 L3：

```text
L3(A) ----\
            > merge_pair / level_ladder motif ---- L4(C)
L3(B) ----/                 |
                              +-- 当前可触发动作
                              +-- 安全保护动作
                              +-- 破坏风险动作
                              +-- 可见队列兼容槽
```

动作消息可以沿“动作 -> motif -> L3 材料/L4 目标”传播；水果的 blocker、支撑和
顶部可达关系又会反向影响 motif 与动作表示。这样模型不只知道“附近等级相似”，而是
能学习：

1. 当前能否触发前一级合成；
2. 新水果是否可能沿路径到达材料；
3. 合成后的等级是否已有下一阶段目标；
4. 当前不能触发时，哪些动作更可能把结构保留到 q1～q3；
5. 哪些动作会先压住成员、制造空腔或切断入口。

如果真实物理在同一动作内产生 `L3 -> L4 -> L5`，辅助 target 会用新 fruit ID 被后续
merge 消费这一谱系证据确认“真实连锁”；两次互不相干的同时合成不会冒充连锁。
连锁 target 的效用也只累计同一条连通 fruit-ID 谱系，旁边同时发生的无关大水果
合成不会抬高该连锁标签。

## 5. 关系感知 GNN 与动作价值

每条边先编码为隐藏表示。每层消息传播对 `source / target / edge` 三元组执行：

1. `message_mlp` 生成候选消息；
2. `relation_gate` 根据边表示生成逐通道门，区分支撑、阻挡、伙伴、motif 等关系；
3. `attention_gate` 为具体边生成标量权重；
4. 按目标节点做加权平均；
5. 经残差连接和 `LayerNorm` 更新节点。

这避免把“接触”“盖压”“可达伙伴”当成同一种邻接。正式 V2 配置采用
`hidden_dim=256`、`message_layers=4`，让动作、motif、局部水果关系和全局状态能在
更宽、更深的共享表示中交互；当前正式模型共有 3,580,940 个可训练参数。

Q 值使用 dueling 分解：

```text
Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
```

`V(s)` 从每张图唯一的 global 节点读取，`A(s,a)` 从动作节点读取。这样全局局面价值
与各投放列的相对优劣不必挤在同一个动作头里。

## 6. 六维一步结构监督

`forward_with_aux()` 与 Q 头共享同一次图编码，为每个动作输出六维预测。训练只选取
实际执行动作对应的一行，与该动作真实产生的 `analysis[t] -> analysis[t+1]` 比较。
未执行动作不会凭空获得伪标签。

| 维度 | 方向与含义 |
| --- | --- |
| `top_connected_capacity_delta` | 有符号的顶部连通投放容量变化；正值表示改善。 |
| `recoverability_delta` | 有符号的水果可恢复性变化；正值表示改善。 |
| `chain_readiness_delta` | 有符号的局部连锁就绪度变化；正值表示改善。 |
| `new_dead_or_blocked_fruit_risk` | `[0,1]` 风险；关注新形成的零入口死果和投放路径损失，低级、深埋且无伙伴的水果权重更高。 |
| `sealed_cavity_delta` | `next - previous` 的封闭空腔比例变化；正值表示空腔增加，负值表示改善。 |
| `realized_chain_or_terminal_risk` | `-1` 表示真实 terminal，`0` 表示没有谱系连锁，正值按真实连锁深度和合成效用增加。 |

六维值都限制在 `[-1,1]`。前五维只有相邻分析具备同 worker/episode、连续 step、
相同 analyzer 指纹、稳定且未降级时才有效；最后一维独立依赖本步
`PhysicsResult`、有序 `MergeEvent` 和 terminal 证据。逐维 `valid_mask` 会把不可信
标签完全排除。

结构损失使用实际动作的 masked SmoothL1，并以
`lambda_structural=0.15` 加入总 loss。3-step replay 只聚合奖励；结构 target 始终
属于窗口起始动作自己的单步结果，不把后两步结构变化错误归给第一步。

## 7. 集中式 GPU actor

旧并行采集让每个 rollout worker 各自在 CPU 上做小图推理，物理和分析占满 CPU，
GPU 主要只在 learner 更新时短暂工作。V2 正式配置使用集中式 actor：

```text
16 个 CPU rollout worker
        |
        | greedy 动作的 GraphTensor 请求
        v
主进程 actor 队列 -- 最多 16 图 / 最多等待 2 ms --> GPU actor model
        |
        +--> 各 worker 的动作 Q 值

GPU learner online model ---- 每个 worker_sync_interval ----> actor 参数副本
```

epsilon 随机动作不需要请求模型。集中 actor 持有独立的 eval 参数副本，learner 可以
继续更新 online model；只在同步边界加锁刷新 actor。相对旧 worker 模式，策略陈旧
边界仍由 `worker_sync_interval` 控制，集中化不改变 Reward、replay 或动作空间语义。

正式值为：

- `centralized_actor_inference=true`；
- `actor_batch_size=16`；
- `actor_batch_wait_ms=2.0`；
- `actor_request_timeout_seconds=120.0`。

微批大小会随 epsilon 和 worker 到达时间变化。训练早期 epsilon 高时 greedy 请求少，
平均 batch 小是正常现象；判断是否有效应同时比较吞吐、动作选择耗时和 GPU 利用率，
不能只追求单一的 GPU 百分比。

## 8. 如何启用

训练入口默认支持同名 CLI 参数；TOML 使用下划线字段。V2 正式基线应包含：

```toml
[training]
collect_per_update = 16
batch_size = 128

[causal]
causal_batch_size = 64
lambda_structural = 0.15

[model]
hidden_dim = 256
message_layers = 4

[parallel]
num_envs = 16
centralized_actor_inference = true
actor_batch_size = 16
actor_batch_wait_ms = 2.0
actor_request_timeout_seconds = 120.0

[counterfactual]
counterfactual_workers = 6
counterfactual_cpu_core_ratio = 0.24
```

通用 `DQNTrainerConfig` 保留 `lambda_structural=0` 的兼容默认，以免外部自定义模型
没有 `forward_with_aux()` 时被强制启用；正式 `train_dqn` 入口和三阶段配置使用
`0.15`。设为 `0` 只关闭结构辅助 loss，不会删除已经构建的结构图。需要完整消融时
使用 `GraphAblator` 的 `no_structure_analysis` preset；它除遮罩普通结构特征外，
还会删除 motif 虚拟节点及关联边，避免 motif 数量和连接拓扑继续泄漏结构。首次长训
不启用该消融。

启动 V2 烟测时必须选择全新空目录，且不传 `--resume`：

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u \
  -m daxigua_rl.scripts.train_dqn \
  --config configs/train_dqn_causal_smoke_5k.toml \
  --run-dir runs/<new-v2-smoke-run>
```

正式 500k 同样从 update 0 开始，不允许指向旧 run 的 `replay_cold` 目录。

## 9. Checkpoint 与 replay 兼容边界

| 产物 | 文件能否被基础加载器识别 | 能否用于 V2 续训 | 正确用途 |
| --- | --- | --- | --- |
| 旧 H128/L3 checkpoint | checkpoint 外层可能仍能解析 | **不能** | 用旧 commit/旧环境做历史观看或对比 |
| 旧 optimizer / target state | 外层可能仍能解析 | **不能** | 只随旧模型归档 |
| 旧 hot replay | 旧 transition 可把缺失结构 target 解释为 `None` | **不能** | 只随旧 run 归档 |
| 旧 cold replay v1/v2 | segment loader 仍认识旧记录格式 | **不能** | 只读迁移或历史审计 |
| 旧 causal replay | 数据结构可能仍可反序列化 | **不能** | 旧图上下文属于旧 schema，只作历史证据 |
| 同一 V2 run 的可信 checkpoint | 能 | 能，遵循严格配置指纹和 hot-resume 规则 | 中断恢复 |

拒绝混用的原因不是只有缺失六维 target。旧图的节点/边特征维度与 V2 不同；模型又新增
relation gate、attention、dueling advantage 和结构头，并从 H128/L3 改为
H256/L4。即使对权重使用 `strict=False`，新增层仍是随机参数，结果既不是续训也不是
可比较的迁移学习，因此禁止用于首次 V2 长训。

replay v1/v2 的读取兼容只保证旧文件可以被校验和迁移，不代表图语义兼容新网络。
新的 run 目录会自然创建新的冷 replay；无需删除历史目录，也不能把旧段文件复制进去。

hybrid replay 的 checkpoint 语义仍是 hot-resume：同一 V2 架构内恢复时只恢复有界
热层，未保存的冷层不会被宣称为精确恢复。

## 10. 首轮训练要观察什么

### 10.1 结构辅助监督

`metrics.csv` 应持续记录：

- `structural_loss`；
- `weighted_structural_loss`；
- `structural_valid_count`；
- `structural_sample_count`；
- `structural_mean_abs_error`。

`structural_valid_count` 是有效“维度”数量，不超过
`6 * structural_sample_count`，两者不能混为样本数。正常训练中它们应持续非零，
loss/MAE 必须有限；若长期为零，应优先检查相邻 `StateAnalysis` provenance、降级率
和 transition target 是否贯通。

正式 preflight 使用更严格的单批门禁：128 个样本必须全部带有六维有效标签，即
`structural_sample_count=128`、`structural_valid_count=768`。它不会接受“只有第六维
物理结果有效”或 batch 中仅少量样本有效，因此前五维相邻 `StateAnalysis` 的稳定性、
连续 provenance 和 analyzer 指纹也会在首次 optimizer step 前被真实验证。

当前 CSV 是六维聚合误差，不能单独证明稀有连锁维已经学会。连锁效果还要联合观察：

- `collect_attribution_chain_merge_count` 和
  `collect_attribution_max_chain_depth`；
- `collect_mean_previous_chain_readiness` /
  `collect_mean_next_chain_readiness`；
- 规则/反事实样本是否真实进入 optimizer；
- greedy eval 的真实 episode score、最大水果等级和 terminal 表现。

### 10.2 集中 actor

应记录并观察：

- `actor_inference_requests`；
- `actor_inference_batches`；
- `actor_inference_mean_batch_size`；
- `actor_inference_max_batch`；
- `actor_inference_seconds`。

请求数应与 greedy 动作数一致或能由异步窗口边界解释；最大 batch 不得超过配置值。
平均 batch 需要结合当时 epsilon 解读。最终性能判断以 env steps/s、updates/s、
`collect_action_select_seconds`、CPU 利用率和 GPU 利用率的共同变化为准。

正式 preflight 不只检查这些开关：它会临时启动 16 个真实 rollout worker 和位于正式
CUDA device 的 H256/L4 actor，先同步一次并显式执行 `start_collect_steps` /
`finish_collect_steps`，再同步一次并执行第二轮采集。两轮均以 `epsilon=0` 强制每个
环境 step 等待 actor 响应，最后验证 request、batch、最大 batch、策略版本、replay
flush 和 16 个 worker finalization 后关闭线程、进程与队列。该探针不创建训练 run。

### 10.3 仍然是主目标的指标

辅助 loss 下降不等于策略变好。扩大到 500k 前至少确认：

- TD、结构、规则和反事实 loss 均无 NaN/Inf；
- 总梯度范数没有持续异常；
- StateAnalyzer 降级率没有因 16 worker 压力显著上升；
- 物理吞吐、队列延迟和 actor timeout 正常；
- episode score、合成等级分布和真实连锁统计没有系统性退化；
- 反事实/Shapley 仍遵守共享 10% 硬预算。

## 11. 首次长训证据重置

旧 V1 的 5k/10k 证明了当时 Reward V2、完整归因、反事实和 Shapley 链路能运行，
但不能批准 V2 的 500k。以下项目必须在 V2 最终 commit 上重新产生：

1. 全量单元测试与 compile 门禁；
2. 正式配置 preflight；
3. 从 update 0 开始的全新 5k 烟测；
4. 从 update 0 开始的全新 10k 标定；
5. 对结构 target、集中 actor、吞吐、内存、GPU 和真实游戏指标的审查；
6. 最终 500k 启动批准。

这次重跑不是为了重新讨论是否保留结构设计，而是为了尽早发现 schema、吞吐、数值和
资源问题。第一次正式 500k 直接使用完整结构图、motif、关系 GNN、六维监督和集中
actor，不先训练一个被刻意削弱的长期基线。
