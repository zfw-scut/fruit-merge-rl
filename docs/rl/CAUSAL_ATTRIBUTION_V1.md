# 完整状态归因 V1 设计规格

状态：归因语义已实现；模型输入与长训放行已由结构感知 V2 接续

版本：V1

日期：2026-07-27

> 2026-07-28 更新：本文继续定义 Reward V2、历史归因、反事实和局部 Shapley
> 语义。当前模型如何消费结构分析、如何训练六维辅助目标、如何集中 GPU actor，以及
> 为什么旧 checkpoint/replay 不能续训，统一见
> [`STRUCTURE_AWARE_GNN_V2.md`](STRUCTURE_AWARE_GNN_V2.md)。旧 V1 5k/10k
> 证据不能批准新架构 500k。
>
> 2026-07-29 场景迁移说明：本文中的 15 动作描述是旧场景的历史规格。当前项目
> 默认场景已改为 `560x1120 / spawn_y=252 / 21 actions`，状态分析 schema 升为
> v3。归因机制不变，但旧 replay/mask 不兼容；同架构网络只能通过 weights-only
> `--init-checkpoint` 进入新 run，详见 V2 文档第 9.1 节。

## 1. 文档目标

本文固定第一次大规模训练使用的奖励、状态分析、历史贡献归因和稀疏反事实方案。

项目只计划进行数天开发和两到三次大规模训练，因此 V1 的目标不是建立长期维护的
通用因果推断框架，而是在有限时间内完成一套：

- 能覆盖合成铺垫、连锁、封路、埋死、解救和可投放空间变化的完整状态归因；
- 能直接影响现有 GNN-Q 动作排序；
- 不依赖全量反事实重演；
- 可以在第一次大规模训练中完整启用；
- 有明确开销上限、日志和故障降级路径；
- 能通过少量人工场景和短训练排除明显实现错误。

本文中的“完整状态归因”指归因类型覆盖完整，不表示对每个状态枚举全部动作并重演完整
episode。绝大多数经验使用实际轨迹上的机制证据归因，反事实仅仲裁少数高价值歧义事件。

## 2. 当前训练基线

当前训练主链路为：

```text
DaxiguaEnv
-> GraphBuilder
-> GNNQNetwork
-> TensorTransition
-> ReplayBuffer
-> DQNTrainer
```

现有可复用基础包括：

- `DropResult.fruit_id`：投放产生的根水果 ID。
- `MergeEvent.source_ids`：一次合成消耗的两个父水果 ID。
- `MergeEvent.new_fruit_id`：合成后新水果 ID。
- `MergeEvent.score_delta`：游戏规则层产生的合成分数。
- 每个状态包含水果位置、速度、等级、边界距离和 q0～q3 队列。
- 图中已经包含动作、水果、队列、边界和全局节点，以及动作投放路径关系。
- 正式训练默认使用 15 个离散投放动作、8 个并行环境和 `fast30` 物理模式。
- 当前长训配置为 500000 次更新，每次更新采集 8 次投放。
- 每个环境在 worker 内缓存相邻 `StateAnalysis`，Reward V2 已消费 C/R/K potential。
- 每个 worker 已维护独立 `AttributionTracker`，在主 replay 写入前消费真实
  drop/merge/状态转移证据。
- `RolloutStats` 和训练 CSV 已记录 Reward V2、StateAnalyzer 与 AttributionTracker
  的核心性能和事件生命周期指标。

当前实现边界如下：

- 主 `TensorTransition` 保持不可变，不塞入会延迟到达的历史归因；历史监督进入独立
  `CausalReplayBuffer`。
- TD 主链路已经使用 Double DQN 和 worker-local 3-step return。
- `EngineSnapshot` 可以精确保存并恢复 Pymunk 空间、队列、RNG 与 episode 状态；
  正式预检执行连续物理复现。
- 规则排序、物理反事实差值和极稀疏局部 Shapley 已接入同一个 Q 网络 optimizer。
- 反事实只跨进程传输确定性抽样后的候选，主进程再通过跨窗口优先级池和共享 token
  账本仲裁；完整状态分析、事件追踪和规则归因仍逐投放执行。

## 3. V1 总体结构

V1 把“什么有价值”“谁参与了结果”“怎样训练模型”分成三个层次：

```text
Reward V2
定义任务价值和局面质量
        |
        v
StateAnalyzer + AttributionTracker
识别实际轨迹中的机制贡献
        |
        v
CausalReplayBuffer + Q ranking loss
把历史贡献直接教给当前 Q 网络
        |
        v
稀疏反事实与局部 Shapley
只仲裁高价值歧义事件
```

必须遵守以下原则：

1. 水果存放时间不产生逐步奖励或惩罚；时间只用于确认一个结构问题是否持续。
2. 最高水果高度不直接决定奖励；风险由可投放通道和顶部连通空间表达。
3. 合成奖励按新水果等级近指数增长。
4. 同一个合成事件只生成一份任务价值，不因历史归因被复制多次。
5. 靠墙、相邻、等级有序等静态现象本身不发奖励；只有状态变化或后续兑现才形成归因事件。
6. 实际轨迹上的参与关系和“这个横坐标优于其它动作”是不同结论，分别记录置信度。
7. 高置信规则从第一次正式训练开始直接影响 Q 排序，不等待后续大规模验证。
8. 反事实任务不得阻塞正常采样；超出预算时必须丢弃低优先级任务。

## 4. Reward V2

### 4.1 总公式

V1 的环境奖励定义为：

\[
r_t =
r_t^{task}
+ \lambda_{\Phi}
\left[
\gamma\Phi(\bar s_{t+1})-\Phi(s_t)
\right]
\]

其中：

- \(r_t^{task}\)：真实合成事件的战略效用。
- \(\Phi(s)\)：可投放空间、可恢复性和连锁就绪度组成的状态 potential。
- \(\bar s_{t+1}\)：用于 shaping 的下一状态；真实终态使用零 potential。
- \(\gamma\)：与 DQN target 使用相同的折扣因子。

明确删除：

- 存活奖励；
- 最高水果高度变化奖励；
- 绝对危险高度持续惩罚；
- 默认固定终局惩罚。

游戏失败后的未来回报为零，已经表达失败代价。若短跑证明完全没有终局学习信号，再以独立
配置加入小额终局惩罚，不能默认恢复 `-100`。

### 4.2 合成任务效用

合成到等级 \(L\) 的默认效用为：

\[
g(L)=2^{(L-2)/2}, \qquad L\in[2,11]
\]

它每升两级翻倍，默认数值近似为：

| 新等级 | 效用 |
| --- | ---: |
| 2 | 1.000 |
| 3 | 1.414 |
| 4 | 2.000 |
| 5 | 2.828 |
| 6 | 4.000 |
| 7 | 5.657 |
| 8 | 8.000 |
| 9 | 11.314 |
| 10 | 16.000 |
| 11 | 22.627 |

一次物理稳定过程发生多个合成事件时：

\[
r_t^{task}=\sum_{e\in E_t}g(L_e)
\]

不额外添加固定 `chain_bonus`。连锁本身已经产生多个逐级增大的合成效用，再叠加固定连锁
奖励会重复放大同一结果。连锁信息用于历史贡献分配、因果回放分层和训练指标。

### 4.3 状态 potential

默认 potential 为：

\[
\Phi(s)=0.6C(s)+0.3R(s)+0.1K(s)
\]

各分量限制在 `[0, 1]`：

- \(C(s)\)：q0～q3 的顶部可达投放容量。
- \(R(s)\)：场上水果的可恢复性。
- \(K(s)\)：已经形成的连锁就绪结构。

默认：

```text
lambda_phi = 0.5
queue_decay = 0.5
```

短跑时必须检查单步 shaping 绝对值。默认要求其 p95 不超过 5 级合成效用的 25%，避免局面
塑形压过真实合成目标。

### 4.4 顶部可达投放容量

对于队列中的半径 \(r\)，在 15 个合法投放动作上估计：

- `landing_depth(a, r)`：该投放列可以深入场地的归一化深度。
- `safe(a, r)`：落点是否仍有最低安全深度。

定义：

\[
C_r(s)=
0.7\operatorname{mean}_a landing\_depth(a,r)
+0.3\operatorname{mean}_a safe(a,r)
\]

结合 q0～q3：

\[
C(s)=
\frac{\sum_{j=0}^{3}\eta^jC_{r(q_j)}(s)}
{\sum_{j=0}^{3}\eta^j},
\qquad \eta=0.5
\]

V1 可以先用离散投放列和膨胀圆形障碍估计，不要求求解完整动态落点。重要的是衡量从顶部
真正可使用的空间，而不是场地总面积减去水果圆面积。

### 4.5 水果可恢复性

对每颗水果 \(i\) 定义：

- `reachable_fraction_i`：同级待投水果能接近它的合法投放动作比例。
- `partner_reachable_i`：是否存在可到达的同级伙伴或可形成同级伙伴的局部路径。
- `burial_depth_i`：水果位于顶部可达边界之下的归一化深度。
- `low_level_weight_i`：低级水果更容易成为永久占位，权重更高。

不可恢复负担：

\[
B(s)=
\operatorname{normalize}
\sum_i
\omega(L_i)
(1-reachable_i)
(1-partner_i)
burial\_depth_i
\]

\[
R(s)=1-\operatorname{clip}(B(s),0,1)
\]

V1 不使用 `age_frames` 计算负担。年龄受物理模式和稳定帧配置影响，只能用于事件确认。

### 4.6 连锁就绪度

连锁 motif 至少包含：

- 两个可以相互接近或已经接触的同级水果；
- 预计合成位置附近存在下一级水果；
- 或者存在 `L -> L+1 -> L+2` 的接触/支撑阶梯；
- q0～q3 中有能够触发该结构的水果。

`K(s)` 根据最大 motif 深度、可达触发动作比例和队列兼容性归一化到 `[0,1]`。

`K(s)` 只作为 potential 的小权重分量。历史动作是否真正获得铺垫贡献，还必须等待 motif
被实际合成链使用。

### 4.7 terminal 与 truncated

- 真实游戏终态：\(\Phi(\bar s_{t+1})=0\)。
- 非终止物理截断：保留下一状态 potential，并允许 bootstrap。
- 物理异常且无法生成可信下一状态：丢弃该 transition，记录异常，不得伪装成正常死亡。

当前实现已经满足上述语义：terminal 传给 Reward V2 的 `next_analysis` 仍为
`None`，把下一 potential 强制为零；环境会额外生成只供 AttributionTracker 使用的
`post_action_state_analysis`，让终局责任依据动作后局面而不是动作前局面。该分析若
诊断无效，只会用 `terminal_geometry_untrusted` 撤销未决证据，不得确认负向事件。
truncated 生成同 episode 的降级 `analysis[t+1]`，保留 potential 和 bootstrap，
但 reset 后不会跨 episode 复用。

## 5. 状态分析模型

### 5.1 `StateAnalysis`

每个物理稳定边界生成一个只读分析对象：

```text
StateAnalysis
├── transition_key
├── incoming_transition_key
├── action_indices
├── action_drop_x_by_offset
├── fruit_analyses
├── free_space_probe_physics_radius
├── free_space_regions
├── support_edges
├── contact_influence_edges
├── partner_components
├── chain_motifs
├── queue_lane_analyses
├── queue_decay
├── top_connected_capacity
├── recoverability
├── chain_readiness
└── diagnostics
```

`transition_key` 必须包含：

```text
(worker_id, episode_id, step_index)
```

`fruit_id` 会在不同 episode 中重复，不能单独作为训练侧全局键。

`transition_key.step_index=t` 固定表示“动作 `t` 执行前的稳定边界”。因此跨步状态变化
统一比较 `analysis[t] -> analysis[t+1]`，真实终态也可以生成只供分析使用的 `t+1`
边界键。`contact_influence_edges` 是产生当前边界的前一动作证据，不是静态几何属性；
存在这类边时，`incoming_transition_key` 必须是同 worker、同 episode 的 `t-1`。

所有 15 位动作 mask 均按 `action_offset` 编位，而不是直接按外部 `action_index`
编位。`action_indices[a]` 保存 offset `a` 对应的环境动作号，
`action_drop_x_by_offset[a]` 保存当前 q0 的真实投放横坐标。

### 5.2 `FruitAnalysis`

每颗水果至少分析：

```text
fruit_id
level
physics_radius
probe_physics_radius
reachable_action_mask
reachable_action_count
top_visible_ratio
top_blocker_ids_by_action
partner_ids
partner_reachable
support_parent_ids
supported_child_ids
burial_depth
inversion_count
connected_region_id
reachable_partner_ids
critical_blocker_ids
inversion_blocker_ids
```

`reachable_action_mask` 使用 15 位掩码。第 `a` 位为 1，表示一个同级水果从动作 `a` 对应的
生成位置出发，在当前静态近似下仍能接近目标水果的可接触区域。

`physics_radius` 表示当前水果 shape 的真实半径；`probe_physics_radius` 表示未来
直接投放的同级水果用于路径膨胀的半径。合成生成水果与直接投放水果即使等级相同，
这两个半径也可能不同。`critical_blocker_ids` 和 `inversion_blocker_ids` 为负向规则
归因保留可回溯对象；`partner_reachable=True` 允许表示“当前没有现存同级伙伴，但仍有
形成伙伴的局部路径”，不能简单等同于 `bool(partner_ids)`。

### 5.3 队列槽位容量

q0 到 q3 分别生成 `QueueLaneAnalysis`：

```text
queue_index
level
physics_radius
drop_x_by_action
landing_depths_by_action
safe_action_mask
safe_action_count
blocker_ids_by_action
capacity
```

不同等级水果的合法横坐标范围不同，因此 q1 到 q3 必须保存各自的 15 个
`drop_x_by_action`，不能复用 q0 的横坐标。按动作排列的 tuple 均恰好 15 项，
mask 的 count 字段必须与 `bit_count()` 一致。schema 会校验每个槽位的
`capacity` 符合 4.4 节的 0.7/0.3 公式，并使用对象记录的 `queue_decay` 重新聚合
q0 到 q3，拒绝与 `top_connected_capacity` 不一致的重复真值。

### 5.4 V1 静态几何分析算法

当前 `StateAnalyzer` 使用两套互补、分别承担不同职责的静态近似：

1. q0～q3 落点容量、水果可达动作和 blocker 使用解析圆形竖直投放列；
2. 顶部连通自由空间和封闭空腔使用一个规范的最小直接投放水果探针栅格。

解析投放列在每个合法横坐标上，将场上水果按“当前真实物理半径 + 待投探针半径”
膨胀。投放列与膨胀圆的最上方交点决定第一落点：

- q0～q3 分别使用自身等级的直接投放半径和合法横坐标，不能共享 q0 横坐标；
- `landing_depth` 由生成线到第一障碍或地板接触点的深度归一化；
- `blocker_ids_by_action` 只记录形成该列第一落点的水果，几何并列时保留全部并列 ID；
- 分析目标水果时，只有目标是该列最先接触对象时才把对应动作记为可达；
- 位于目标接触点之前的水果进入该动作的 `top_blocker_ids_by_action`；仅当目标已经
  失去全部可达列时，最小阻挡集合才汇总为 `critical_blocker_ids` 并允许形成
  `caps`，避免把渐进路径损失提前标成最终封口。

该列模型有意不模拟落果沿曲面滚动、弹跳和后续碰撞重排。它为 15 个实际动作提供
确定、廉价且能回溯到水果 ID 的 blocker 证据；动态歧义仍由后续 tracker 的接触证据和
少量反事实处理。

自由空间分析统一使用等级 1 的直接投放物理半径作为
`free_space_probe_physics_radius`。分析器在探针圆心可进入的棋盘范围内建立规范栅格，
将与膨胀水果相交的单元标记为占用，对剩余单元执行四邻域 BFS。每个连通分量产生一个
`FreeSpaceRegionAnalysis`：

```text
region_id
top_connected
reachable_action_mask
cell_count
area_ratio
centroid_x / centroid_y
min_x / max_x / min_y / max_y
boundary_fruit_ids
touches_left_wall / touches_right_wall / touches_floor
```

接触顶部栅格行的分量属于顶部连通空间；其它达到最小单元阈值的分量属于封闭空腔，
且其动作 mask 必须为零。区域 ID 由当前状态的规范栅格确定，只保证单状态内稳定，
不是跨步永久身份。tracker 后续比较 `analysis[t] -> analysis[t+1]` 时必须结合面积、
质心、包围盒重叠和边界水果匹配区域，不能只比较 `region_id`。

静态支撑图使用几何接触容差生成：

- 地板到水果的 `supports`；
- 左右墙到水果的 `wall_constraint`；
- 下方水果到上方水果的 `supports`；
- 上方 blocker 对下方水果的 `caps`；
- 两侧支撑共同形成上方横向结构时的 `bridges`。

同级水果在局部范围内接触或共享可达动作时连成伙伴图，并输出至少包含两颗水果且内部
连通的确定排序 `PartnerComponent`。当前存在顶部接近路径时，
`partner_reachable` 也可以表示未来仍能形成同级伙伴，不要求场上已经存在伙伴。
`merge_pair` 和邻近下一级水果组成的
`level_ladder` 是第一版 `ChainMotif`；只有 q0～q3 存在兼容等级且结构仍有实际触发
动作时才产生正的 motif。它们只用于 `chain_readiness` 和后续兑现追踪，不会直接复制
合成奖励。

`recoverability` 按 4.5 节从可达比例、伙伴路径、埋藏深度和低级权重聚合。当前埋藏深度
是相对生成线和地板的几何深度，年龄仍不参与计算。当前只要还存在直接入口，就认为
未来同级伙伴仍可形成，因此该分量保守地集中惩罚“零入口死果”；渐进减少的 15 位路径
仍完整保留给容量和后续 tracker 使用。ladder 暂以 pair 中点近似未来合成位置，实际
引擎合成坐标可能偏向较低父水果，该误差属于静态近似。常规分析属于动作前稳定边界的
worker-local 只读快照；真实 terminal 另有只供归因使用的动作后分析。环境已经缓存
相邻分析并接入 Reward V2，collector 负责提供稳定 `TransitionKey`、调用
`AttributionTracker` 和聚合性能统计。完整分析与事件仍不进入主 replay；独立
`CausalReplayBuffer` 使用身份/动作对索引和分层随机桶，容量满时保持有界淘汰，
checkpoint 会精确恢复样本、索引状态和独立 RNG。

### 5.5 四张基础图

#### 合成谱系图

叶节点是历史投放，内部节点是 `MergeEvent`。新水果保存两个父水果 ID，能够递归追溯到
所有原始投放。

#### 接触影响图

每次投放只保存压缩后的实际影响：

- 接触水果对；
- 显著位移；
- 可用时保存最大冲量；
- 当前落果到旧水果合成之间的接触路径。

不保存每个物理帧的完整碰撞日志。

#### 支撑图

记录水果与水果、墙壁、地板之间的稳定支撑关系。V1 至少需要区分：

- 被地板支撑；
- 被墙壁限制横向移动；
- 被另一水果从下方支撑；
- 对上方水果形成盖压或桥接。

#### 可达图

针对真实物理半径膨胀障碍，从生成线沿离散动作列或栅格自由空间判断：

- 哪些动作列仍能接近目标水果；
- 每条路径由哪些水果阻挡；
- 哪些水果组成最后的阻挡割集；
- 哪些顶部连通区域已经变为封闭空腔。

V1 不要求严格求解连续空间最小割；15 条离散动作路径的阻挡集合已经足以完成第一版归因。

## 6. 归因事件数据模型

### 6.1 事件结构

统一事件模型：

```text
AttributionEvent
├── event_id
├── episode_key
├── attribution_version
├── tracker_config_fingerprint
├── detected_step
├── resolved_step
├── event_type
├── status
├── sign
├── target_fruit_ids
├── contributors
├── utility
├── link_confidence
├── placement_confidence
├── evidence
├── budget_key
└── delay
```

状态：

```text
pending
confirmed
cancelled
```

truncated、手动 reset 和 worker shutdown 属于证据中断，不得成为负向训练样本。
V1 的落地结构仍用 `cancelled` 完成不可变事件的收口，但通过
`resolution_reason=truncated/manual_reset/worker_shutdown` 和独立
`interrupted_pending_count` 区分；统计取消率时必须把这些中断原因单列。

贡献者结构：

```text
Contributor
├── transition_key
├── action_index
├── fruit_id
├── evidence_type
├── evidence_weight
└── role
```

角色可以是：

```text
material
trigger
mechanical_trigger
support
path_opener
path_blocker
motif_creator
motif_breaker
rescuer
victim_drop
```

### 6.2 两类置信度

`link_confidence` 表示实际轨迹中存在机制连接的可信度：

- 合成谱系 ID：接近 1.0。
- 唯一接触影响链：0.9 以上。
- 稳定可达性割集变化：0.75～0.9。
- 单纯几何相关：不高于 0.7。

`placement_confidence` 表示当前横坐标优于其它合法动作的可信度：

- 明确存在安全替代动作且当前动作完成唯一封口：0.8 以上。
- 实际铺垫后来兑现，但未比较替代动作：0.55～0.8。
- 只知道水果进入谱系：不能自动设为高置信。

训练路由：

| 等级 | 条件 | 用途 |
| --- | --- | --- |
| A | `link >= 0.90` 且 `placement >= 0.80` | 完整规则排序权重 |
| B | `link >= 0.75` 且 `placement >= 0.55` | 0.25～0.5 规则排序权重 |
| C | 其它结构候选 | 日志、potential 或反事实候选 |

## 7. 正贡献事件

### 7.1 实际合成相关

#### `MERGE_LINEAGE`

某历史投放的水果或其后代成为合成事件材料。谱系链接置信度为 1.0，但投放位置置信度需要
结合其它证据。

#### `DIRECT_TRIGGER`

当前投放水果或其后代进入本步第一场合成。

#### `MECHANICAL_TRIGGER`

当前水果没有进入合成谱系，但接触影响链推动两个旧水果发生合成。

#### `CHAIN_TRIGGER`

同一次物理稳定过程中，新合成水果继续成为后续事件 source。整条连锁作为复合事件分析，
当前触发动作不能独占所有历史铺垫贡献。

### 7.2 已兑现铺垫

#### `REALIZED_ADJACENCY`

某动作首次建立后来被实际合成使用的同级邻接、同一可达盆地或关键接触边。

#### `REALIZED_LADDER`

某动作补全 `L -> L+1 -> ...` 等级阶梯，且后续真实连锁沿该结构发生。

#### `WALL_ANCHOR_REALIZED`

小水果在墙边保持顶部可达、没有下沉，并在后续成为实际合成谱系。

#### `SUPPORT_PATH_REALIZED`

墙壁、地板或水果支撑关系被后续连锁真实使用。

### 7.3 空间恢复

#### `PARTNER_CONNECTED`

两个同级水果从不同不可达区域进入同一可达区域，随后真实合成。

#### `FRUIT_RESCUED`

不可达水果重新变为可达，并在确认窗口或本局稍后参与合成。

#### `CORRIDOR_OPENED_USED`

动作打开新的投放通道，后续水果真实经过该区域并产生合成。

当前 tracker 不会仅凭静态可达性恢复发射该事件；在物理层尚未提供“水果真实经过
新通道”的路径证据前，该类型保持禁用，避免把普通局面变化伪装成通道贡献。

#### `INVERSION_RESOLVED`

低级水果从高级水果下方恢复到顶部可达区域。

#### `BLOCKER_CLEARED`

高级合成或推动移除了实际割点，使多个动作列或多个水果重新可达。

## 8. 负贡献事件

### 8.1 埋死与封路

#### `BORN_BURIED`

水果投放稳定后即无任何顶部可达路径。主要责任归原始投放动作。

#### `REACHABILITY_SEALED`

水果从至少一条可达路径变为零条路径，并在确认窗口内没有恢复。

#### `PUSHED_BURIED`

当前落果的接触影响链使旧低级水果明显下移，同时减少顶部开口或可达动作数量。

#### `MERGE_EXPANSION_BLOCK`

合成产生的大水果切断原有投放通道。该合成仍保留正任务效用，同时记录独立的负空间副作用。

#### `CORNER_CAPPED`

墙边小水果被另一水果从上方封住，墙壁与盖子共同切断路径。

#### `ROOF_BRIDGE_CREATED`

若干水果形成稳定横向桥接或屋顶，封闭下方区域。

#### `SEALED_CAVITY_CREATED`

顶部连通自由空间中形成新的封闭空腔。

### 8.2 合成机会破坏

#### `PARTNER_ISOLATED`

原本位于同一可达区域的同级水果被分割，且没有其它可达同级伙伴。

#### `PAIR_SCATTERED`

明确具备合成机会的同级水果被实际接触影响链显著撞散。

#### `LAST_LANE_BLOCKED`

当前动作占据目标水果或 q0～q3 的最后安全投放路径。

#### `CHAIN_MOTIF_BROKEN`

已存在的关键等级阶梯、支撑边或触发位置被本步实际影响链破坏。

### 8.3 层级与终局

#### `LARGE_BLOCKER_CREATED`

新大水果成为多个目标或多个动作列的共同割点。不能只因水果体积大而触发，必须有真实
可达性损失。

#### `INVERSION_CREATED`

低级水果被压入高级水果下方，并同时失去顶部路径或伙伴路径。

#### `TERMINAL_SUPPORT_CREATED`

某动作建立高位支撑平台或最后阻挡，使水果持续无法下落并触发真实终局。

#### `DEAD_LOW_FRUIT_CONFIRMED`

低级水果直到确认窗口或终局仍不可达、无伙伴路径且未进入任何合成谱系。责任回溯到最早
封路事件，而不是确认时的最后一步。

## 9. 不允许单独发奖罚的静态信号

以下只能生成候选、potential 或诊断指标：

- 水果存在时间长；
- 水果位于底部；
- 水果靠墙；
- 最高水果很高；
- 两个同级水果距离近；
- 小水果周围存在高级水果；
- 当前总圆面积；
- 当前 `empty_space_ratio`；
- 视觉上存在等级阶梯但没有后续兑现；
- 单纯“底部等级应该更高”的全局相关性。

如果静态状态没有发生转变、没有实际影响链、没有后续兑现，则不能建立历史因果样本。

## 10. 埋死水果确认与责任回溯

### 10.1 待定封路

若：

\[
|R_i(s_t)|>0,\qquad |R_i(s_{t+1})|=0
\]

则为水果 \(i\) 创建 `pending REACHABILITY_SEALED`，记录：

- `seal_step`；
- 本步消失的动作路径；
- 每条路径的新阻挡水果；
- 当前投放水果；
- 本步接触影响链；
- 前后顶部可见比例；
- 前后伙伴可达性。

### 10.2 V1 默认确认阈值

```text
transient_stable_steps = 3
burial_confirm_steps = 12
```

- 连续 3 个稳定状态仍不可达：排除短暂抖动。
- 连续 12 次投放仍不可达、没有伙伴路径且位置没有显著改善：确认长期埋死。
- episode 真实终止时仍满足条件：立即确认。
- 期间重新可达或进入合成谱系：取消待定负贡献。

12 次投放只决定是否确认，惩罚不会随等待时间线性增加。

真实终局仍保持 Reward V2 的 `next potential = 0`，但环境会额外生成
`post_action_state_analysis` 供 tracker 检查终局动作后的可达性。它不会进入 reward、
主 replay 或 DQN bootstrap。这样终局动作刚好解封时可以取消 pending，新建稳定支撑
割点并触发终局时则可记录 `TERMINAL_SUPPORT_CREATED`。

### 10.3 单步与渐进责任

单步封死时，主要责任归使最后路径消失的动作。

渐进封死时，每个历史动作的证据为：

\[
e_j=
\alpha\Delta blocked\_lanes_j
+\beta\Delta top\_occlusion_j
+\eta\Delta critical\_contacts_j
+\zeta\Delta downward\_displacement_j
\]

默认只要求四项归一化后非负，具体系数在 2%～5% 小跑中按数量级校准。贡献权重为：

\[
w_j=\frac{confidence_j e_j}
{\sum_k confidence_k e_k}
\]

若当前落果没有成为最终阻挡物，但通过接触链推动旧水果完成封路，责任仍归当前动作。

### 10.4 解封

待定封路被解除时：

- 将原事件标记为 `cancelled`，不得写入负向因果训练样本；
- 若解封后水果真实合成，生成 `FRUIT_RESCUED` 正贡献；
- 只恢复可达但未被使用时，仅计入 potential，不产生高置信历史正归因。

## 11. 事件预算与共享贡献

### 11.1 单一价值包

每个合成事件只有一个任务价值包：

\[
U_e=g(L_e)
\]

规则归因不向环境 scalar reward 复制新的 \(U_e\)。历史贡献通过额外 Q 排序监督进入训练。

同一步连锁仍保留各个真实合成事件自己的 \(U_e\)，但不得再额外创建与其总价值重复的固定
连锁奖励。

### 11.2 贡献证据

候选动作的未归一化贡献：

\[
\hat w_i =
c_i
\left(
\alpha L_i
+\beta T_i
+\eta S_i
+\zeta A_i
\right)
\]

其中：

- \(L_i\)：谱系材料证据。
- \(T_i\)：直接或机械触发证据。
- \(S_i\)：已兑现支撑、墙边锚点或等级阶梯证据。
- \(A_i\)：通道、伙伴可达性、解封或封路证据。
- \(c_i\)：综合置信度。

同一事件内归一化：

\[
w_i=\frac{\hat w_i}{\sum_j\hat w_j}
\]

V1 不把共享贡献固定写成最近动作平均或全部 `50/50`。谱系中的两个同级父节点可以递归
各分一半作为材料先验，但最终动作训练权重还要结合触发、结构和可达性证据。

### 11.3 负向结构预算

埋死、封路、空腔和通道损失属于独立的局面质量事件，不从合成价值中偷偷重复扣除。

其规则排序 margin 根据以下因素计算：

- 丢失的安全动作比例；
- 受影响水果数量；
- 目标水果低等级权重；
- 顶部连通空间损失；
- 事件确认置信度。

默认对单个负向事件的目标 margin 做裁剪，最大不超过 5 级合成效用，避免一个几何误判主导
整批梯度。

## 12. 因果训练数据

### 12.1 主 ReplayBuffer

现有 `TensorTransition` 和冷热 `ReplayBuffer` 保持任务 TD 训练职责，不追加可变历史字段。

这样避免：

- 修改 frozen transition；
- 随机更新冷磁盘段；
- 重复插入相同状态动作但 reward 不同的经验；
- 让因果标注阻塞正常 transition 采集。

### 12.2 `CausalReplayBuffer`

新增纯内存稀疏回放：

```text
CausalSample
├── graph
├── actual_action_offset
├── comparison_action_offset
├── direction
├── target_margin
├── confidence
├── cause_type
├── delay
├── transition_key
└── attribution_version
```

默认：

```text
causal_replay_capacity = 20000
causal_batch_size = 32
causal_update_interval = 2
```

按以下三类分层采样：

```text
positive_setup
negative_blocking
counterfactual
```

### 12.3 规则比较动作

规则样本必须提供实际动作和比较动作：

- 正铺垫：实际动作应高于没有建立关键结构、但仍合法的结构较差动作。
- 负封路：实际动作应低于保留最多可达路径的安全动作。
- 左右对称状态优先选择镜像动作作为检查样本。
- 没有可信比较动作时，不生成高权重规则排序，只记录事件或触发反事实。

### 12.4 规则排序损失

令：

- \(y=+1\)：实际动作优于比较动作。
- \(y=-1\)：实际动作劣于比较动作。
- \(m\)：按事件价值、贡献权重和置信度得到的目标 margin。

\[
L_{rule}
=
c
\max
\left(
0,
m-y[Q(s,a)-Q(s,a^{cmp})]
\right)
\]

默认：

```text
lambda_rule = 0.15
```

### 12.5 反事实差值损失

反事实得到：

\[
\Delta J =
J(s,a)-J(s,a^{cf})
\]

使用：

\[
L_{cf}
=
c\operatorname{Huber}
\left(
Q(s,a)-Q(s,a^{cf}),
\operatorname{stopgrad}(\Delta J)
\right)
\]

默认：

```text
lambda_cf = 0.10
```

反事实回报差进入 loss 前必须按当前 Reward V2 尺度归一化并裁剪。

### 12.6 总训练损失

\[
L =
L_{DoubleDQN,3step}
+\lambda_{rule}L_{rule}
+\lambda_{cf}L_{cf}
\]

第一次大规模训练必须启用：

- Double DQN；
- 3-step return；
- 规则排序；
- 稀疏反事实差值监督。

V1 不增加独立 causal head，直接约束现有 Q 动作排序，以减少实现范围并让归因从第一次长训
开始产生策略影响。

## 13. 稀疏反事实

### 13.1 使用范围

反事实只用于：

- 新等级不低于 7 的高价值合成；
- 两段及以上连锁；
- 多个动作都可能是封路原因；
- 同一动作同时存在强正和强负证据；
- 规则 `placement_confidence` 位于中间区域；
- 少量随机规则审计样本。

高置信谱系、唯一机械触发和唯一封路割点默认不运行反事实。

### 13.2 替代动作

正式首轮配置中，每个任务最多比较两个替代动作，按以下顺序稳定去重：

1. 左右镜像动作；
2. 最近的结构安全动作；
3. 当前冻结策略的 runner-up 动作（只有前两项去重后仍有空位时使用）。

不枚举全部 15 个动作。

### 13.3 重演范围

默认：

```text
counterfactual_horizon = 10
counterfactual_horizon_min = 8
counterfactual_horizon_max = 12
```

- 固定相同队列、RNG 和物理配置。
- 结果已经确定时提前停止。
- 到达 horizon 后使用冻结 target Q bootstrap。
- 数值重演无法复现原始动作结果时，丢弃该任务并记录。

### 13.4 性能预算

默认：

```text
counterfactual_cost_ratio = 0.06
counterfactual_cost_hard_limit = 0.10
counterfactual_external_token_reserve_ratio = 0.01
counterfactual_min_real_steps = 256
counterfactual_cpu_core_ratio = 0.34
counterfactual_queue_capacity = 256
snapshot_ring_size = 32
counterfactual_proposal_sample_rate = 0.0625
counterfactual_max_alternatives = 2
counterfactual_workers = 2
```

解释：

- 普通反事实等价物理步以 6% 为软预算，高优先级普通任务最多借用到 9%；
- 最后 1% 专供已经被 selector 选中的局部 Shapley。外部任务仍需等真实步数累积出
  足够额度，普通任务不能提前耗尽该份额；
- 普通反事实与局部 Shapley 合计不得超过 10%；
- 任何情况下不得超过 10%；
- 每 256 个真实投放最多创建一个任务；
- 首轮云容器的 6 个 cgroup 有效核配置使用 3 个 rollout 与 2 个物理归因进程，
  其中 1 个供 Shapley，并为主进程保留 1 核；启动器同时按 affinity/cgroup 配额和
  34% 比例验算，所有物理子进程各限制为 1 个 Torch CPU 线程；
- 常规 proposal 只按稳定 SHA-256 身份抽取 1/16 跨进程传输，7 级以上高价值合成
  无条件保留；
- 主进程候选池跨 256 步窗口保留更高优先级事件，整批完成 Shapley 路由后再按
  `priority + proposal_id` 确定性派发，避免“第一个到达者”占用稀缺槽位；
- 候选池或执行队列满时淘汰低优先级任务，不能阻塞 rollout；
- 每个 worker 保存最近 32 个稳定边界的快照环。

### 13.5 局部 Shapley

局部 Shapley 仅用于最高价值且存在明显协同歧义的事件：

```text
shapley_event_ratio_max = 0.0005
shapley_candidate_limit = 3
shapley_paired_permutations = 4
```

即最多选择约 0.05% 的事件，候选历史动作不超过 3 个，使用少量正反排列估计共享贡献。
未选中的普通观察身份保存在有界 LRU；真正被 selector 选中的身份永久去重，避免长训中
无界保存所有 proposal，又不会让已消耗 Shapley 配额的事件在 LRU 淘汰后重复进入。
关闭时还必须满足 `selected = completed + failed + terminal_dropped`。其中
`terminal_dropped` 只解释未执行任务的最终去向，不能冒充物理结果；只要 selected
大于零，阶段门禁仍要求没有 terminal drop，并至少有一个通过复现、样本写入和 optimizer
消费的完成结果。

## 14. 完整物理快照

`EngineSnapshot` 至少保存：

```text
fruit bodies:
    fruit_id
    level
    physics_radius
    position
    velocity
    angle
    angular_velocity
    age_frames

episode state:
    score
    last_score
    fail_count
    alive
    step_count
    physics_frame
    next_fruit_id
    fruit_queue
    RNG state

physics config:
    fps
    space_iterations
    stable thresholds
    relevant collision parameters
```

快照测试必须先验证：

```text
snapshot
-> restore
-> execute original action
-> reproduce original merge event IDs/levels and terminal result
```

如果 Pymunk 数值顺序导致轻微位置误差，可以使用几何容差；但合成事件、等级、得分和终止结果
必须一致，否则该快照不能用于反事实标签。

## 15. 并行采集要求

- 每个 worker 独立维护 `StateAnalyzer`；`AttributionTracker` 和快照环实现后也必须
  保持同一 worker-local 边界。
- 全局键必须使用 `(worker_id, episode_id, step_index)`。
- tracker、事件和样本类型必须定义在模块顶层并可 pickle，兼容 Windows `spawn`。
- 因果任务和结果通过独立有界队列传递。
- 主采集统计区分真实环境步、因果样本数和反事实模拟步。
- worker 关闭前记录未确认事件数量；不得静默把 pending 事件当作已确认。

## 16. 训练日志

第一次长训至少记录：

### 任务表现

- 原始游戏 score；
- Reward V2 总量及各组成；
- 各等级合成次数；
- 最大水果等级；
- 最大连锁深度；
- 每局投放次数；
- terminated/truncated 比例。

### 状态归因

- 各 `cause_type` 的 confirmed/cancelled 数量；
- pending 事件数量；
- 平均及 p95 归因延迟；
- 新埋死水果数；
- 持续埋死水果数；
- 被解救水果数；
- 等级倒置创建/解除数；
- 封路/解封事件数；
- 平均可达动作数；
- 顶部连通容量；
- 正负因果样本比例；
- A/B/C 置信度分布。

### 训练信号

- `td_loss`；
- `rule_rank_loss`；
- `counterfactual_loss`；
- 规则 batch 命中率；
- 正负 pair 排序正确率；
- 因果样本 buffer 大小；
- 各原因类型采样比例；
- 规则与反事实方向一致率。

### 性能

- `StateAnalyzer` 调用次数、总/平均耗时、缓存命中率和降级率；（已接入）
- potential shaping 绝对值 p95；（已接入）
- `AttributionTracker` 调用次数、事件 created/confirmed/cancelled/interrupted、
  当前 pending、同一步最大连锁深度和归因延迟；（已接入）
- 反事实模拟步数；
- 反事实成本比例；
- 反事实队列长度和丢弃任务数；
- 因果训练额外 GPU 时间；
- 总环境步吞吐变化。

## 17. 必须覆盖的测试场景

### 状态分析

- 墙边小水果保持可达。
- 墙边小水果被大水果盖帽。
- 单条投放通道从可达到封死。
- 多步渐进减少通道。
- 顶部连通空间与封闭空腔面积相同但可用性不同。
- 同级水果被分割到不同区域。
- 等级阶梯和随机散布具有不同 `chain_readiness`。
- 左右镜像得到镜像分析结果。

### 事件生命周期

- 封路后 3 步内恢复，事件取消。
- 封路持续 12 步，事件确认一次。
- 终局时仍封路，回溯最初责任动作。
- 解封后真实合成，产生正营救事件。
- 时间增加不会重复产生惩罚。
- 同一步多级连锁不遗漏、不重复创建额外价值包。

### 谱系与并行

- 单次合成的两个父 ID 正确。
- 多级合成可以追溯全部根投放。
- reset 后相同 fruit ID 不会跨 episode 串线。
- 两个 worker 的相同 fruit ID 不会冲突。

### Reward V2

- 存活时间不改变 reward。
- 单独最高点变化不直接改变 reward。
- potential shaping 满足轨迹望远镜关系。
- 真实 terminal potential 为零。
- truncated 保留下一状态 potential 和 bootstrap。

### 因果训练

- 正 pair 提高实际动作相对 Q。
- 负 pair 降低实际动作相对 Q。
- 置信度为零的样本不产生梯度。
- cause type 分层采样不会长期饿死某一类别。
- 主 ReplayBuffer 格式保持兼容。

### 反事实

- 恢复后原动作可复现。
- 队列和 RNG 在所有分支一致。
- 三个替代动作数量上限生效。
- horizon 和提前停止生效。
- token budget、CPU 和队列上限生效。
- 无法复现的任务被丢弃而不是生成伪标签。

## 18. 第一次大规模训练前流程

### 18.1 烟测

执行约 5000 次更新或等价短运行，只检查：

- 崩溃和 NaN；
- episode/fruit ID 串线；
- 事件重复记账；
- pending 事件能否确认和取消；
- loss 数量级；
- 队列和并行采集是否死锁；
- 反事实预算是否生效。

烟测不用于判断是否启用完整归因。

### 18.2 规模校准小跑

执行正式规模的 2%～5%，检查：

- 单步 shaping p95；
- 每局归因事件数量；
- 正负样本比例；
- 事件取消率；
- A/B/C 置信度比例；
- `rule_rank_loss` 与 `td_loss` 数量级；
- 反事实成本；
- 原始 score 是否能够正常学习。

允许在小跑后调整数值阈值，但调整必须写入正式 TOML 配置并更新
`attribution_version`。不得在第一次大规模训练过程中无记录地修改归因语义。

### 18.3 第一次大规模训练

第一次大规模训练必须同时启用：

```text
Reward V2
Double DQN
3-step return
完整 StateAnalyzer
完整 AttributionTracker
延迟确认和撤销
CausalReplayBuffer
规则 Q 排序
配置化软预算的稀疏反事实（首轮正式训练为 6%，共享硬上限为 10%）
极稀疏局部 Shapley
完整归因与性能日志
```

不运行纯旧 reward 或纯无归因的大规模基线。

## 19. 后续两次训练的用途

第二次大规模训练根据第一次结果调整：

- potential 与任务奖励比例；
- 埋死确认阈值；
- 规则损失权重；
- 各原因类别采样比例；
- 反事实触发阈值；
- 反事实是否带来可测增益。

如果有第三次：

- 完整方案明显更好：更换训练 seed 复现最佳配置；
- 反事实无明显增益：保留规则归因，关闭或进一步缩小反事实；
- 规则与反事实长期冲突：只修改冲突事件类型，不扩大全量 Shapley。

## 20. 推荐源码结构

实现阶段建议新增：

```text
src/daxigua_rl/attribution/
├── __init__.py
├── schema.py
├── state_analyzer.py
├── tracker.py
├── causal_replay.py
└── counterfactual.py
```

职责：

- `schema.py`：状态分析、事件、贡献者、因果样本和配置 dataclass。
- `state_analyzer.py`：可达性、支撑、伙伴、连锁 motif 和 potential 分量。
- `tracker.py`：谱系、pending 事件、确认、撤销和责任回溯。
- `causal_replay.py`：内存因果回放和分层采样。
- `counterfactual.py`：快照任务、预算调度、分支重演和局部 Shapley。

对应测试：

```text
tests/test_state_analyzer.py
tests/test_attribution_tracker.py
tests/test_causal_replay.py
tests/test_causal_training.py
tests/test_counterfactual.py
```

## 21. 实现顺序

1. 修正 terminated/truncated 语义。（已完成）
2. 暴露真实物理半径和完整 episode/worker 键。（已完成）
3. 实现 `StateAnalysis` schema。（已完成）
4. 实现 15 动作可达性、顶部连通容量、封闭空腔和支撑图。（已完成）
5. 实现 Reward V2。（已完成）
6. 实现谱系、正负归因事件和 pending 生命周期。
   （已完成；接触依赖型事件只在存在显式 `ContactInfluenceEdge` 时发射，
   `CORRIDOR_OPENED_USED` 等待真实路径使用证据。）
7. 实现 `CausalReplayBuffer` 与规则排序 loss。
8. 将 DQN 改为 Double DQN 和 3-step return。
9. 增加归因日志和人工场景测试。（tracker 部分已完成，因果训练指标随第 7 步补齐。）
10. 实现 `EngineSnapshot` 和原动作复现。
11. 实现预算反事实和极稀疏局部 Shapley。
12. 完成烟测、小跑和第一次完整状态归因长训。

## 22. V1 非目标

V1 不追求：

- 对每个动作进行完整因果证明；
- 构建长期可扩展的通用强化学习框架；
- 全量枚举 15 个动作；
- 重演完整 episode；
- 对所有历史动作计算精确 Shapley；
- 用复杂学习模型替代已有游戏谱系和物理规则；
- 在第一次训练前完成大量消融实验；
- 用人类策略硬编码固定左墙或固定分区奖励。

## 23. 已知风险与降级策略

| 风险 | 降级方式 |
| --- | --- |
| 可达性静态近似误判动态滚动 | 降低 placement confidence，交给少量反事实审计 |
| fast30 与精确物理的事件分布不同 | 快照和反事实使用同一训练物理模式，并记录模式 |
| 规则样本数量过多压过 TD | 分层限额、降低 `lambda_rule`、裁剪 margin |
| 封闭空腔区域 ID 不能跨状态直接对齐 | ID 只作单状态引用；跨步结合面积、质心、包围盒重叠和边界水果匹配 |
| 负封路标签过多 | 提高确认阈值，要求伙伴路径同时消失 |
| 反事实队列积压 | 丢弃低价值任务，不阻塞 rollout |
| 快照无法确定性复现 | 关闭反事实损失，保留规则归因继续训练 |
| cause type 极度不平衡 | CausalReplayBuffer 分层采样 |
| 规则与反事实方向冲突 | 降低对应事件类型置信度，不扩大全量反事实 |
| 归因日志过重 | 保留聚合指标，仅抽样保存详细事件 |

即使反事实模块失败，Reward V2、状态分析、规则归因、延迟确认和 Q 排序仍必须能够独立完成
第一次大规模训练。
