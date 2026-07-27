# RL 接口 v0

文件名中的 `v0` 保留最初“先闭合游戏到训练”的接口沿革；本文内容已经同步到第一次
完整因果归因训练前的当前实现。历史设计依据仍以
`CAUSAL_ATTRIBUTION_V1.md` 为准。

## 目标

本接口用于先跑通强化学习训练闭环：

```text
reset -> observe -> choose action -> step -> reward / next_state / done
```

当前提供无渲染游戏接口、带 StateAnalyzer/Reward V2 的 RL 环境壳层、GNN 图构建
基础设施、GNN-Q 前向模型、`EngineSnapshot`、3-step `TensorTransition`、
冷热 `ReplayBuffer`、独立 `CausalReplayBuffer`、worker-local
`AttributionTracker`、单/多进程 collector、预算反事实、局部 Shapley、
Double DQN 联合更新器、版本化 checkpoint/hot-resume 和正式训练前门禁。

## 边界

- `daxigua.core.engine.HeadlessGame` 属于游戏本体，只负责规则、物理、状态和动作候选。
- `daxigua_rl.env.DaxiguaEnv` 属于 RL 包，只通过 `HeadlessGame` 访问游戏。
- `daxigua` 不允许 import `daxigua_rl`。
- `daxigua_rl` 的训练、环境、模型和图构建代码不应 import `daxigua.app.Board`、pygame 渲染、HUD、音频或手动输入代码。
- 视觉观看脚本可以作为例外懒加载 `daxigua.app.Board`，用于把模型接到真实游戏窗口上检查实际游玩效果；该例外不能反向污染训练接口。

## 游戏本体接口

### `HeadlessGame`

主要方法：

- `reset(seed=None, fruit_queue=None) -> GameState`
- `get_state() -> GameState`
- `get_action_candidates(k=15) -> tuple/list[ActionCandidate]`
- `drop_at(x) -> DropResult`
- `advance_physics(max_frames=None, until_stable=True, stable_frames=15) -> PhysicsResult`
- `is_done() -> bool`

`HeadlessGame` 的一轮训练动作通常是：

```text
drop_at(x)
advance_physics(...)
get_state()
```

`HeadlessGame` 支持训练侧配置物理步频和 Pymunk 迭代次数：

- `fps`: 每次 `space.step(1 / fps)` 的物理积分步长来源。
- `space_iterations`: Pymunk 每个物理步的约束求解迭代次数。

这两个参数用于后续 accurate/fast 训练模式对比。游戏表现层仍使用自己的渲染循环，不依赖 RL 对比脚本。

当 `until_stable=True` 时，`PhysicsResult.stable` 只会在连续满足
`stable_frames` 帧稳定条件后为真。达到 `max_frames` 但没有完成这段连续窗口时，
返回 `truncated=True`，不能把最后一帧的瞬时静止当作已经稳定。

### `EngineSnapshot`

`HeadlessGame` 当前提供：

- `capture_snapshot(canonicalize=True) -> EngineSnapshot`
- `restore_snapshot(snapshot) -> GameState`
- `HeadlessGame.from_snapshot(snapshot) -> HeadlessGame`
- `execute_action(drop_x, ...) -> EngineActionOutcome`
- `HeadlessGame.replay_action(snapshot, drop_x, ...) -> EngineActionOutcome`
- `HeadlessGame.replay_and_compare_original_action(snapshot, expected_outcome, ...)`

快照只能在 reset 后或完全稳定、非终局、Space 未锁定且没有 pending 回调的动作边界
捕获。它不只是公开 `GameState`：除水果、边界、队列、分数、ID、RNG 和物理配置外，
还保存 Pymunk `Space` 的序列化内部状态，包括 cached arbiters、接触点、shape ID
counter、timestamp 和 timestep。恢复前会验证 schema、checksum、配置指纹、
`pymunk` / Chipmunk 精确版本及不支持的约束/睡眠/线程状态。

默认 `canonicalize=True` 会先用快照规范化真实分支的内部表示，再返回第二份快照。
这是为了让继续运行的 factual 分支与重建的反事实分支从相同 broadphase/求解器状态
出发；规范化前后公开 `GameState` 必须完全相同。

原动作比较要求水果 ID、等级、合成顺序、得分、队列和终止/截断结果一致，只对已知的
亚像素坐标差采用小容差。未通过 factual reproduction 的分支只能记录失败原因，不能
生成反事实或 Shapley 标签。

## RL 环境接口

### `DaxiguaEnv`

主要方法：

- `reset(seed=None, fruit_queue=None) -> (GameState, info)`
- `DaxiguaEnv.from_snapshot(snapshot, config=None, state_analyzer=None) -> DaxiguaEnv`
- `action_candidates() -> list[ActionCandidate]`
- `step(action_index, *, transition_key=None) -> (GameState, reward, terminated, truncated, info)`

这里的 `step(action_index)` 表示一次完整投放，不是一帧游戏画面。

`from_snapshot()` 创建一个可直接执行下一动作的 live 环境。快照不保存 Reward V2 或
StateAnalyzer 配置，因此调用方仍提供环境侧配置；其中 `physics_fps` 和
`space_iterations` 必须与快照一致。快照也不携带 rollout worker/episode 身份，
正式重演仍由调用方提供对应 `TransitionKey`。

正式 rollout 会在动作执行前传入
`TransitionKey(worker_id, episode_id, step_index)`。直接调用环境时可以省略，
环境会使用 worker 0 和本地递增的 episode 编号；无论采用哪种方式，
`step_index` 都必须等于当前 `GameState.step_count`。

终止语义：

- `terminated=True`: 游戏规则意义上的真实终局。
- `truncated=True`: 物理推进达到上限但尚未稳定；这是采集 episode 边界，但不是
  MDP 终态，返回的可信下一状态仍用于 potential 和 DQN bootstrap。
- `truncated` 后 collector 会重置环境，后续归因 tracker 应把未确认事件标记为中断，
  不能按真实终局规则确认。

`DaxiguaEnvConfig` 中和物理速度相关的字段：

- `physics_fps`: headless 物理步频，默认使用项目 `FPS`。
- `max_physics_frames`: 一次投放后最多推进多少物理帧。
- `stable_frames`: 连续多少帧稳定后结束当前 step。
- `space_iterations`: Pymunk 每个物理步的迭代次数。

状态分析配置通过 `state_analyzer_config: StateAnalyzerConfig` 传入。
`DaxiguaEnv` 构造时在当前进程内创建分析器；并行训练只把可 pickle 的配置送入
Windows spawn worker，不从主进程传递分析器实例。

当前环境使用 Reward V2：

```text
task_reward = sum(2 ** ((merge.new_level - 2) / 2))
Phi(s) = 0.6 * C(s) + 0.3 * R(s) + 0.1 * K(s)
potential_shaping_reward =
    lambda_phi * (gamma * Phi(effective_next) - Phi(previous))
reward = task_reward + potential_shaping_reward + terminal_penalty
```

当前默认配置：

```python
RewardConfig(
    gamma=0.99,
    lambda_phi=0.5,
    capacity_weight=0.6,
    recoverability_weight=0.3,
    chain_readiness_weight=0.1,
    terminal_penalty=0.0,
)
```

各项含义：

- `task_reward` 只读取本次真实 `MergeEvent`，不直接复用原游戏分数。
- `C(s)` 是 q0～q3 的顶部可达投放容量，`R(s)` 是水果可恢复性，
  `K(s)` 是连锁就绪度；三项均来自 `StateAnalysis` 且位于 `[0, 1]`。
- `gamma` 必须与 DQN target 的折扣因子相同，训练入口从同一个 `--gamma`
  参数构造两份配置。
- `terminal_penalty` 默认是 0，仅保留为短跑发现信号不足后的显式开关。
- 存活时间、最高水果高度、绝对危险高度和固定 `-100` 终局惩罚已经删除。

真实 terminal 的有效 next potential 强制为 0，而且
`info["next_state_analysis"]` 为 `None`。环境同时生成只供归因使用的
`info["post_action_state_analysis"]`，使 tracker 能检查终局动作后的解封、新封路和
终局支撑；它不进入 reward、DQN bootstrap 或主 replay。truncated 不是 MDP 终态：
它会生成同 episode 的 `analysis[t+1]`，保留 next potential 和 bootstrap；
该不稳定分析标记为 degraded、不能产生高置信归因，并在 reset 时丢弃缓存。

`DaxiguaEnv` 为每个 worker 持有一个 `StateAnalyzer`。初始 step 分析当前和下一
边界；之后正常连续 step 复用上一轮的 next analysis，因此通常每次投放只新增一次
分析。分析始终使用规范 15 动作列，即使某些小型测试用更少策略动作。

`DaxiguaEnv.step()` 的 `info` 包含：

- `reward_breakdown`: `RewardBreakdown`；
- `previous_state_analysis` / `next_state_analysis`: worker-local 前后分析；
- `post_action_state_analysis`: 动作后的分析；非终局时与 `next_state_analysis`
  是同一对象，真实终局时仅供 AttributionTracker 使用；
- `state_analysis_calls` / `state_analysis_seconds`: 本 step 真正新增的分析次数和耗时；
- `state_analysis_cache_hit`: 是否复用前一轮 next analysis；
- `state_analysis_degraded_count`: 本 step 新分析中的降级数量。

完整 `StateAnalysis` 不进入 `TensorTransition` 或主 `ReplayBuffer`，并行 worker
也只向主进程返回 scalar reward 和聚合统计。

训练入口会把采集窗口内的 reward breakdown 均值写入 `metrics.csv`：

- `collect_mean_reward_total`
- `collect_mean_task_reward`
- `collect_mean_potential_shaping_reward`
- `collect_mean_terminal_penalty`
- `collect_mean_previous_potential` / `collect_mean_next_potential`
- `collect_mean_potential_delta`
- 前后 `top_connected_capacity`、`recoverability`、`chain_readiness`
- `collect_mean_merge_event_count`
- `collect_p95_abs_potential_shaping_reward`

同一 CSV 还记录 StateAnalyzer 调用次数、总/平均耗时、缓存命中率和降级率，以及
AttributionTracker 调用/耗时、事件 created/confirmed/cancelled/interrupted、
pending 数量、谱系合成数、同一步最大连锁深度、平均/p95 延迟和事件状态 JSON。
`plots/reward_breakdown_curves.png` 会分别绘制 task/shaping、potential 分量和
next-state C/R/K，供短跑校准 shaping 是否压过真实合成效用。

### `AttributionTracker`

`daxigua_rl.attribution.AttributionTracker` 在每个 rollout worker 内独立维护：

- drop 与 ordered `MergeEvent` 的完整水果谱系和根材料权重；
- 每个真实合成唯一的 `MergeValueKey`，只有对应 `MERGE_LINEAGE` 可持有非零任务
  价值，其它触发/铺垫事件共享预算且 utility 为 0；
- `BORN_BURIED` / `REACHABILITY_SEALED` pending、12 个稳定边界确认、恢复或进入
  合成谱系时撤销；
- 渐进通道损失责任、后续真实合成兑现的邻接/阶梯/墙锚/支撑/伙伴/营救事件；
- 终局后状态归因、truncated/reset/shutdown 中断收口和轻量统计。

事件键包含 `(worker_id, episode_id, event_index)`，贡献动作使用完整
`TransitionKey`。`AttributionEvent` 还保存 `attribution_version` 和 tracker 配置
指纹。tracker 对象、事件、谱系和 pending 状态均可 pickle，兼容 Windows spawn。
归因动作必须和规范 15 列分析一一对应；正式训练固定 `action_count=15`，tracker 会
拒绝其它动作数或与分析列不一致的 action index/drop x，避免把责任写到错误 Q 下标。

只有显式接触路径才能产生 `MECHANICAL_TRIGGER`、接触型 motif 破坏等机械归因。
当前 `HeadlessGame` 尚未生成逐帧 `ContactInfluenceEdge`，因此这些类型不会从静态
几何推测；`CORRIDOR_OPENED_USED` 也在缺少真实路径使用证据时保持禁用。

`FruitLineageRecord.chain_depth` 表示跨投放的谱系深度；训练日志中的
`attribution_max_chain_depth` 则由当前一次物理稳定过程内的 merge DAG 单独计算，
二者不能混用。

## 状态数据

当前 `GameState` 包含：

- `board_fruits`: 场地中真实水果快照。
- `fruit_queue`: q0 到 q3 的待投放水果序列。
- `score`: 当前分数。
- `step_count`: 已投放次数。
- `physics_frame`: 无渲染物理累计帧。
- `done`: 是否结束。
- `geometry`: 场地宽高、生成线、墙体宽度、地板位置。
- `max_height`、`fruit_count`、`max_level`、`empty_space_ratio`: 全局摘要状态。

`FruitState` 同时保留：

- `radius`: 显示半径。
- `physics_radius`: 当前 Pymunk shape 的真实碰撞半径。

同等级水果的真实半径可能因“直接投放”或“合成生成”而不同，不能只根据等级反推。
到边界/危险线的距离、`max_height`、`empty_space_ratio` 和图中的几何关系均使用
`physics_radius`。

`distance_to_danger_line` 表示水果外缘到生成线的几何距离；当前游戏失败检测使用
水果圆心是否持续越线，因此该字段不能直接当作精确的终止倒计时。

`ActionCandidate.current_radius` 仍是显示半径，用于保持合法投放横坐标范围不变；
`current_physics_radius` 是新投放水果将使用的碰撞半径。图的 `radius` 特征和
碰撞/投放路径关系改用真实物理半径，但节点 28 维、边 26 维的结构不变。
旧 checkpoint 的网络结构仍可加载，但 `radius` 输入语义会有轻微分布变化；第一次
完整归因训练应从新配置重新训练。

## 状态归因数据契约与分析器

当前 `daxigua_rl.attribution` 包已经提供只读 schema 和静态 `StateAnalyzer`：

- `StateAnalysis`: 一个动作前稳定边界的完整分析快照。
- `FruitAnalysis`: 单水果可达性、伙伴、支撑、埋藏和等级倒置摘要。
- `QueueLaneAnalysis`: q0 到 q3 各自在 15 个动作列上的投放容量。
- `FreeSpaceRegionAnalysis`: 规范最小水果探针下的顶部连通空间或封闭空腔。
- `SupportEdge`: 方向固定为 supporter -> supported fruit 的稳定约束。
- `ContactInfluenceEdge`: 产生当前边界的前一动作所留下的压缩接触证据。
- `PartnerComponent`、`ChainMotif`: 同级伙伴分量和局部连锁结构。
- `StateAnalysisDiagnostics`: 稳定性、归因有效性、降级码和分析耗时。
- `StateAnalyzerConfig`: 分析容差、栅格精度和结构阈值配置。
- `StateAnalyzer`: 从 `GameState`、15 个动作候选和 `TransitionKey` 生成完整快照。

导入方式：

```python
from daxigua_rl.attribution import StateAnalyzer, StateAnalyzerConfig

analyzer = StateAnalyzer(StateAnalyzerConfig())
analysis = analyzer.analyze(
    state,
    action_candidates,
    transition_key,
)
```

当前契约固定以下语义：

- `StateAnalysis.transition_key.step_index=t` 表示动作 `t` 执行前的边界；跨步变化比较
  `analysis[t] -> analysis[t+1]`。
- 如果状态携带前一动作的接触证据，`incoming_transition_key` 必须指向同 episode
  的 `t-1`；初始边界没有 incoming key。
- 动作 mask 固定为 15 位，并按 `action_offset` 编位。
  `action_indices[offset]` 单独保存环境动作号，避免和 Q 值下标混淆。
- q0 到 q3 各自保存 15 个 `drop_x_by_action`。水果半径不同会改变合法投放区间，
  不能让后续队列槽复用 q0 的横坐标。
- `FruitAnalysis.physics_radius` 是当前 shape 半径；
  `probe_physics_radius` 是未来直接投放同级水果时用于路径膨胀的半径。
- `probe_physics_radius` 和每个队列槽的 `physics_radius` 必须符合游戏本体当前等级的
  直接投放半径规则；场上 `physics_radius` 仍保留真实 shape 值，不能按等级重算。
- 所有数据类均为 frozen、slots、深 tuple 数据；mask/count、15 项数组、比例范围、
  当前水果引用和支撑缓存会在构造时校验。
- 场上水果等级限制为 1 到 11，队列槽等级限制为可直接投放的 1 到 4；每槽
  `capacity` 和 q0-q3 聚合后的 `top_connected_capacity` 必须与 Reward V2 公式一致。
- q0～q3 容量和单水果可达 mask 使用解析圆形竖直投放列：以直接投放探针半径膨胀
  当前真实圆形障碍，第一接触点决定落点、可达性和 blocker。该近似不模拟沿圆滚动。
- 自由空间统一用等级 1 的直接投放半径作规范探针，对圆心可进入区域执行四邻域栅格
  BFS。区域对象保存顶部连通性、面积、质心、包围盒、边界水果和墙/地板接触。
- `FreeSpaceRegionAnalysis.region_id` 只保证当前分析内唯一。跨步区域关联应组合面积、
  质心、包围盒重叠和边界水果，不能把 ID 当作永久身份。
- 支撑分析区分地板支撑、墙约束、下方水果支撑、盖压和桥接；同级伙伴分量及
  `merge_pair` / `level_ladder` motif 用于 `recoverability` 和 `chain_readiness`。
  `critical_blocker_ids` / `caps` 只在目标零可达时形成，正 motif 同时要求队列兼容
  等级和实际触发动作，避免给局部受阻或完全封死结构错误加分。
- 不稳定的 truncated 边界可以保留诊断分析，但必须设置
  `valid_for_attribution=False`，不能生成高置信归因事件。

`StateAnalyzer` 是纯只读、无物理推进的 worker-local 组件。调用方可以把前一动作已经
压缩好的 `ContactInfluenceEdge` 和对应 `incoming_transition_key` 一并交给分析器，
但当前分析器不会自行采集逐帧碰撞日志。`DaxiguaEnv` 已把相邻分析接入 Reward V2，
collector 负责提供稳定轨迹键并汇总分析性能；分析对象仍不写入
`TensorTransition` / 主 `ReplayBuffer`。`AttributionTracker` 已实现并由
collector 接入；confirmed 事件会和 worker 暂存的原 transition 图上下文关联，
经 `RuleCausalSampleBuilder` 写入独立 `CausalReplayBuffer`。需要动态仲裁的少量
事件则由同一 worker 的快照环生成有界反事实 proposal。物理引擎仍不保存逐帧完整
碰撞日志，接触依赖型事件只在已有显式机制证据时产生。

## 图构建接口

当前 `daxigua_rl.graph` 包提供：

- `GraphBuilder`: 将 `GameState` 和 `ActionCandidate` 转换成框架无关的 `GraphData`。
- `GraphAblator`: 在不改变图维度的前提下按配置置零部分节点或边特征，用于消融研究。

当前优化后的图特征维度：

```text
node_feature_dim = 28
edge_feature_dim = 26
```

详细节点和边特征以 `docs/rl/gnn_daxigua_design_reference.md` 为准。

## 模型前向接口

当前 `daxigua_rl.models` 包提供：

- `GNNQNetwork`: 统一图 message passing Q 网络。
- `MessagePassingLayer`: 基于 mean aggregation 的单层消息传递。

当前 `daxigua_rl.graph.tensor` 提供：

- `graph_to_tensor(graph) -> GraphTensor`
- `collate_graph_tensors(graphs) -> GraphBatch`
- `GraphTensor.to(device=None, dtype=None)`
- `GraphBatch.to(device=None, dtype=None)`

最小前向链路：

```text
DaxiguaEnv.reset()
    -> GameState + ActionCandidate
    -> GraphBuilder.build(...)
    -> graph_to_tensor(...)
    -> GNNQNetwork(...)
    -> q_values[action_count]
```

模型支持单图和批量图两种输入：

- `GraphData` / `GraphTensor` 输入时，输出 shape 为 `[action_count]`。
- `GraphBatch` 输入时，输出 shape 为 `[total_action_count]`，每张原始图对应的动作区间由 `GraphBatch.action_slices` 记录。

Q 值在训练前没有策略意义。

## 训练经验结构

当前 `daxigua_rl.training` 包提供：

- `TensorTransition`: 张量化 DQN 经验记录，正式训练主链路使用它。
- `ReplayBuffer`: 固定容量内存或热内存/冷磁盘 TD 回放池。
- `NStepTransitionAccumulator`: worker-local n-step return 聚合器。
- `RolloutCollector` / `ParallelRolloutCollector`: 单/多进程 rollout 采集器。
- `TransitionKey`: 一次训练 run 内的稳定轨迹身份。
- `DQNTrainer`: Double DQN + n-step + 因果联合更新器。

字段含义：

- `graph`: 当前状态图，也就是状态 `s`。
- `action_offset`: 被选择动作在 `q_values` 中的下标，也就是训练 loss 读取 `q_values[action_offset]` 的位置。
- `reward`: 从当前动作开始累计的 1～3 步折扣 reward。
- `bootstrap_steps`: 该 reward 实际覆盖的连续环境步数。常规为 3，
  episode 尾部可为 1 或 2。
- `next_graph`: 下一状态图，也就是状态 `s'`；只有真实 terminal transition 可以为
  `None`，truncated transition 必须保存可信 final observation。
- `terminated`: 游戏规则导致的结束。
- `truncated`: 环境流程导致的截断，例如物理推进达到上限仍未稳定。

派生属性：

- `action_index`: 从当前图里读取的环境动作编号，主要用于日志和动作映射检查。
- `action_node_index`: 被选择 action 节点在 `graph.node_features` 中的行号。
- `done`: `terminated or truncated`。
- `can_bootstrap`: 是否可以使用 `next_graph` 计算下一状态 Q 值。

当前约定：

```text
q_value = q_values[transition.action_offset]
next_action = argmax Q_online(next_graph)
target = reward + gamma**bootstrap_steps * Q_target(next_graph, next_action)
                                        # 正常或 truncated transition
target = reward                        # 仅 terminated transition
```

`done = terminated or truncated` 只表示采集 episode 边界；bootstrap mask 只由
`terminated` 决定。主 `TensorTransition` 和冷热 `ReplayBuffer` 不追加归因历史字段，
因果样本走独立 `CausalReplayBuffer`。

### `ReplayBuffer`

接口：

- `ReplayBuffer(capacity=100_000, seed=None)`: 创建固定容量回放池。
- `push(transition)`: 写入一条经验对象。
- `extend(transitions)`: 批量写入。
- `sample(batch_size) -> tuple[...]`: 随机无放回采样。
- `is_ready(batch_size) -> bool`: 判断是否足够采样一个 batch。
- `clear()`: 清空。
- `len(buffer)`: 当前已保存经验数量。
- `checkpoint_manifest()` / `validate_checkpoint_manifest(manifest)`：
  保存并校验存储语义。
- `checkpoint_state_dict()` / `load_checkpoint_state_dict(state)`：
  保存和恢复有界训练状态。

当前约定：

- 默认容量是 `100_000`，也就是十万条经验。
- 容量满后覆盖最旧经验。
- buffer 只接受并保存 CPU `TensorTransition`；其它经验类型会在 `push()` 时拒绝。
- 正式训练主链路由 collector 的 n-step 累加器写入。
- 纯内存模式使用均匀随机采样并在 checkpoint 中精确保存整个环。
- hybrid 模式按配置混合热/冷采样；checkpoint 只保存最近
  `hot_capacity` 条、采样 RNG 和来源计数，`resume_policy="hot_only"` 且
  `omitted_cold_count` 明确记录未恢复的冷经验。这是有意的 hot-resume，
  避免每次 checkpoint 复制完整冷 replay。

### `NStepTransitionAccumulator`

正式训练每个 rollout worker 独立维护：

```python
NStepTransitionAccumulator(n_step=3, gamma=0.99)
```

收到第 3 个连续 transition 时，它输出从最早状态开始的折扣累计 reward，并令
`bootstrap_steps=3`。遇到 terminated 或 truncated 时会把剩余前缀全部输出，尾部
分别记录真实的 1 或 2 步 horizon。truncated 仍保留最终 `next_graph` 和 bootstrap；
terminated 尾部不 bootstrap。不同 worker、episode 或 reset 之间不得串接窗口。

### `RolloutCollector`

导入方式：

```python
from daxigua_rl.training import RolloutCollector
```

主要接口：

- `RolloutCollector(..., causal_replay_buffer=None, n_step=1, gamma=0.99, policy_version=None, counterfactual_enabled=False, counterfactual_ring_size=32, ...)`：
  创建单环境采集器；正式训练显式传 `n_step=3`、因果 replay 和反事实开关。
- `reset(seed=None, fruit_queue=None)`: 显式重置环境并开始新 episode。
- `collect_steps(step_count, epsilon=1.0) -> RolloutStats`: 收集指定数量的 transition 并写入 replay buffer。
- `drain_counterfactual_proposals() -> tuple[CounterfactualProposal, ...]`：
  非阻塞排空轻量 proposal outbox。

当前采集流程：

```text
当前 GameState + action_candidates
-> GraphBuilder.build(...)
-> graph_to_tensor(...)
-> epsilon-greedy 选择 action_offset
-> 生成 TransitionKey
-> DaxiguaEnv.step(action_offset, transition_key=...)
-> StateAnalyzer 前后边界 + Reward V2
-> AttributionTracker.observe_transition(...)
-> confirmed 事件 -> RuleCausalSampleBuilder -> CausalReplayBuffer
-> 可选 EngineSnapshot / factual outcome -> CounterfactualProposalBuilder
-> 构建 next_graph
-> NStepTransitionAccumulator
-> 1～3 步 TensorTransition(...)
-> ReplayBuffer.push(...)
```

当前约定：

- `epsilon=1.0` 时可以不提供模型，collector 会完全随机探索。
- `epsilon<1.0` 时必须提供 Q 网络模型，用于 greedy 分支。
- 采集时模型会临时切到 `eval()`，结束后恢复原本训练模式。
- episode 结束后 collector 会自动 `reset()` 并继续采集，直到达到指定 transition 数。
- n-step 累加器按 worker/episode 隔离；episode 边界会 flush 尾部，不能把下一局 reward
  聚合到上一局。
- 反事实关闭时不会调用 `capture_snapshot()`。启用后每 worker 只保存最近配置数量的
  稳定边界；完整快照不和主 TD transition 一起发送。
- `RolloutCollector` 依赖 PyTorch，因此不会被 `daxigua_rl` 顶层自动导入。

collector 会在动作执行前生成：

```text
TransitionKey(worker_id, episode_id, step_index)
```

- 单进程 `worker_id` 默认是 `0`，并行 worker 使用固定的 `worker_index`。
- `episode_id` 在每个 worker 内随成功 reset 单调增加。
- `step_index` 从 `0` 开始，并跨多次 `collect_steps()` 调用保持连续。
- 键在一次训练 run 内唯一；跨 run 时由 run 目录或未来的 `run_id` 提供外层命名空间。
- collector 把键传给环境，使 `analysis[t] -> analysis[t+1]`、scalar reward 和
  transition 使用同一身份；键也通过 `RolloutStats.transition_keys` 按采集顺序
  返回，但不写入主 ReplayBuffer。

`RolloutStats` 提供：

- `steps`: 本次写入多少条 transition。
- `transition_keys`: 与本次 transition 一一对齐的稳定轨迹键。
- `episodes`: 本次完成多少局。
- `total_reward`: 本次总 reward。
- `episode_rewards`、`episode_lengths`、`episode_scores`: 本次已结束 episode 的统计。
- `random_actions`、`greedy_actions`: 探索/利用动作数量。
- `buffer_size`: 采集后 replay buffer 大小。
- `reward_breakdown_totals`: Reward V2 各项累计值。
- `potential_shaping_abs_values`: 用于日志窗口计算 shaping 绝对值 p95 的轻量标量。
- `state_analysis_calls` / `state_analysis_seconds`: 真正执行的状态分析次数和总耗时。
- `state_analysis_cache_hits`: 复用上一轮 next analysis 的次数。
- `state_analysis_degraded_count`: truncated 等不稳定边界产生的降级分析数。
- `attribution_tracker_calls` / `attribution_tracker_seconds`: tracker 调用次数和耗时。
- `attribution_events_created/confirmed/cancelled/interrupted`: 事件生命周期计数。
- `attribution_pending_event_count`: 当前 worker（并行时为所有 worker 最新快照之和）
  尚未解决的事件数量。
- `attribution_lineage_merge_count` / `attribution_chain_merge_count` /
  `attribution_max_chain_depth`: 谱系合成和同一物理过程连锁统计。
- `attribution_event_status_counts` / `attribution_delays`: 按类型/状态计数和归因延迟。
- 规则因果 builder 的输入/输出/跳过原因、因果样本数和因果 replay 大小。
- 快照调用/耗时/失败、历史环大小/淘汰、proposal 事件数/生成数/跳过原因/
  序列化字节和 outbox 大小。

完整事件与谱系不会写入 `TensorTransition`。collector 关闭时会先让每个 worker
finalize pending；真实解封/进入谱系与 truncated/reset/shutdown 中断通过
`resolution_reason` 区分。

### `CausalReplayBuffer`

因果监督不修改主 `TensorTransition`，而是使用独立的纯内存有界池：

```text
CausalSample
├── graph + actual/comparison action offset
├── direction + target margin/return difference
├── confidence + cause type + delay
├── transition/event/budget identity
├── attribution/policy/config provenance
└── supervision_kind
```

`RuleCausalSampleBuilder` 只消费 confirmed 事件，并从
`CausalTransitionContext` 找回原始状态图和规范 15 动作。正铺垫选择结构更差的合法
比较动作，负封路选择保留更多容量的安全动作；没有可信比较动作时只记录跳过原因。
同一个 merge budget 不复制 Reward V2 任务价值。

`CausalReplayBuffer(capacity=20_000)` 按正铺垫、负封路和经验反事实类别分层采样，
对相同监督身份去重，并可用 `checkpoint_state_dict()` /
`load_checkpoint_state_dict()` 精确保存样本、类别、游标、计数和 RNG。

### 预算反事实

采集侧的 `CounterfactualProposalBuilder` 维护 worker-local 的 32 边界环，并把延迟
confirmed 事件回连到原 `EngineSnapshot`、factual outcome、图上下文和最多配置数量的
替代动作。高价值合成、两段以上连锁、规则正负冲突及中等 placement confidence 是
主要候选；确定性规则事件不需要普遍重演。

主进程 `CounterfactualCoordinator`：

1. 只在 target network 同步时调用 `refresh_target_policy(...)` 冻结模型与
   Reward/图/物理配置；
2. 用 `record_real_steps()` 登记真实采样成本；
3. `submit()` / `submit_many()` 经 `BudgetedCounterfactualScheduler` 非阻塞接纳；
4. 独立 CPU worker 恢复快照、先重演 factual branch，再运行有限 alternatives；
5. `poll()` 只把 `completed` 且 reproduction 通过的结果转成因果样本。

默认 soft cost ratio 为配置值，所有普通反事实与 Shapley 通过同一 external-token
接口共享 `counterfactual_hard_limit=0.10`。真实步门槛、队列容量、每 worker
in-flight、CPU 比例和连续失败熔断都在调度侧执行；拒绝任务只增加 drop reason，
不阻塞 rollout，也不产生伪标签。

### 局部 Shapley

`LocalShapleyCoordinator` 只选择高价值、至少两个且最多配置数量候选的
`shapley_ready` proposal，累计比例上限默认为 `0.0005`。它使用一个独立进程：

- 按 factual 轨迹逐步验证 grand coalition；
- 对 2～4 个局部历史动作执行 subset replay 与缓存；
- 使用少量配对正反排列估计边际贡献；
- 检查 Shapley 和与 grand-empty 差值之间的效率残差；
- 通过后生成 `cause_type=LOCAL_SHAPLEY`、
  `supervision_kind=shapley` 的因果样本。

Shapley 先从共享硬预算预留 token；被选中后跳过同一 proposal 的普通反事实，避免重复
计算。比例、预算、候选、重演或效率门禁失败都不会生成标签。

### `DQNTrainer`

导入方式：

```python
from daxigua_rl.training import DQNTrainer, DQNTrainerConfig
```

接口：

- `DQNTrainer(online_model, target_model, replay_buffer, optimizer, config=None, loss_fn=None, causal_replay_buffer=None)`
- `train_step() -> DQNTrainStats`
- `is_ready() -> bool`
- `sync_target_model()`
- `restore_update_step(update_step)`：resume 时恢复 target 同步周期相位。

默认配置：

```python
DQNTrainerConfig(
    gamma=0.99,
    n_step=3,
    batch_size=32,
    target_update_interval=1000,
    grad_clip_norm=10.0,
    sync_target_on_init=True,
    causal_batch_size=32,
    causal_update_interval=2,
    lambda_rule=0.15,
    lambda_counterfactual=0.10,
    counterfactual_return_scale=merge_utility(7),
    counterfactual_target_clip=5.0,
)
```

当前 Double DQN + n-step target：

```text
current_batch = collate_graph_tensors(batch.graph)
current_q_flat = online_model(current_batch)
current_q = current_q_flat[action_slice.start + action_offset]

if transition.can_bootstrap:
    next_batch = collate_graph_tensors(bootstrap_next_graphs)
    selected_next_action = argmax(online_model(next_batch))
    bootstrap_q = target_model(next_batch)[selected_next_action]
    target = (
        transition.reward
        + gamma**transition.bootstrap_steps * bootstrap_q
    )
else:
    target = transition.reward
```

当前约定：

- TD 默认使用 `SmoothL1Loss`。
- 初始化时会把 `online_model` 参数同步到 `target_model`。
- `target_model` 参数会被冻结，只用于无梯度推理。
- 每隔 `target_update_interval` 次 `train_step()` 同步一次 target network。
- 当前使用 GraphBatch，把 batch 内多张图拼成不连通大图执行批量 forward。
- 每 `causal_update_interval` 次 TD update 最多读取 `causal_batch_size` 个因果样本；
  样本不足不会阻塞 TD 更新。
- 规则样本使用带 confidence 的 hinge ranking loss；counterfactual / Shapley 使用
  归一化、裁剪 target delta 的 Huber loss。总 loss 为
  `td + lambda_rule*rule + lambda_counterfactual*empirical`。
- ReplayBuffer 保存 CPU `TensorTransition`，训练时再把 GraphBatch 搬到模型设备。
- 默认使用梯度裁剪 `grad_clip_norm=10.0`。
- Q、target、三路 loss 与梯度范数必须全部有限；检查失败发生在
  `optimizer.step()` 前，避免把 NaN/Inf 写入模型和 optimizer 状态。

`DQNTrainStats` 提供：

- `update_step`: 已完成更新次数。
- `loss` / `td_loss`: 总 loss 与 TD 子损失。
- `mean_q`: 当前 Q 平均值。
- `mean_target`: target 平均值。
- `mean_reward`: reward 平均值。
- `mean_abs_td_error`: 平均绝对 TD 误差。
- `bootstrap_count`: batch 中使用 next_graph bootstrap 的 transition 数量。
- `grad_norm`: 裁剪前梯度范数。
- `target_synced`: 本次是否同步 target network。
- `causal_update_applied`、因果/规则/反事实/Shapley batch 大小。
- `rule_rank_loss` / `weighted_rule_rank_loss`。
- `counterfactual_loss` / `weighted_counterfactual_loss`。
- 规则 pair 正确率/margin 满足率、经验差值符号正确率/平均绝对误差。
- 因果 replay 采样、collate 和 forward 的额外耗时。

## 训练入口

当前正式训练脚本：

```text
src/daxigua_rl/scripts/train_dqn.py
```

第一次长训前先运行不启动训练的 preflight：

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u \
  tools/preflight_training.py --config configs/train_dqn_causal_500k.toml
```

门禁验证解析后的正式配置、Python/Pymunk/Chipmunk、CUDA 模型前后向、完整因果
optimizer step、局部 Shapley 物理链路、多次 `EngineSnapshot` 重演、输出盘和
反事实 CPU 余量。任一 required check 失败时返回非零并写 JSON；它不创建训练 run。

三套配置共享 `configs/train_dqn_fast30_parallel.toml` 的完整算法/物理基线：

| 配置 | 用途 |
| --- | --- |
| `train_dqn_causal_smoke_5k.toml` | 5000 update 集成烟测 |
| `train_dqn_causal_calibration_10k.toml` | 10000 update 标定；样本不足时从 update 0 另起独立 25000 校准 |
| `train_dqn_causal_500k.toml` | 第一次 500000 update 正式训练 |

配置文件存在不代表相应运行已经通过；结果必须以实际 run 产物为准。

训练示例：

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u \
  -m daxigua_rl.scripts.train_dqn \
  --config configs/train_dqn_causal_smoke_5k.toml
```

普通 `conda run` 会捕获子进程输出；使用 `--no-capture-output` 才能实时看到心跳。

默认训练流程：

```text
warmup 随机收集经验
-> 每轮 collect_per_update 条新经验
-> worker 输出 3-step TD transition、规则因果样本和反事实 proposal
-> Shapley 极稀疏选择；其余 proposal 进入预算反事实调度
-> poll 物理结果并写 CausalReplayBuffer
-> DQNTrainer 联合 TD / rule / counterfactual / Shapley 更新
-> target sync 时刷新 rollout 模型和冻结反事实 target payload
-> epsilon 按 schedule 衰减
-> 每 3 秒打印轻量进度
-> 终端日志
-> metrics.csv
-> 原子版本化 checkpoint
-> greedy 评估
-> matplotlib 训练曲线图
```

默认输出目录：

```text
runs/dqn_YYYYMMDD_HHMMSS/
├── config.json
├── metrics.csv
├── episode_metrics.csv
├── attribution_warmup.json
├── attribution_shutdown.json
├── counterfactual_shutdown.json
├── failure_latest.json                 # 仅异常时
├── resume_<时间戳>.json                # 仅恢复时
├── resume_config_<时间戳>.json         # 仅恢复时
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   ├── failure_last_normal.pt          # 非有限训练故障时
│   └── step_XXXXXXXX.pt
└── plots/
    └── training_curves.png
```

`metrics.csv` 是核心可视化数据源，记录：

- update step、环境步数、epsilon、buffer 大小。
- total/TD/rule/counterfactual loss、各类因果 batch、pair/sign 正确率、mean Q、
  mean target、TD error 和 grad norm。
- 采集阶段 episode 统计，以及 Reward V2 的 task、potential shaping、C/R/K 和
  merge event 数量。
- shaping 绝对值 p95，以及 StateAnalyzer 调用次数、耗时、缓存命中率和降级率。
- AttributionTracker 调用耗时、事件生命周期、pending、谱系/连锁、延迟和
  event/status JSON。
- 规则因果样本 builder、`CausalReplayBuffer` 分层/监督类型/存储状态。
- 快照/proposal 数、重演 strict/numeric/semantic 三态、五类连续误差最大值、模拟
  步、soft/hard token 比例、队列、熔断和 drop reason。
- 局部 Shapley 的考虑/选择/完成/失败、三态复现、subset、效率门禁、共享 token 和
  样本数。
- greedy 评估均分、最高分、最低分、历史最高分、平均 reward、平均 episode 长度。
- 采样和训练速度。

`attribution_warmup.json` 单独汇总随机 warmup 的归因事件和期末 pending，避免将
warmup 混进第一行训练曲线；`attribution_shutdown.json` 记录各 worker 在退出时因
`worker_shutdown` 或此前 `manual_reset` 收口的 pending。即使 replay flush 失败，
训练清理仍会尝试 finalize tracker、先关闭 Shapley、再关闭普通反事实、写
`counterfactual_shutdown.json` 和归因 sidecar，并关闭日志。

### 版本化 checkpoint 与 hot-resume

checkpoint 使用同目录临时文件、`fsync` 和 `os.replace` 原子写入，schema 中保存：

- online/target 模型、optimizer 和 trainer update step；
- env step、epsilon、最佳评估、最新指标等训练计数；
- Python random、PyTorch CPU 和全部 CUDA RNG；
- `RunManifest`、规范配置指纹、运行时/依赖/Git 元数据；
- TD replay 和 `CausalReplayBuffer` 的 manifest/state；
- 反事实与 Shapley 的有界协调状态摘要。

恢复示例：

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u \
  -m daxigua_rl.scripts.train_dqn \
  --config configs/train_dqn_causal_500k.toml \
  --resume runs/<run>/checkpoints/latest.pt
```

`--resume` 默认从 checkpoint 所属 run 继续，不能与
`--overwrite-run-dir` 组合。加载前按配置指纹拒绝模型、Reward、物理、replay、
因果预算等训练语义漂移；允许变化的字段只限总更新数、输出路径、日志/保存/评估/绘图
频率和 resume 控制字段。

纯内存 TD replay 和 `CausalReplayBuffer` 精确恢复。正式 hybrid TD replay 只恢复
checkpoint 中保存的热层，因此称为 hot-resume；`resume_<时间戳>.json` 明确记录
source count、omitted cold count、episode ID 续接和 RNG 恢复契约。若热层不足 batch，
入口会执行有记录的 `resume_warmup`。`metrics.csv` / `episode_metrics.csv` 先裁切到
checkpoint update，再追加，避免重复行或保留 checkpoint 之后的不一致日志。

`smooth` epsilon 的分母由首次 checkpoint 中的
`epsilon_schedule_total_updates` 冻结。恢复时即使增加 `total_updates`，探索率也不会
重新放大；`resume_<时间戳>.json` 会记录请求总步数以及
`epsilon_schedule_extended_without_reexpansion`，便于区分普通恢复、连续延长和
从 update 0 新建的独立校准 run。

任何训练异常都会写 `failure_<时间戳>.json` 和 `failure_latest.json`，记录阶段、
update、异常和最近统计。若在 optimizer 前触发非有限值 fail-fast，还会原子保存
`failure_last_normal.pt`，其更新计数保持在最后一次成功参数更新。
成功恢复并完成最终 checkpoint/曲线后只会删除活动指针 `failure_latest.json`，
带时间戳的历史诊断继续保留。

`episode_metrics.csv` 按 episode 结束事件逐行记录训练过程中每局完整游戏的分数：

- `episode_index`: 当前 run 中第几个结束的 episode。
- `phase`: `warmup` 或 `train`。
- `update_step`、`env_steps`、`epsilon`: 该局结束时的训练位置。
- `score`、`episode_reward`、`episode_length`: 单局得分、累计 reward 和投放次数。
- `terminated`、`truncated`: 该局结束原因。

每次评估刷新历史最高单局分数时，会额外保存：

```text
checkpoints/best.pt
```

训练入口默认每 `3` 秒打印一次轻量进度心跳：

```text
[progress] | phase=train | 1200/10000 | 12.0% | env_steps=2200 | buffer=2200 | eps=0.958 | speed=40.12 env_steps/s | loss=0.1234
```

可以通过参数调整或关闭：

```bash
--progress-interval 3
--progress-interval 0
```

`training_curves.png` 会从 `metrics.csv` 当前内存记录生成，包含：

- loss 曲线。
- 训练过程中每局完整游戏 score、采集均分、评估均分、评估最高分和历史最好分。
- epsilon 衰减。
- TD error。
- grad norm。
- mean Q / mean target。

默认 epsilon schedule 是 `smooth`，按训练进度百分比平滑下降。默认
`epsilon_start=1.0`、`epsilon_end=0.05` 时，曲线大致满足：

```text
0%   -> 1.00
30%  -> 0.50
50%  -> 0.20
70%  -> 0.07
80%+ -> 0.05
```

如需恢复旧的按环境步数线性下降方式，可以使用：

```bash
--epsilon-schedule linear
```

## 模型观看入口

当前真实游戏窗口观看脚本：

```text
src/daxigua_rl/scripts/watch_dqn.py
```

运行方式：

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u -m daxigua_rl.scripts.watch_dqn \
  --checkpoint runs/<run>/checkpoints/latest.pt
```

常用参数：

- `--checkpoint`: 必填，训练脚本保存的 checkpoint 路径。
- `--action-count`: 候选投放动作数量；默认读取 checkpoint 中的训练参数。
- `--decision-delay-ms`: 模型选定落点后等待多久再投放，默认 `240` 毫秒，方便肉眼看清当前水果移动到哪里。
- `--print-actions`: 每次投放时打印 action、drop_x 和 Q 值摘要，便于对照画面调试。

当前观看流程：

```text
加载 checkpoint
-> 重建 GNNQNetwork
-> 打开原 pygame Board
-> playable_adapter 把实时 Board 转成 GameState + ActionCandidate
-> GraphBuilder.build(...)
-> GNNQNetwork 输出候选动作 Q 值
-> 选择 argmax 动作
-> 通过原 Board 的投放逻辑落子
```

当前约定：

- 观看入口复用原游戏画面，适合检查模型最终在真实窗口中的操作效果。
- `playable_adapter.py` 和 `watch_dqn.py` 属于 RL 侧代码；游戏本体不 import 它们。
- 观看入口是可视化检查工具，不替代无渲染训练、评估和数据采集。
- 观看脚本会打开 pygame 窗口并持续运行，退出方式沿用原游戏窗口关闭逻辑。

## 后续扩展

- 当前首轮大规模训练接口已经闭合，下一步是按 5k 烟测、10k（必要时独立 25k）标定、
  preflight、500k 正式训练的顺序产生运行证据，而不是继续扩张框架范围。
- 若标定要求修改阈值或 loss 权重，必须写入 TOML、更新归因版本并保留旧配置，
  不得在正式长训中静默改变语义。
- 后续训练脚本仍放在 `daxigua_rl.scripts`；游戏本体不得反向依赖训练、因果 replay
  或 checkpoint 设施。
- 如果未来需要进一步性能优化，先分开 profile headless 物理、StateAnalyzer、
  snapshot capture、反事实模拟和 GPU update，再决定具体降级项。
