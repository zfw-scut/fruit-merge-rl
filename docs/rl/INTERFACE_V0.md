# RL 接口 v0

## 目标

本接口用于先跑通强化学习训练闭环：

```text
reset -> observe -> choose action -> step -> reward / next_state / done
```

当前提供无渲染游戏接口、RL 环境壳层、GNN 图构建基础设施、最小 GNN-Q 前向模型、`Transition` 经验记录、基础 `ReplayBuffer`、单进程 `RolloutCollector` 和最小标准 `DQNTrainer`；暂不包含完整训练脚本。

## 边界

- `daxigua.core.engine.HeadlessGame` 属于游戏本体，只负责规则、物理、状态和动作候选。
- `daxigua_rl.env.DaxiguaEnv` 属于 RL 包，只通过 `HeadlessGame` 访问游戏。
- `daxigua` 不允许 import `daxigua_rl`。
- `daxigua_rl` 不应 import `daxigua.app.Board`、pygame 渲染、HUD、音频或手动输入代码。

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

## RL 环境接口

### `DaxiguaEnv`

主要方法：

- `reset(seed=None, fruit_queue=None) -> (GameState, info)`
- `action_candidates() -> list[ActionCandidate]`
- `step(action_index) -> (GameState, reward, terminated, truncated, info)`

这里的 `step(action_index)` 表示一次完整投放，不是一帧游戏画面。

默认 reward：

```text
reward = score_delta
if terminated:
    reward += terminal_penalty
```

后续复杂奖励设计应放在 `daxigua_rl`，不要写回游戏规则层。

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

`FruitState` 包含位置、速度、等级、半径、年龄、稳定状态和到边界/危险线的距离。

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
- `GraphTensor.to(device=None, dtype=None)`

最小前向链路：

```text
DaxiguaEnv.reset()
    -> GameState + ActionCandidate
    -> GraphBuilder.build(...)
    -> graph_to_tensor(...)
    -> GNNQNetwork(...)
    -> q_values[action_count]
```

当前模型只验证前向链路和反向传播是否可运行；Q 值在训练前没有策略意义。

## 训练经验结构

当前 `daxigua_rl.training` 包提供：

- `Transition`: 一条 DQN 经验记录。
- `ReplayBuffer`: 固定容量内存回放池。
- `RolloutCollector`: 单进程 rollout 采集器。
- `DQNTrainer`: 标准 DQN 单步更新器。

字段含义：

- `graph`: 当前状态图，也就是状态 `s`。
- `action_offset`: 被选择动作在 `q_values` 中的下标，也就是训练 loss 读取 `q_values[action_offset]` 的位置。
- `reward`: 执行动作后的即时奖励。
- `next_graph`: 下一状态图，也就是状态 `s'`；terminal transition 可以为 `None`。
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
target = reward + gamma * max_next_q   # 仅当 transition.can_bootstrap 为 True
target = reward                        # terminal/truncated transition
```

### `ReplayBuffer`

第一版接口：

- `ReplayBuffer(capacity=100_000, seed=None)`: 创建固定容量回放池。
- `push(transition)`: 写入一条 `Transition`。
- `extend(transitions)`: 批量写入。
- `sample(batch_size) -> tuple[Transition, ...]`: 随机无放回采样。
- `is_ready(batch_size) -> bool`: 判断是否足够采样一个 batch。
- `clear()`: 清空。
- `len(buffer)`: 当前已保存经验数量。

当前约定：

- 默认容量是 `100_000`，也就是十万条经验。
- 容量满后覆盖最旧经验。
- `sample()` 返回原始 `Transition` 元组，不在 buffer 层拼 tensor batch。
- 第一版使用均匀随机采样，不做优先经验回放。

### `RolloutCollector`

导入方式：

```python
from daxigua_rl.training import RolloutCollector
```

第一版接口：

- `RolloutCollector(env, graph_builder, replay_buffer, model=None, policy=None, seed=None)`: 创建单环境采集器。
- `reset(seed=None, fruit_queue=None)`: 显式重置环境并开始新 episode。
- `collect_steps(step_count, epsilon=1.0) -> RolloutStats`: 收集指定数量的 transition 并写入 replay buffer。

当前采集流程：

```text
当前 GameState + action_candidates
-> GraphBuilder.build(...)
-> epsilon-greedy 选择 action_offset
-> DaxiguaEnv.step(action_offset)
-> 构建 next_graph
-> Transition(...)
-> ReplayBuffer.push(...)
```

当前约定：

- `epsilon=1.0` 时可以不提供模型，collector 会完全随机探索。
- `epsilon<1.0` 时必须提供 Q 网络模型，用于 greedy 分支。
- 采集时模型会临时切到 `eval()`，结束后恢复原本训练模式。
- episode 结束后 collector 会自动 `reset()` 并继续采集，直到达到指定 transition 数。
- `RolloutCollector` 依赖 PyTorch，因此不会被 `daxigua_rl` 顶层自动导入。

`RolloutStats` 提供：

- `steps`: 本次写入多少条 transition。
- `episodes`: 本次完成多少局。
- `total_reward`: 本次总 reward。
- `episode_rewards`、`episode_lengths`、`episode_scores`: 本次已结束 episode 的统计。
- `random_actions`、`greedy_actions`: 探索/利用动作数量。
- `buffer_size`: 采集后 replay buffer 大小。

### `DQNTrainer`

导入方式：

```python
from daxigua_rl.training import DQNTrainer, DQNTrainerConfig
```

第一版接口：

- `DQNTrainer(online_model, target_model, replay_buffer, optimizer, config=None, loss_fn=None)`
- `train_step() -> DQNTrainStats`
- `is_ready() -> bool`
- `sync_target_model()`

默认配置：

```python
DQNTrainerConfig(
    gamma=0.99,
    batch_size=32,
    target_update_interval=1000,
    grad_clip_norm=10.0,
    sync_target_on_init=True,
)
```

当前标准 DQN target：

```text
current_q = online_model(graph)[action_offset]

if transition.can_bootstrap:
    target = reward + gamma * max(target_model(next_graph))
else:
    target = reward
```

当前约定：

- 默认 loss 使用 `SmoothL1Loss`，也就是 Huber 风格损失。
- 初始化时会把 `online_model` 参数同步到 `target_model`。
- `target_model` 参数会被冻结，只用于无梯度推理。
- 每隔 `target_update_interval` 次 `train_step()` 同步一次 target network。
- 第一版是标准 DQN，不做 Double DQN。
- 第一版逐条图 forward，不做 GraphBatch。
- 默认使用梯度裁剪 `grad_clip_norm=10.0`。

`DQNTrainStats` 提供：

- `update_step`: 已完成更新次数。
- `loss`: 本次 TD loss。
- `mean_q`: 当前 Q 平均值。
- `mean_target`: target 平均值。
- `mean_reward`: reward 平均值。
- `mean_abs_td_error`: 平均绝对 TD 误差。
- `bootstrap_count`: batch 中使用 next_graph bootstrap 的 transition 数量。
- `grad_norm`: 裁剪前梯度范数。
- `target_synced`: 本次是否同步 target network。

## 后续扩展

- 完整训练脚本应继续放在 `daxigua_rl`，组合 `RolloutCollector`、`ReplayBuffer` 和 `DQNTrainer`。
- 多进程采样、replay buffer、模型训练也应在 `daxigua_rl` 内部实现。
- 如果未来需要性能优化，优先 profile `HeadlessGame`，再决定是否替换底层实现。
