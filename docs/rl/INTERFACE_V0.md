# RL 接口 v0

## 目标

本接口用于先跑通强化学习训练闭环：

```text
reset -> observe -> choose action -> step -> reward / next_state / done
```

当前提供无渲染游戏接口、RL 环境壳层、GNN 图构建基础设施、GNN-Q 前向模型、`Transition` / `TensorTransition` 经验记录、基础 `ReplayBuffer`、单进程 `RolloutCollector`、标准 `DQNTrainer` 和第一版 DQN 训练入口。

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

## RL 环境接口

### `DaxiguaEnv`

主要方法：

- `reset(seed=None, fruit_queue=None) -> (GameState, info)`
- `action_candidates() -> list[ActionCandidate]`
- `step(action_index) -> (GameState, reward, terminated, truncated, info)`

这里的 `step(action_index)` 表示一次完整投放，不是一帧游戏画面。

默认 reward 已从单一 `score_delta` 扩展为可配置 reward shaping：

```text
reward =
    score_reward
    + survival_bonus
    + height_delta_reward
    + danger_penalty
    + terminal_penalty
```

当前默认配置：

```python
RewardConfig(
    score_scale=1.0,
    survival_bonus=0.05,
    height_delta_weight=0.02,
    danger_height_weight=1.0,
    terminal_penalty=-100.0,
)
```

各项含义：

- `score_reward = score_delta * score_scale`: 保留真实合成分数作为主奖励。
- `survival_bonus`: 没有死亡时给一个很小的存活奖励。
- `height_delta_reward`: 堆叠高度降低时为正，堆叠升高时为负。
- `danger_penalty`: 堆叠越接近顶部危险线，持续惩罚越大。
- `terminal_penalty`: 游戏失败时的终局惩罚。

`DaxiguaEnv.step()` 会在 `info["reward_breakdown"]` 中返回 `RewardBreakdown`，用于查看本次 reward 的组成。复杂奖励设计应继续放在 `daxigua_rl`，不要写回游戏规则层。

训练入口会把采集窗口内的 reward breakdown 均值写入 `metrics.csv`：

- `collect_mean_reward_total`
- `collect_mean_score_reward`
- `collect_mean_survival_bonus`
- `collect_mean_height_delta_reward`
- `collect_mean_danger_penalty`
- `collect_mean_terminal_penalty`
- `collect_mean_previous_height_ratio`
- `collect_mean_next_height_ratio`
- `collect_mean_height_delta_ratio`

同时会在 `plots/reward_breakdown_curves.png` 中单独画出奖励组成和高度比例曲线，方便观察辅助奖励是否压过真实得分奖励。

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

- `Transition`: 框架无关 DQN 经验记录，保留给调试和对照。
- `TensorTransition`: 张量化 DQN 经验记录，正式训练主链路使用它。
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

接口：

- `ReplayBuffer(capacity=100_000, seed=None)`: 创建固定容量回放池。
- `push(transition)`: 写入一条经验对象。
- `extend(transitions)`: 批量写入。
- `sample(batch_size) -> tuple[...]`: 随机无放回采样。
- `is_ready(batch_size) -> bool`: 判断是否足够采样一个 batch。
- `clear()`: 清空。
- `len(buffer)`: 当前已保存经验数量。

当前约定：

- 默认容量是 `100_000`，也就是十万条经验。
- 容量满后覆盖最旧经验。
- buffer 只负责保存和采样对象，不关心对象内部是 `Transition` 还是 `TensorTransition`。
- 当前正式训练主链路由 `RolloutCollector` 写入 CPU `TensorTransition`。
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
-> graph_to_tensor(...)
-> epsilon-greedy 选择 action_offset
-> DaxiguaEnv.step(action_offset)
-> 构建 next_graph
-> TensorTransition(...)
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
current_batch = collate_graph_tensors(batch.graph)
current_q_flat = online_model(current_batch)
current_q = current_q_flat[action_slice.start + action_offset]

if transition.can_bootstrap:
    next_batch = collate_graph_tensors(bootstrap_next_graphs)
    next_q_flat = target_model(next_batch)
    target = reward + gamma * max(next_q_flat[each_next_action_slice])
else:
    target = reward
```

当前约定：

- 默认 loss 使用 `SmoothL1Loss`，也就是 Huber 风格损失。
- 初始化时会把 `online_model` 参数同步到 `target_model`。
- `target_model` 参数会被冻结，只用于无梯度推理。
- 每隔 `target_update_interval` 次 `train_step()` 同步一次 target network。
- 第一版是标准 DQN，不做 Double DQN。
- 当前使用 GraphBatch，把 batch 内多张图拼成不连通大图执行批量 forward。
- ReplayBuffer 正式训练路径保存 CPU `TensorTransition`，训练时再把 `GraphBatch` 搬到模型设备，因此可直接支持 GPU batch 训练。
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

## 训练入口

第一版正式训练脚本：

```text
src/daxigua_rl/scripts/train_dqn.py
```

运行方式：

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u -m daxigua_rl.scripts.train_dqn
```

注意：普通 `conda run` 会捕获子进程输出，可能导致进度信息等到训练结束才一次性显示。需要使用 `--no-capture-output` 才能实时看到每 3 秒的进度心跳。

默认训练流程：

```text
warmup 随机收集经验
-> 每轮 collect_per_update 条新经验
-> DQNTrainer.train_step()
-> epsilon 按 schedule 衰减
-> 每 3 秒打印轻量进度
-> 终端日志
-> metrics.csv
-> checkpoint
-> greedy 评估
-> matplotlib 训练曲线图
```

默认输出目录：

```text
runs/dqn_YYYYMMDD_HHMMSS/
├── config.json
├── metrics.csv
├── episode_metrics.csv
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   └── step_XXXXXXXX.pt
└── plots/
    └── training_curves.png
```

`metrics.csv` 是核心可视化数据源，记录：

- update step、环境步数、epsilon、buffer 大小。
- loss、mean Q、mean target、mean reward、TD error、grad norm。
- 采集阶段 episode 统计。
- greedy 评估均分、最高分、最低分、历史最高分、平均 reward、平均 episode 长度。
- 采样和训练速度。

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

第一版真实游戏窗口观看脚本：

```text
src/daxigua_rl/scripts/watch_dqn.py
```

运行方式：

```bash
PYTHONPATH=src conda run --no-capture-output -n python-torch python -u -m daxigua_rl.scripts.watch_dqn \
  --checkpoint runs/dqn_baseline_h128_l3_10k_eps10k/checkpoints/latest.pt
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

- 后续训练脚本仍应继续放在 `daxigua_rl.scripts`，组合 `RolloutCollector`、`ReplayBuffer` 和 `DQNTrainer`。
- 多进程采样、replay buffer、模型训练也应在 `daxigua_rl` 内部实现。
- 如果未来需要性能优化，优先 profile `HeadlessGame`，再决定是否替换底层实现。
