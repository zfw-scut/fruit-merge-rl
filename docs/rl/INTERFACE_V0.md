# RL 接口 v0

## 目标

本接口用于先跑通强化学习训练闭环：

```text
reset -> observe -> choose action -> step -> reward / next_state / done
```

当前只提供无渲染游戏接口和 RL 环境壳层，不包含模型、GNN 图构建、replay buffer 或训练循环。

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

## 后续扩展

- GNN 图构建器应单独放在 `daxigua_rl`，读取 `GameState` 和 `ActionCandidate`。
- 多进程采样、replay buffer、模型训练也应在 `daxigua_rl` 内部实现。
- 如果未来需要性能优化，优先 profile `HeadlessGame`，再决定是否替换底层实现。
