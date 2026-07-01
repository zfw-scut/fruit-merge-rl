# 项目文件索引

最后更新：2026-07-01

## 项目定位

本项目当前是一个基于 `pygame` 和 `pymunk` 的《合成大西瓜》桌面小游戏。当前只保留游戏本体，旧实验代码和旧环境封装已经删除。后续自动游玩/RL 能力会通过 `daxigua_rl` 包和游戏本体暴露的稳定接口接入，不让游戏本体反向依赖自动化代码。

## 核心源码

| 路径 | 作用 | 主要入口或可复用点 |
| --- | --- | --- |
| `Main.py` | 兼容旧启动方式的薄入口，将 `src/` 加入 import 路径后调用 `daxigua.app.main()`。 | `main()` |
| `src/daxigua/app.py` | 游戏应用入口和当前表现层实现。负责固定窗口、输入、正式渲染、鼠标跟随投放、预览线、顶部独立信息层、待投放水果队列、HUD、粒子、飘字、震动和音效反馈。 | `Board.next_frame()`、`Board.run()`、`main()` |
| `src/daxigua/config.py` | 项目路径和基础配置。 | `PROJECT_ROOT`、`FRUIT_ASSET_DIR`、`DEFAULT_WINDOW_SIZE`、`SPAWN_LINE_Y`、`FPS` |
| `src/daxigua/core/board.py` | 游戏公共逻辑。负责 pygame 画布、pymunk 物理世界、动态墙体、碰撞合成、计分、失败检测，并向表现层暴露合成事件钩子。 | `GameBoard`、`resize_world()`、`create_ball()`、`setup_collision_handler()`、`check_fail()` |
| `src/daxigua/core/engine.py` | 无渲染游戏引擎。负责 headless 物理世界、投放、队列推进、动作候选、状态快照和稳定推进，供训练环境调用。 | `HeadlessGame` |
| `src/daxigua/core/fruit.py` | 水果显示精灵和贴图加载。根据等级创建单一 `Fruit` 显示对象，并复用 `rules.py` 中的半径规则。 | `create_fruit(level, x, y)`、`Fruit`、`fruit_image_path()`、`load_fruit_image()` |
| `src/daxigua/core/rules.py` | 纯规则常量和辅助函数。集中维护水果半径、队列长度、随机生成范围、合成分数和物理半径。 | `FRUIT_RADII`、`FRUIT_QUEUE_LENGTH`、`fruit_radius()`、`merge_score()` |
| `src/daxigua/core/state.py` | 训练友好的纯数据状态结构。 | `GameState`、`FruitState`、`ActionCandidate`、`DropResult`、`PhysicsResult` |
| `src/daxigua_rl/` | 自动游玩/RL 相关代码。游戏本体不得 import 它。训练主链路通过 `HeadlessGame` 访问游戏；观看脚本可在 RL 侧懒加载真实 `Board`。 | `DaxiguaEnv`、`DaxiguaEnvConfig`、`README.md` 中记录边界规则 |
| `src/daxigua_rl/env.py` | 类 Gymnasium 的 RL 环境壳层。一次 `step(action_index)` 表示一次投放和无渲染物理稳定。 | `DaxiguaEnv.reset()`、`DaxiguaEnv.step()`、`action_candidates()` |
| `src/daxigua_rl/reward.py` | 强化学习 reward shaping 逻辑。根据动作前后状态和物理结果计算奖励，并返回奖励明细。 | `RewardConfig`、`RewardBreakdown`、`compute_reward()` |
| `src/daxigua_rl/playable_adapter.py` | 真实 pygame 游戏窗口到 RL 输入结构的适配层。把正在运行的 `Board` 转成 `GameState` 和 `ActionCandidate`，用于观看模型实际游玩。 | `board_game_state()`、`board_action_candidates()` |
| `src/daxigua_rl/graph/` | GNN 图构建相关代码。负责把游戏状态和动作候选转换成模型输入图，并提供训练实验用的特征消融层。 | `GraphBuilder`、`GraphAblator` |
| `src/daxigua_rl/graph/schema.py` | 框架无关的图数据结构和节点/边特征名。 | `GraphData`、`GraphNodeRef`、`GraphEdgeRef`、`NODE_FEATURE_NAMES`、`EDGE_FEATURE_NAMES` |
| `src/daxigua_rl/graph/builder.py` | 从 `GameState` 和 `ActionCandidate` 构建 GNN 输入图。 | `GraphBuilder.build()` |
| `src/daxigua_rl/graph/ablation.py` | 图特征消融工具。在不改变图维度的前提下按配置置零部分节点或边特征。 | `GraphAblator`、`FeatureAblationConfig`、`FeatureMask`、`ABLATION_PRESETS` |
| `src/daxigua_rl/graph/tensor.py` | PyTorch 张量转换层。把框架无关 `GraphData` 转成单图 `GraphTensor`，并把多张图拼成不连通 `GraphBatch`。 | `graph_to_tensor()`、`collate_graph_tensors()`、`GraphTensor`、`GraphBatch` |
| `src/daxigua_rl/models/` | 强化学习模型代码。当前只包含最小 GNN-Q 前向模型，不包含训练循环。 | `GNNQNetwork` |
| `src/daxigua_rl/models/gnn_q.py` | 统一图 message passing Q 网络。输入 `GraphData`、`GraphTensor` 或 `GraphBatch`，输出单图或批量扁平动作 Q 值。 | `GNNQNetwork.forward()`、`MessagePassingLayer` |
| `src/daxigua_rl/training/` | 强化学习训练侧数据结构和后续训练组件目录。当前包含经验记录、回放池、单进程采集器和 DQN 更新器。 | `Transition`、`ReplayBuffer`、`RolloutCollector`、`DQNTrainer` |
| `src/daxigua_rl/training/transition.py` | DQN 经验记录。保存当前图、动作下标、奖励、下一状态图和终止标记。 | `Transition` |
| `src/daxigua_rl/training/tensor_transition.py` | DQN 张量化经验记录。保存 CPU `GraphTensor`，用于正式训练主链路和 GraphBatch 拼接。 | `TensorTransition` |
| `src/daxigua_rl/training/replay_buffer.py` | DQN 固定容量经验回放池。保存经验对象，容量满后覆盖最旧经验，并支持均匀随机采样。 | `ReplayBuffer` |
| `src/daxigua_rl/training/collector.py` | 单进程 rollout 采集器。使用 epsilon-greedy 动作选择让模型或随机策略游玩无渲染环境，并把 CPU `TensorTransition` 写入 `ReplayBuffer`。 | `RolloutCollector`、`EpsilonGreedyPolicy`、`RolloutStats` |
| `src/daxigua_rl/training/dqn.py` | 标准 DQN 单步更新器。从 `ReplayBuffer` 采样，拼接 `GraphBatch`，计算 TD target 和 SmoothL1Loss，更新 online Q 网络，并定期同步 target network。 | `DQNTrainer`、`DQNTrainerConfig`、`DQNTrainStats` |
| `src/daxigua_rl/scripts/` | 强化学习命令行脚本目录。用于放正式训练、评估、观看、导出等入口。 | `train_dqn.py`、`watch_dqn.py` |
| `src/daxigua_rl/scripts/train_dqn.py` | 第一版正式 DQN 训练入口。组合 collector、replay buffer、DQN trainer、epsilon 衰减、日志、checkpoint、评估和 matplotlib 曲线图。 | `python -m daxigua_rl.scripts.train_dqn` |
| `src/daxigua_rl/scripts/watch_dqn.py` | 第一版 DQN 可视化观看入口。加载训练 checkpoint，复用原 pygame `Board` 画面，并在 RL 侧注入自动控制器选择落点。 | `python -m daxigua_rl.scripts.watch_dqn --checkpoint ...` |

## 资源和说明

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `assets/fruits/` | 水果图片资源目录，包含 `01.png` 到 `11.png`。 | 游戏运行时直接读取。 |
| `assets/fruits.zip` | 原始水果图片压缩包归档。 | 不参与运行，只作资源备份。 |
| `README.md` | 当前项目说明，包含游戏运行方式和操作说明。 | 已更新为游戏本体说明。 |
| `requirements.txt` | 当前游戏依赖文件。 | 只保留 `pygame` 和 `pymunk`。 |
| `LICENSE` | 开源许可证。 | Apache 2.0。 |

## 临时工具

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `tools/temporary_rollout_smoke_test.py` | 临时 GNN rollout 验证脚本。用于检查 `DaxiguaEnv -> GraphBuilder -> GNNQNetwork -> step()` 链路是否闭合。 | 不是正式训练入口；验证完成或正式训练脚本落地后可删除或改造。 |

## 测试目录

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `tests/test_graph_batch_training.py` | GraphBatch 和张量化 DQN 训练链路测试。验证批量图前向与单图前向一致，并确认 collector/trainer 可以写入和训练 `TensorTransition`。 | 使用标准库 `unittest`，在 `python-torch` 环境中运行。 |
| `tests/test_epsilon_schedule.py` | epsilon 衰减曲线测试。验证 smooth schedule 的关键锚点、单调性，以及 linear schedule 的旧行为。 | 使用标准库 `unittest`。 |

## 文档目录

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `docs/README.md` | 文档目录入口。 | 说明文档阅读顺序。 |
| `docs/CODING_STYLE.md` | 项目编码风格说明。 | 当前记录游戏源码采用教学型详细注释，后续改代码时应同步维护注释。 |
| `docs/codex/` | Codex 较大修改记录。 | 每次较大修改按编号追加记录。 |
| `docs/project_map/` | 项目文件职责索引。 | 结构变化后需要同步更新。 |
| `docs/learning/` | 强化学习项目化学习文档。 | 放学习路线、阶段规划、练习说明和学习笔记。 |
| `docs/rl/` | 强化学习算法和环境接口设计文档。 | 当前包含 GNN 状态图设计参考，后续模型搭建前优先阅读。 |
| `docs/rl/INTERFACE_V0.md` | RL v0 接口说明。 | 记录 `HeadlessGame`、`DaxiguaEnv`、状态数据和边界规则。 |

## 学习练习目录

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `practice/` | 学习者练习代码、实验脚本和草稿的空白工作区。 | 初始保持干净，仅用 `.gitkeep` 保留目录。 |

## 本地和生成目录

| 路径 | 作用 | 处理建议 |
| --- | --- | --- |
| `.git/` | Git 仓库元数据。 | 不手动修改。 |
| `.vscode/` | VS Code 本地配置和缓存。 | 已忽略。 |
| `__pycache__/` | Python 字节码缓存。 | 已忽略。 |
| `src/**/__pycache__/` | 包内 Python 字节码缓存。 | 已忽略。 |
| `runs/` | DQN 训练输出目录，包含 `metrics.csv`、checkpoint 和曲线图。 | 已忽略。 |
| `.agents/`、`.codex/` | 当前工作环境辅助目录。 | 不属于原项目核心源码。 |

## 可复用组件

- `GameBoard`：后续优化游戏时可复用的物理和合成基类。
- `create_fruit(level, x, y)`：统一创建 pygame 水果显示对象，避免外部关心贴图路径和 rect 同步细节。
- `load_fruit_image(path, size)`：缓存水果贴图加载和缩放结果，避免重复磁盘读取。
- `create_ball(space, x, y, m, r, i)`：统一创建 pymunk 圆形刚体。
- `Board.fruit_queue`：手动游戏的待投放水果队列，q0 是当前水果，q1 到 q3 是后续水果。
- `HeadlessGame`：后续训练环境优先使用的无渲染游戏接口，不依赖 pygame 窗口。
- `DaxiguaEnv`：隔离在 `daxigua_rl` 中的 RL 环境壳层，只通过 `HeadlessGame` 访问游戏。
- `compute_reward()`：RL reward shaping 入口，组合合成得分、存活奖励、高度变化、危险高度和终局惩罚，并返回 `RewardBreakdown`。
- `GraphBuilder`：把无渲染游戏状态和候选动作转换成框架无关 `GraphData`，供后续 GNN/Q 网络使用。
- `GraphAblator`：训练实验用的图特征消融层，通过置零特征对比不同信息组对模型的影响。
- `graph_to_tensor()`：把 `GraphData` 转成 PyTorch 张量，形成 `node_features`、`edge_index`、`edge_features` 和 `action_node_indices`。
- `collate_graph_tensors()`：把多张 `GraphTensor` 拼成不连通 `GraphBatch`，记录每张图的 action slice。
- `GNNQNetwork`：当前 GNN-Q 前向模型，输入单图输出 `[action_count]`，输入 `GraphBatch` 输出 `[total_action_count]`。
- `Transition`：DQN 训练使用的一条经验记录，保存 `graph`、`action_offset`、`reward`、`next_graph`、`terminated` 和 `truncated`。
- `TensorTransition`：正式训练主链路使用的张量化经验记录，保存 CPU `GraphTensor`，避免重复转换。
- `ReplayBuffer`：固定容量经验回放池，默认保存十万条经验对象，采样时返回原始对象元组。
- `RolloutCollector`：单进程经验采集器，串联 `DaxiguaEnv`、`GraphBuilder`、Q 网络和 `ReplayBuffer`，用于收集张量化训练经验。
- `DQNTrainer`：标准 DQN 单步更新器，使用 GraphBatch、online/target 双网络、SmoothL1Loss 和梯度裁剪更新 Q 网络。
- `train_dqn.py`：第一版训练入口，输出 `metrics.csv`、`checkpoints/latest.pt` 和 `plots/training_curves.png`。
- `board_game_state()` / `board_action_candidates()`：把原 pygame `Board` 的实时局面转换成 RL 图构建所需的数据结构。
- `watch_dqn.py`：第一版模型可视化观看入口，用真实游戏窗口检查 checkpoint 的实际操作效果。
- `resize_world(width, height)`：按窗口尺寸重设 pygame 画布和 pymunk 边界。当前手动游戏窗口固定，此函数主要作为内部调试或未来实验工具保留。
- `setup_collision_handler()`：水果合成逻辑所在位置，已兼容新版 `pymunk.Space.on_collision`，并在合成后调用可选的 `on_fruit_merged()`。

## 已知注意事项

- 游戏运行时直接读取 `assets/fruits/`，不再需要手动解压资源。
- 当前手动游戏窗口固定为 `400x800`，不再通过拖动窗口边框改变场地大小。
- 顶部信息层和当前悬浮水果层已经分开；生成线固定为 `180px`，用于避免待投放队列与当前水果视野冲突。
- `daxigua` 游戏本体不得 import `daxigua_rl`；训练、环境和模型代码只通过稳定游戏接口访问游戏。
- `watch_dqn.py` 是视觉检查用入口，会在脚本内部懒加载 `daxigua.app.Board` 并打开真实 pygame 窗口；这不是训练路径，也不要求游戏本体 import RL。
- `Transition` 不依赖 PyTorch，保存的是框架无关 `GraphData`；当前正式训练主链路改用 `TensorTransition` 保存 CPU `GraphTensor`。
- `daxigua_rl.graph.tensor` 和 `daxigua_rl.models` 依赖 PyTorch；它们不会在 `daxigua_rl` 顶层自动导入，避免非训练环境被强制要求安装 torch。
- `RolloutCollector` 和 `DQNTrainer` 依赖 PyTorch 模型前向；它们通过 `daxigua_rl.training` 懒加载导入，不放进 `daxigua_rl` 顶层导出。
- `train_dqn.py` 依赖 PyTorch 和 matplotlib；matplotlib 使用 `Agg` 后端生成 png，并把缓存目录放到当前 run 目录下。
- `tools/temporary_rollout_smoke_test.py` 依赖 PyTorch，建议在 `python-torch` conda 环境中运行；它只做临时链路验证，不训练模型。
- 当前 `src/daxigua/core/board.py` 已为 `pymunk 7.3.0` 做兼容处理。
- 当前 `src/daxigua/app.py` 仍集中承载表现层细节；后续如确实需要拆分，再创建对应表现层模块。
