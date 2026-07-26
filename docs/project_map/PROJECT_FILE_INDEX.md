# 项目文件索引

最后更新：2026-07-27

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
| `src/daxigua/core/state.py` | 训练友好的纯数据状态结构；水果和动作候选同时公开显示半径与真实碰撞半径。 | `GameState`、`FruitState`、`ActionCandidate`、`DropResult`、`PhysicsResult` |
| `src/daxigua_rl/` | 自动游玩/RL 相关代码。游戏本体不得 import 它。训练主链路通过 `HeadlessGame` 访问游戏；观看脚本可在 RL 侧懒加载真实 `Board`。 | `DaxiguaEnv`、`DaxiguaEnvConfig`、`README.md` 中记录边界规则 |
| `src/daxigua_rl/env.py` | 类 Gymnasium 的 RL 环境壳层。一次 step 表示一次投放和无渲染物理稳定；根据 `DaxiguaEnvConfig.state_analyzer_config` 在 worker 内创建分析器、缓存相邻 `StateAnalysis`，接收 collector 的 `TransitionKey` 并计算 Reward V2。 | `DaxiguaEnv.reset()`、`DaxiguaEnv.step(action_index, transition_key=...)`、`action_candidates()` |
| `src/daxigua_rl/reward.py` | Reward V2 纯计算层。按合成等级计算指数 task utility，并以相邻 `StateAnalysis` 的 C/R/K potential 差完成 shaping。 | `RewardConfig`、`RewardBreakdown`、`merge_utility()`、`compute_state_potential()`、`compute_reward()` |
| `src/daxigua_rl/playable_adapter.py` | 真实 pygame 游戏窗口到 RL 输入结构的适配层。把正在运行的 `Board` 转成 `GameState` 和 `ActionCandidate`，用于观看模型实际游玩。 | `board_game_state()`、`board_action_candidates()` |
| `src/daxigua_rl/attribution/` | 完整状态归因模块。当前提供 worker 内使用的纯只读数据契约和静态状态分析器；分析结果已接 Reward V2，后续再接 tracker 和因果 replay。 | `StateAnalyzer`、`StateAnalysis`、`FruitAnalysis`、`FreeSpaceRegionAnalysis` |
| `src/daxigua_rl/attribution/schema.py` | 固定 15 动作状态分析结构、时间语义、队列槽位、自由空间区域和廉价引用不变量；全部类型可 pickle/Windows spawn。 | `StateAnalysis`、`FruitAnalysis`、`FreeSpaceRegionAnalysis`、`SupportEdge`、`ContactInfluenceEdge`、`PartnerComponent`、`ChainMotif`、`QueueLaneAnalysis` |
| `src/daxigua_rl/attribution/state_analyzer.py` | 在动作前边界执行只读静态分析：解析圆形竖直列计算 15 动作可达性/队列容量，规范最小水果探针栅格计算顶部连通空间和空腔，并构建支撑、伙伴与基础连锁结构。 | `StateAnalyzer`、`StateAnalyzerConfig` |
| `src/daxigua_rl/graph/` | GNN 图构建相关代码。负责把游戏状态和动作候选转换成模型输入图，并提供训练实验用的特征消融层。 | `GraphBuilder`、`GraphAblator` |
| `src/daxigua_rl/graph/schema.py` | 框架无关的图数据结构和节点/边特征名。 | `GraphData`、`GraphNodeRef`、`GraphEdgeRef`、`NODE_FEATURE_NAMES`、`EDGE_FEATURE_NAMES` |
| `src/daxigua_rl/graph/builder.py` | 从 `GameState` 和 `ActionCandidate` 构建 GNN 输入图；几何、接触和投放路径关系使用真实碰撞半径，图维度保持不变。 | `GraphBuilder.build()` |
| `src/daxigua_rl/graph/ablation.py` | 图特征消融工具。在不改变图维度的前提下按配置置零部分节点或边特征。 | `GraphAblator`、`FeatureAblationConfig`、`FeatureMask`、`ABLATION_PRESETS` |
| `src/daxigua_rl/graph/tensor.py` | PyTorch 张量转换层。把框架无关 `GraphData` 转成单图 `GraphTensor`，并把多张图拼成不连通 `GraphBatch`。 | `graph_to_tensor()`、`collate_graph_tensors()`、`GraphTensor`、`GraphBatch` |
| `src/daxigua_rl/models/` | 强化学习模型代码。当前只包含最小 GNN-Q 前向模型，不包含训练循环。 | `GNNQNetwork` |
| `src/daxigua_rl/models/gnn_q.py` | 统一图 message passing Q 网络。输入 `GraphData`、`GraphTensor` 或 `GraphBatch`，输出单图或批量扁平动作 Q 值。 | `GNNQNetwork.forward()`、`MessagePassingLayer` |
| `src/daxigua_rl/training/` | 强化学习训练侧数据结构和后续训练组件目录。当前包含张量化经验记录、分层回放池、单进程/多进程采集器和 DQN 更新器。 | `TensorTransition`、`ReplayBuffer`、`RolloutCollector`、`ParallelRolloutCollector`、`DQNTrainer` |
| `src/daxigua_rl/training/identity.py` | 定义一次训练 run 内稳定、可哈希、可跨进程序列化的轨迹身份。 | `TransitionKey(worker_id, episode_id, step_index)` |
| `src/daxigua_rl/training/tensor_transition.py` | DQN 张量化经验记录。保存 CPU `GraphTensor`，用于正式训练主链路和 GraphBatch 拼接。 | `TensorTransition` |
| `src/daxigua_rl/training/replay_buffer.py` | DQN 固定容量经验回放池。小实验默认纯内存；大容量训练可使用热内存 + 冷磁盘分层存储，并对冷段中的共享图做去重。 | `ReplayBuffer` |
| `src/daxigua_rl/training/collector.py` | 单进程 rollout 采集器。生成并向环境传递稳定 `TransitionKey`，累计 Reward V2/shaping p95 与 StateAnalyzer 性能，保留 truncated bootstrap，并把 CPU `TensorTransition` 写入 replay。 | `RolloutCollector`、`EpsilonGreedyPolicy`、`RolloutStats` |
| `src/daxigua_rl/training/parallel_collector.py` | 多进程 rollout 调度器。每个 Windows spawn worker 独立持有环境、StateAnalyzer 和 CPU 模型；主进程只回收 transition 与聚合 reward/分析统计，不传完整分析对象。 | `ParallelRolloutCollector` |
| `src/daxigua_rl/training/dqn.py` | 标准 DQN 单步更新器。从 `ReplayBuffer` 采样，拼接 `GraphBatch`，计算 TD target 和 SmoothL1Loss，更新 online Q 网络，并记录训练阶段 profiling。 | `DQNTrainer`、`DQNTrainerConfig`、`DQNTrainStats` |
| `src/daxigua_rl/scripts/` | 强化学习命令行脚本目录。用于放正式训练、评估、观看、导出等入口。 | `train_dqn.py`、`watch_dqn.py` |
| `src/daxigua_rl/scripts/train_dqn.py` | 第一版正式 DQN 训练入口。组合 collector、replay buffer、DQN trainer、epsilon 衰减、日志、checkpoint、评估和曲线；Reward V2 与 DQN 共用 gamma，并记录 task/potential、shaping p95 和分析器性能。 | `python -m daxigua_rl.scripts.train_dqn`；输出 `metrics.csv`、`episode_metrics.csv`、`plots/training_curves.png`、`plots/reward_breakdown_curves.png`。 |
| `src/daxigua_rl/scripts/watch_dqn.py` | 第一版 DQN 可视化观看入口。加载训练 checkpoint，复用原 pygame `Board` 画面，并在 RL 侧注入自动控制器选择落点。 | `python -m daxigua_rl.scripts.watch_dqn --checkpoint ...` |
| `src/daxigua_rl/scripts/compare_physics_modes.py` | accurate/fast headless 物理模式对比工具。用于测试降低 fps、最大物理帧、稳定帧和 Pymunk 迭代次数后的速度收益与游戏分布偏移。 | `python -m daxigua_rl.scripts.compare_physics_modes --checkpoint ...`；输出 `summary.csv`、`episode_metrics.csv` 和 `plots/physics_mode_comparison.png`。 |
| `configs/` | 项目配置文件目录。用于存放训练、实验或运行参数配置。 | `train_dqn_fast30_parallel.toml` |
| `configs/train_dqn_fast30_parallel.toml` | DQN fast30 并行训练参数配置。集中记录训练、并行和 Reward V2 的 `lambda_phi`、C/R/K 权重及默认关闭的终局惩罚。 | 由 `train_dqn.py --config` 或 `scripts/train_dqn.sh` 读取。 |
| `scripts/` | 项目级启动脚本目录。只放薄启动器，具体训练参数放在 `configs/`。 | `train_dqn.sh` |
| `scripts/train_dqn.sh` | DQN 训练启动器。默认读取 `configs/train_dqn_fast30_parallel.toml`，设置 `PYTHONPATH`，通过 `python-torch` conda 环境启动训练并 tee 日志。 | `./scripts/train_dqn.sh` |

## 资源和说明

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `assets/fruits/` | 水果图片资源目录，包含 `01.png` 到 `11.png`。 | 游戏运行时直接读取。 |
| `assets/fruits.zip` | 原始水果图片压缩包归档。 | 不参与运行，只作资源备份。 |
| `README.md` | 当前项目说明，包含游戏运行方式和操作说明。 | 已更新为游戏本体说明。 |
| `requirements.txt` | 当前游戏依赖文件。 | 只保留 `pygame` 和 `pymunk`。 |
| `LICENSE` | 开源许可证。 | Apache 2.0。 |

## 辅助工具

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `tools/cuda_stress_test.py` | 独立 PyTorch CUDA 计算压力测试脚本。只做矩阵乘法和可选显存预留，并采集 GPU、系统内存、进程内存和内核 NVIDIA/Xid 日志。 | 用于判断黑屏/Xid 是否能在脱离游戏和 RL 训练代码后复现；默认输出到 `runs/cuda_stress/<时间戳>/`。 |
| `tools/export_training_catalog.py` | 训练实验归档工具。扫描本地 `runs/`，识别 Reward V2 task/potential 与历史 Reward V1 指标，并从配置、训练指标和文件信息生成轻量实验目录。 | 不复制 checkpoint、replay 和完整指标 CSV；运行方式见 `docs/training_runs/README.md`。 |
| `tools/monitor_training_resources.py` | 训练资源旁路监控脚本。独立于训练入口，按固定间隔记录系统内存、swap、目标训练进程、NVIDIA GPU 和 GPU 计算进程。 | 用于定位长时间训练时的 OOM、显存压力、GPU 查询失败和显示栈异常；默认输出到 `runs/resource_monitor/<时间戳>/`。 |
| `tools/temporary_rollout_smoke_test.py` | 临时 GNN rollout 验证脚本。用于检查 `DaxiguaEnv -> GraphBuilder -> GNNQNetwork -> step()` 链路是否闭合。 | 不是正式训练入口；验证完成或正式训练脚本落地后可删除或改造。 |

## 测试目录

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `tests/test_graph_batch_training.py` | GraphBatch 和张量化 DQN 训练链路测试。验证批量图前向、next_graph 缓存、分层 replay、并行 collector、Reward V2 分析统计和 DQN 更新链路。 | 使用标准库 `unittest`，在 `python-torch` 环境中运行。 |
| `tests/test_attribution_foundations.py` | 完整状态归因基础语义测试。验证稳定窗口、truncated bootstrap、真实碰撞半径、图几何和 worker/episode/step 身份键。 | 使用标准库 `unittest`，在 `python-torch` 环境中运行。 |
| `tests/test_attribution_schema.py` | `StateAnalysis` 数据契约测试。验证深只读、15 位 mask、队列槽位、跨对象引用、时间语义、pickle 和真实 Windows spawn 往返。 | 使用标准库 `unittest`，在 `python-torch` 环境中运行。 |
| `tests/test_state_analyzer.py` | `StateAnalyzer` 人工几何场景测试。验证空棋盘、解析投放列与并列 blocker、支撑方向、伙伴/merge/ladder motif、封闭空腔、输入降级和左右镜像。 | 使用标准库 `unittest`，不依赖 Pymunk 随机稳定过程。 |
| `tests/test_reward_v2.py` | Reward V2 纯公式和契约测试。验证指数合成效用、potential 权重、轨迹望远镜关系、相邻分析身份以及 terminal/truncated 差异。 | 不推进真实 Pymunk 物理。 |
| `tests/test_reward_v2_integration.py` | Reward V2 环境与采集集成测试。验证 StateAnalysis 缓存、worker/episode/step 键、terminal/truncated、统计合并及 Windows spawn 边界。 | 使用标准库 `unittest`。 |
| `tests/test_epsilon_schedule.py` | epsilon 衰减曲线测试。验证 smooth schedule 的关键锚点、单调性，以及 linear schedule 的旧行为。 | 使用标准库 `unittest`。 |
| `tests/test_training_catalog.py` | 训练实验归档工具测试。验证新旧指标格式、奖励分解加权汇总、checkpoint 摘要和文档输出。 | 使用临时目录，不依赖本地 `runs/`。 |
| `tests/test_training_metrics.py` | 训练指标测试。验证 Reward V2 breakdown、shaping p95、StateAnalyzer 性能、gamma 同源、TOML 参数和 episode 指标。 | 使用标准库 `unittest`。 |
| `tests/test_compare_physics_modes.py` | 物理模式对比工具测试。验证 Reward V2 配置能够传入评估环境，且对比入口继续输出预期摘要。 | 使用标准库 `unittest`。 |

## 文档目录

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `docs/README.md` | 文档目录入口。 | 说明文档阅读顺序。 |
| `docs/CODING_STYLE.md` | 项目编码风格说明。 | 当前记录游戏源码采用教学型详细注释，后续改代码时应同步维护注释。 |
| `docs/codex/` | Codex 较大修改记录。 | 每次较大修改按编号追加记录。 |
| `docs/project_map/` | 项目文件职责索引。 | 结构变化后需要同步更新。 |
| `docs/training_runs/` | 可提交到 Git 的训练实验目录。 | 总览见 `INDEX.md`；每个实验保留摘要、配置、指标统计和原始产物索引。 |
| `docs/learning/` | 强化学习项目化学习文档。 | 放学习路线、阶段规划、练习说明和学习笔记。 |
| `docs/rl/` | 强化学习算法和环境接口设计文档。 | 当前包含 GNN 状态图设计参考，后续模型搭建前优先阅读。 |
| `docs/rl/CAUSAL_ATTRIBUTION_V1.md` | 第一次大规模训练的完整状态归因 V1 规格。 | 固定 Reward V2、状态分析、归因事件、因果 Q 排序、反事实预算、测试和长训流程；前五步基础语义、分析器和 Reward V2 已经落地。 |
| `docs/rl/gnn_daxigua_design_reference.md` | GNN 状态图节点、边和特征语义参考。 | 当前 `radius` 节点特征和相关几何边特征均表示真实碰撞半径。 |
| `docs/rl/INTERFACE_V0.md` | RL v0 接口说明。 | 记录 `HeadlessGame`、`DaxiguaEnv`、状态数据和边界规则。 |
| `docs/rl/TRAINING_SPEED_OPTIMIZATION_PLAN.md` | 训练速度优化计划。 | 记录 profiling、next_graph 缓存、并行采样、fast physics、图构建优化和日志频率等优化顺序。 |

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
| `runs/` | DQN 训练输出目录，包含 `metrics.csv`、`episode_metrics.csv`、checkpoint、replay 和曲线图。 | 已忽略；轻量归档由 `tools/export_training_catalog.py` 生成到 `docs/training_runs/`。 |
| `.agents/`、`.codex/` | 当前工作环境辅助目录。 | 不属于原项目核心源码。 |

## 可复用组件

- `GameBoard`：后续优化游戏时可复用的物理和合成基类。
- `create_fruit(level, x, y)`：统一创建 pygame 水果显示对象，避免外部关心贴图路径和 rect 同步细节。
- `load_fruit_image(path, size)`：缓存水果贴图加载和缩放结果，避免重复磁盘读取。
- `create_ball(space, x, y, m, r, i)`：统一创建 pymunk 圆形刚体。
- `Board.fruit_queue`：手动游戏的待投放水果队列，q0 是当前水果，q1 到 q3 是后续水果。
- `HeadlessGame`：后续训练环境优先使用的无渲染游戏接口，不依赖 pygame 窗口。
- `DaxiguaEnv`：隔离在 `daxigua_rl` 中的 RL 环境壳层，只通过 `HeadlessGame`
  访问游戏；在 worker 内缓存前后 `StateAnalysis` 并计算 Reward V2。
- `merge_utility()` / `compute_state_potential()` / `compute_reward()`：Reward V2
  纯函数入口，分别负责指数合成效用、C/R/K potential 和相邻状态 shaping。
- `GraphBuilder`：把无渲染游戏状态和候选动作转换成框架无关 `GraphData`，供后续 GNN/Q 网络使用。
- `GraphAblator`：训练实验用的图特征消融层，通过置零特征对比不同信息组对模型的影响。
- `graph_to_tensor()`：把 `GraphData` 转成 PyTorch 张量，形成 `node_features`、`edge_index`、`edge_features` 和 `action_node_indices`。
- `collate_graph_tensors()`：把多张 `GraphTensor` 拼成不连通 `GraphBatch`，记录每张图的 action slice。
- `GNNQNetwork`：当前 GNN-Q 前向模型，输入单图输出 `[action_count]`，输入 `GraphBatch` 输出 `[total_action_count]`。
- `TensorTransition`：正式训练主链路唯一使用的张量化经验记录，保存 CPU `GraphTensor`，图特征固定以 `float16` 存储以降低 replay 常驻内存。
- `ReplayBuffer`：固定容量经验回放池，支持纯内存和热内存 + 冷磁盘两种模式；正式大规模训练默认只让最新一部分经验常驻内存。
- `RolloutCollector`：单进程经验采集器，串联 `DaxiguaEnv`、`GraphBuilder`、Q 网络和
  `ReplayBuffer`，复用上一轮 `next_graph`，并将唯一 `TransitionKey` 交给环境完成
  Reward V2 身份对齐。
- `ParallelRolloutCollector`：多进程经验采集器，多个 worker 并行推进 headless 物理，主进程统一写 replay；可通过训练脚本的 `--async-rollout` 与 DQN 更新重叠。
- `DQNTrainer`：标准 DQN 单步更新器，使用 GraphBatch、online/target 双网络、SmoothL1Loss 和梯度裁剪更新 Q 网络，并记录采样、前向、target、反向和优化器耗时。
- `train_dqn.py`：第一版训练入口，输出 `metrics.csv`、`episode_metrics.csv`、
  `checkpoints/latest.pt`、`checkpoints/best.pt`、`plots/training_curves.png` 和
  `plots/reward_breakdown_curves.png`；Reward V2 breakdown 按日志窗口求均值，
  同时记录 shaping p95、StateAnalyzer 性能、训练 profiling 和 replay 分层状态。
- `board_game_state()` / `board_action_candidates()`：把原 pygame `Board` 的实时局面转换成 RL 图构建所需的数据结构。
- `watch_dqn.py`：第一版模型可视化观看入口，用真实游戏窗口检查 checkpoint 的实际操作效果。
- `export_training_catalog.py`：扫描被 Git 忽略的训练输出，生成配置快照、指标摘要、产物清单和跨实验索引，方便迁移后复盘训练数据。
- `compare_physics_modes.py`：物理模式对比入口，用已有 checkpoint 或随机策略比较 accurate 与 fast 模式的速度、分数、局长、物理帧、合成频率和截断率。
- `StateAnalysis`：worker 内的完整状态归因快照。15 位 mask 按 `action_offset`
  编位，保存 q0-q3 独立投放横坐标、真实/探针物理半径、自由空间区域、支撑/接触
  证据、伙伴分量和连锁 motif；不写入主 replay。
- `StateAnalyzer`：从稳定 `GameState`、15 个动作候选和 `TransitionKey` 生成
  `StateAnalysis`。它只做静态只读近似，不推进物理；已通过环境接入 Reward V2，
  尚未接入历史 `AttributionTracker`。
- `configs/train_dqn_fast30_parallel.toml`：正式 DQN 训练参数配置，集中维护 run 目录、训练规模、replay、epsilon、模型、并行采样、reward、评估保存和进度参数。
- `scripts/train_dqn.sh`：TOML 配置启动器，默认读取 `configs/train_dqn_fast30_parallel.toml`，也可以传入其它配置文件路径。
- `resize_world(width, height)`：按窗口尺寸重设 pygame 画布和 pymunk 边界。当前手动游戏窗口固定，此函数主要作为内部调试或未来实验工具保留。
- `setup_collision_handler()`：水果合成逻辑所在位置，已兼容新版 `pymunk.Space.on_collision`，并在合成后调用可选的 `on_fruit_merged()`。

## 已知注意事项

- 游戏运行时直接读取 `assets/fruits/`，不再需要手动解压资源。
- 当前手动游戏窗口固定为 `400x800`，不再通过拖动窗口边框改变场地大小。
- 顶部信息层和当前悬浮水果层已经分开；生成线固定为 `180px`，用于避免待投放队列与当前水果视野冲突。
- `daxigua` 游戏本体不得 import `daxigua_rl`；训练、环境和模型代码只通过稳定游戏接口访问游戏。
- `watch_dqn.py` 是视觉检查用入口，会在脚本内部懒加载 `daxigua.app.Board` 并打开真实 pygame 窗口；这不是训练路径，也不要求游戏本体 import RL。
- 旧的框架无关 `Transition` 已删除；正式训练主链路只保存 `TensorTransition`。
- `daxigua_rl.graph.tensor` 和 `daxigua_rl.models` 依赖 PyTorch；它们不会在 `daxigua_rl` 顶层自动导入，避免非训练环境被强制要求安装 torch。
- `RolloutCollector` 和 `DQNTrainer` 依赖 PyTorch 模型前向；它们通过 `daxigua_rl.training` 懒加载导入，不放进 `daxigua_rl` 顶层导出。
- `train_dqn.py` 依赖 PyTorch 和 matplotlib；matplotlib 使用 `Agg` 后端生成 png，并把缓存目录放到当前 run 目录下。
- `tools/temporary_rollout_smoke_test.py` 依赖 PyTorch，建议在 `python-torch` conda 环境中运行；它只做临时链路验证，不训练模型。
- 当前 `src/daxigua/core/board.py` 已为 `pymunk 7.3.0` 做兼容处理。
- 当前 `src/daxigua/app.py` 仍集中承载表现层细节；后续如确实需要拆分，再创建对应表现层模块。
