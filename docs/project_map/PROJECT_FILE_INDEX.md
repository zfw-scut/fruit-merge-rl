# 项目文件索引

最后更新：2026-07-27

## 项目定位

本项目同时包含基于 `pygame` / `pymunk` 的《合成大西瓜》桌面游戏，以及基于
PyTorch GNN-Q 的无渲染强化学习训练链路。旧实验代码和旧环境封装已经删除，当前
`daxigua_rl` 是重新设计后的 Reward V2、Double DQN + 3-step、完整状态归因、
预算反事实和局部 Shapley 实现。游戏本体通过稳定接口供训练侧使用，始终不反向依赖
自动化代码。

## 核心源码

| 路径 | 作用 | 主要入口或可复用点 |
| --- | --- | --- |
| `Main.py` | 兼容旧启动方式的薄入口，将 `src/` 加入 import 路径后调用 `daxigua.app.main()`。 | `main()` |
| `src/daxigua/app.py` | 游戏应用入口和当前表现层实现。负责固定窗口、输入、正式渲染、鼠标跟随投放、预览线、顶部独立信息层、待投放水果队列、HUD、粒子、飘字、震动和音效反馈。 | `Board.next_frame()`、`Board.run()`、`main()` |
| `src/daxigua/config.py` | 项目路径和基础配置。 | `PROJECT_ROOT`、`FRUIT_ASSET_DIR`、`DEFAULT_WINDOW_SIZE`、`SPAWN_LINE_Y`、`FPS` |
| `src/daxigua/core/board.py` | 游戏公共逻辑。负责 pygame 画布、pymunk 物理世界、动态墙体、碰撞合成、计分、失败检测，并向表现层暴露合成事件钩子。 | `GameBoard`、`resize_world()`、`create_ball()`、`setup_collision_handler()`、`check_fail()` |
| `src/daxigua/core/engine.py` | 无渲染游戏引擎。除投放、队列和稳定推进外，还能在稳定边界捕获、校验和恢复完整 Pymunk 状态，并审计原动作重演。 | `HeadlessGame`、`capture_snapshot()`、`restore_snapshot()`、`replay_and_compare_original_action()` |
| `src/daxigua/core/fruit.py` | 水果显示精灵和贴图加载。根据等级创建单一 `Fruit` 显示对象，并复用 `rules.py` 中的半径规则。 | `create_fruit(level, x, y)`、`Fruit`、`fruit_image_path()`、`load_fruit_image()` |
| `src/daxigua/core/rules.py` | 纯规则常量和辅助函数。集中维护水果半径、队列长度、随机生成范围、合成分数和物理半径。 | `FRUIT_RADII`、`FRUIT_QUEUE_LENGTH`、`fruit_radius()`、`merge_score()` |
| `src/daxigua/core/state.py` | 训练友好的纯数据结构；水果和动作候选同时公开显示/碰撞半径，并定义版本化、带 checksum 的引擎快照及动作重演结果。 | `GameState`、`FruitState`、`ActionCandidate`、`EngineSnapshot`、`EngineActionOutcome`、`OriginalActionReplayReport` |
| `src/daxigua_rl/` | 自动游玩/RL 相关代码。游戏本体不得 import 它。训练主链路通过 `HeadlessGame` 访问游戏；观看脚本可在 RL 侧懒加载真实 `Board`。 | `DaxiguaEnv`、`DaxiguaEnvConfig`、`README.md` 中记录边界规则 |
| `src/daxigua_rl/env.py` | 类 Gymnasium 的 RL 环境壳层。一次 step 表示一次投放和无渲染物理稳定；在 worker 内缓存相邻 `StateAnalysis`、接收 collector 的 `TransitionKey` 并计算 Reward V2；可从兼容 `EngineSnapshot` 创建 live 分支。 | `DaxiguaEnv.reset()`、`DaxiguaEnv.step(...)`、`DaxiguaEnv.from_snapshot()` |
| `src/daxigua_rl/reward.py` | Reward V2 纯计算层。按合成等级计算指数 task utility，并以相邻 `StateAnalysis` 的 C/R/K potential 差完成 shaping。 | `RewardConfig`、`RewardBreakdown`、`merge_utility()`、`compute_state_potential()`、`compute_reward()` |
| `src/daxigua_rl/playable_adapter.py` | 真实 pygame 游戏窗口到 RL 输入结构的适配层。把正在运行的 `Board` 转成 `GameState` 和 `ActionCandidate`，用于观看模型实际游玩。 | `board_game_state()`、`board_action_candidates()` |
| `src/daxigua_rl/attribution/` | 完整状态归因模块。除只读状态/事件契约、静态分析器和历史归因器外，还包含独立因果 replay、反事实 proposal/任务/执行器与局部 Shapley。完整分析对象不进入主 TD replay。 | `StateAnalyzer`、`AttributionTracker`、`CausalReplayBuffer`、`CounterfactualProposalBuilder` |
| `src/daxigua_rl/attribution/schema.py` | 固定 15 动作状态分析结构、时间语义、队列槽位、自由空间、谱系、贡献者、事件预算和 pending 结果不变量；全部类型可 pickle/Windows spawn。 | `StateAnalysis`、`AttributionEvent`、`AttributionStepResult`、`FruitLineageRecord`、`MergeLineageRecord` |
| `src/daxigua_rl/attribution/state_analyzer.py` | 在动作前边界执行只读静态分析：解析圆形竖直列计算 15 动作可达性/队列容量，规范最小水果探针栅格计算顶部连通空间和空腔，并构建支撑、伙伴与基础连锁结构。 | `StateAnalyzer`、`StateAnalyzerConfig` |
| `src/daxigua_rl/attribution/tracker.py` | worker-local 因果归因状态机。按真实 drop/ordered merge 建立谱系和唯一价值包，追踪铺垫兑现、渐进通道损失、pending 封路/埋死、终局确认与 reset/truncated/shutdown 中断。 | `AttributionTracker`、`AttributionTrackerConfig`、`TrackerTransitionInput` |
| `src/daxigua_rl/attribution/causal_replay.py` | 与主 TD replay 隔离的稀疏因果回放。把 confirmed 事件与原始图上下文转成正铺垫/负封路规则动作对，按类别公平采样，并保存可精确恢复的版本化状态。 | `CausalSample`、`CausalReplayBuffer`、`RuleCausalSampleBuilder`、`CausalTransitionContext` |
| `src/daxigua_rl/attribution/counterfactual_proposal.py` | worker-local 反事实候选构建层。维护稳定边界快照环，跨步关联延迟事件、真实结果和原始动作轨迹，选择有限替代动作与 2～4 个 Shapley 候选。 | `CounterfactualHistoryRing`、`CounterfactualProposal`、`CounterfactualProposalBuilder` |
| `src/daxigua_rl/attribution/counterfactual.py` | 反事实的冻结配置、target policy payload、任务/结果数据契约、稳定 ID 和 token 预算调度器。 | `CounterfactualConfig`、`CounterfactualTask`、`CounterfactualResult`、`LocalShapleyConfig`、`BudgetedCounterfactualScheduler` |
| `src/daxigua_rl/attribution/counterfactual_runner.py` | 在独立 CPU 进程中恢复 `EngineSnapshot`，先验证 factual branch，再运行有限替代动作、Reward V2 回报和冻结 target bootstrap；只有可复现结果才能转成因果样本。 | `freeze_target_policy_payload()`、`run_counterfactual_task()`、`counterfactual_result_to_causal_samples()` |
| `src/daxigua_rl/attribution/local_shapley_runner.py` | 对极少数高价值协同事件运行局部物理 Shapley。按实际轨迹逐步检查 grand coalition，缓存 2～4 个候选的 subset，使用配对排列并执行效率残差门禁。 | `LocalShapleyTask`、`LocalShapleyResult`、`run_local_shapley_task()`、`local_shapley_result_to_causal_samples()` |
| `src/daxigua_rl/graph/` | GNN 图构建相关代码。负责把游戏状态和动作候选转换成模型输入图，并提供训练实验用的特征消融层。 | `GraphBuilder`、`GraphAblator` |
| `src/daxigua_rl/graph/schema.py` | 框架无关的图数据结构和节点/边特征名。 | `GraphData`、`GraphNodeRef`、`GraphEdgeRef`、`NODE_FEATURE_NAMES`、`EDGE_FEATURE_NAMES` |
| `src/daxigua_rl/graph/builder.py` | 从 `GameState` 和 `ActionCandidate` 构建 GNN 输入图；几何、接触和投放路径关系使用真实碰撞半径，图维度保持不变。 | `GraphBuilder.build()` |
| `src/daxigua_rl/graph/ablation.py` | 图特征消融工具。在不改变图维度的前提下按配置置零部分节点或边特征。 | `GraphAblator`、`FeatureAblationConfig`、`FeatureMask`、`ABLATION_PRESETS` |
| `src/daxigua_rl/graph/tensor.py` | PyTorch 张量转换层。把框架无关 `GraphData` 转成单图 `GraphTensor`，并把多张图拼成不连通 `GraphBatch`。 | `graph_to_tensor()`、`collate_graph_tensors()`、`GraphTensor`、`GraphBatch` |
| `src/daxigua_rl/models/` | 强化学习模型代码。当前只包含最小 GNN-Q 前向模型，不包含训练循环。 | `GNNQNetwork` |
| `src/daxigua_rl/models/gnn_q.py` | 统一图 message passing Q 网络。输入 `GraphData`、`GraphTensor` 或 `GraphBatch`，输出单图或批量扁平动作 Q 值。 | `GNNQNetwork.forward()`、`MessagePassingLayer` |
| `src/daxigua_rl/training/` | 强化学习训练侧组件。包含 n-step 张量经验、冷热 TD replay、因果 replay 接入、单/多进程采集、Double DQN 更新、预算反事实/局部 Shapley 协调和版本化 checkpoint。 | `TensorTransition`、`ReplayBuffer`、`NStepTransitionAccumulator`、`DQNTrainer`、`CounterfactualCoordinator` |
| `src/daxigua_rl/training/identity.py` | 定义一次训练 run 内稳定、可哈希、可跨进程序列化的轨迹身份。 | `TransitionKey(worker_id, episode_id, step_index)` |
| `src/daxigua_rl/training/tensor_transition.py` | DQN 张量化经验记录。保存 CPU `GraphTensor` 和实际 `bootstrap_steps`，支持 1～3 步 episode 尾部及 `gamma**bootstrap_steps`。 | `TensorTransition` |
| `src/daxigua_rl/training/n_step.py` | 每 worker 独立的 n-step 累加器。正式训练聚合 3-step reward，在 terminated/truncated 或显式 flush 时输出自然缩短的尾部。 | `NStepTransitionAccumulator` |
| `src/daxigua_rl/training/replay_buffer.py` | DQN 固定容量经验回放池。支持纯内存和热内存 + 冷磁盘；checkpoint 在内存模式精确恢复，在 hybrid 模式有界保存热层并明确报告 omitted cold count。 | `ReplayBuffer`、`checkpoint_state_dict()`、`load_checkpoint_state_dict()` |
| `src/daxigua_rl/training/collector.py` | 单进程 rollout 采集器。串联 Reward V2、StateAnalyzer、AttributionTracker、规则因果样本、3-step return、32 边界快照环和反事实 proposal，完整分析仍留在 worker。 | `RolloutCollector`、`EpsilonGreedyPolicy`、`RolloutStats` |
| `src/daxigua_rl/training/parallel_collector.py` | Windows spawn 多进程 rollout 调度器。各 worker 独立持有环境、归因器、n-step 和快照环；主进程回收 transition、因果样本、proposal 与轻量聚合统计，worker 内 PyTorch 限为单线程。 | `ParallelRolloutCollector`、`WorkerAttributionFinalization` |
| `src/daxigua_rl/training/dqn.py` | Double DQN + n-step 更新器。online 网络选择 bootstrap 动作、target 网络估值，同时联合 TD、规则排序和反事实/局部 Shapley Huber loss；非有限值在 optimizer 前 fail-fast。 | `DQNTrainer`、`DQNTrainerConfig`、`DQNTrainStats` |
| `src/daxigua_rl/training/counterfactual_coordinator.py` | 主进程反事实协调器。冻结 target payload、登记真实步、执行软/硬 token 预算、调度独立 CPU runner，并把可复现结果写入因果 replay；预算接口同时供 Shapley 使用。 | `CounterfactualCoordinator`、`recommended_counterfactual_worker_count()` |
| `src/daxigua_rl/training/local_shapley_coordinator.py` | 以一个独立 worker 管理极稀疏局部 Shapley 的筛选、共享预算预留、pending 重试、结果门禁与因果样本写入。 | `LocalShapleyCoordinator`、`LocalShapleyCoordinatorStats` |
| `src/daxigua_rl/training/checkpointing.py` | 原子、版本化训练 checkpoint。维护 run manifest、规范配置指纹、Python/PyTorch/CUDA RNG、可选组件状态及严格 resume 配置校验。 | `RunManifest`、`atomic_torch_save()`、`build_training_checkpoint()`、`load_training_checkpoint()` |
| `src/daxigua_rl/scripts/` | 强化学习命令行脚本目录。用于放正式训练、评估、观看、导出等入口。 | `train_dqn.py`、`watch_dqn.py` |
| `src/daxigua_rl/scripts/train_dqn.py` | 当前完整因果训练入口。组合 Double DQN + 3-step、主/因果 replay、并行采集、预算反事实、局部 Shapley、指标、评估、原子 checkpoint 与 hot-resume；保存配置/运行指纹及 warmup/shutdown/resume/failure sidecar。 | `python -m daxigua_rl.scripts.train_dqn --config ...`、`--resume ...` |
| `src/daxigua_rl/scripts/watch_dqn.py` | DQN 可视化观看入口。加载训练 checkpoint，复用原 pygame `Board` 画面，并在 RL 侧注入自动控制器选择落点。 | `python -m daxigua_rl.scripts.watch_dqn --checkpoint ...` |
| `src/daxigua_rl/scripts/compare_physics_modes.py` | accurate/fast headless 物理模式对比工具。用于测试降低 fps、最大物理帧、稳定帧和 Pymunk 迭代次数后的速度收益与游戏分布偏移。 | `python -m daxigua_rl.scripts.compare_physics_modes --checkpoint ...`；输出 `summary.csv`、`episode_metrics.csv` 和 `plots/physics_mode_comparison.png`。 |
| `configs/` | 项目配置目录。三套首轮因果训练配置通过 `extends` 继承同一完整冻结基线。 | `train_dqn_causal_smoke_5k.toml`、`train_dqn_causal_calibration_10k.toml`、`train_dqn_causal_500k.toml` |
| `configs/train_dqn_fast30_parallel.toml` | 完整算法/环境基线：500k、fast30、8 worker、Reward V2、Double DQN 3-step、冷热 replay、规则排序、预算反事实与局部 Shapley。三套阶段配置继承它。 | `train_dqn.py --config ...` |
| `configs/train_dqn_causal_smoke_5k.toml` | 第一次完整因果训练的 5000-update 集成烟测配置；算法与物理语义不降级，只覆盖规模和日志频率。 | 运行后才可记录烟测结论。 |
| `configs/train_dqn_causal_calibration_10k.toml` | 10000-update 规模标定配置；若稀疏事件样本不足，可通过 CLI 覆盖从 update 0 另起独立 25000 校准。 | 不直接改变总步数恢复；用于校准量级，不预先代表已通过。 |
| `configs/train_dqn_causal_500k.toml` | 第一次 500000-update 大规模训练的稳定启动名，继承完整冻结基线。 | 只在 preflight、烟测和标定门禁完成后启动。 |
| `scripts/` | 项目级启动脚本目录。只放薄启动器，具体训练参数放在 `configs/`。 | `train_dqn.sh` |
| `scripts/train_dqn.sh` | DQN 训练启动器。默认读取 `configs/train_dqn_fast30_parallel.toml`，设置 `PYTHONPATH`，通过 `python-torch` conda 环境启动训练并 tee 日志。 | `./scripts/train_dqn.sh` |

## 资源和说明

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `assets/fruits/` | 水果图片资源目录，包含 `01.png` 到 `11.png`。 | 游戏运行时直接读取。 |
| `assets/fruits.zip` | 原始水果图片压缩包归档。 | 不参与运行，只作资源备份。 |
| `README.md` | 项目总入口，包含手动游戏、完整因果训练概览、preflight、三阶段配置和恢复说明。 | 不记录尚未产生的烟测结果。 |
| `requirements.txt` | 游戏与物理基础依赖。 | 固定 `pygame` 和 `pymunk`；快照重演还校验 Chipmunk 构建版本。 |
| `requirements-training.txt` | 训练侧 Python 依赖版本。 | 复用 `requirements.txt` 并固定 PyTorch、matplotlib；CUDA wheel 来源应按目标驱动选择。 |
| `LICENSE` | 开源许可证。 | Apache 2.0。 |

## 辅助工具

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `tools/cuda_stress_test.py` | 独立 PyTorch CUDA 计算压力测试脚本。只做矩阵乘法和可选显存预留，并采集 GPU、系统内存、进程内存和内核 NVIDIA/Xid 日志。 | 用于判断黑屏/Xid 是否能在脱离游戏和 RL 训练代码后复现；默认输出到 `runs/cuda_stress/<时间戳>/`。 |
| `tools/export_training_catalog.py` | 训练实验归档工具。扫描本地 `runs/`，识别 Reward V2 task/potential 与历史 Reward V1 指标，并从配置、训练指标和文件信息生成轻量实验目录。 | 不复制 checkpoint、replay 和完整指标 CSV；运行方式见 `docs/training_runs/README.md`。 |
| `tools/monitor_training_resources.py` | 训练资源旁路监控脚本。独立于训练入口，按固定间隔记录系统内存、swap、目标训练进程、NVIDIA GPU 和 GPU 计算进程。 | 用于定位长时间训练时的 OOM、显存压力、GPU 查询失败和显示栈异常；默认输出到 `runs/resource_monitor/<时间戳>/`。 |
| `tools/monitor_cgroup_memory.py` | Linux cgroup-v2 内存旁路监控。分别记录原始余量、`inactive_file` 页缓存、可回收工作集余量和 `memory.events`。 | 云容器训练时与通用资源监控并行运行；避免 checkpoint/replay 文件缓存造成假性低内存告警，同时保留真实 OOM/pressure 硬门禁。 |
| `tools/temporary_rollout_smoke_test.py` | 临时 GNN rollout 验证脚本。用于检查 `DaxiguaEnv -> GraphBuilder -> GNNQNetwork -> step()` 链路是否闭合。 | 不是正式训练入口；验证完成或正式训练脚本落地后可删除或改造。 |
| `tools/preflight_training.py` | 正式训练前只做短计算、不创建训练 run 的门禁。验证 TOML、Python/Pymunk/Chipmunk、CUDA 前后向、完整因果 optimizer step、局部 Shapley 物理重演、`EngineSnapshot` 多次确定性重演、磁盘和 CPU 余量。 | JSON 写入 `runs/preflight/latest.json`；任一 required check 失败时返回非零。 |

## 测试目录

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `tests/test_graph_batch_training.py` | GraphBatch 和张量化 DQN 训练链路测试。验证批量图前向、next_graph 缓存、分层 replay、并行 collector、Reward V2 分析统计和 DQN 更新链路。 | 使用标准库 `unittest`，在 `python-torch` 环境中运行。 |
| `tests/test_attribution_foundations.py` | 完整状态归因基础语义测试。验证稳定窗口、truncated bootstrap、真实碰撞半径、图几何和 worker/episode/step 身份键。 | 使用标准库 `unittest`，在 `python-torch` 环境中运行。 |
| `tests/test_attribution_schema.py` | `StateAnalysis` 数据契约测试。验证深只读、15 位 mask、队列槽位、跨对象引用、时间语义、pickle 和真实 Windows spawn 往返。 | 使用标准库 `unittest`，在 `python-torch` 环境中运行。 |
| `tests/test_state_analyzer.py` | `StateAnalyzer` 人工几何场景测试。验证空棋盘、解析投放列与并列 blocker、支撑方向、伙伴/merge/ladder motif、封闭空腔、输入降级和左右镜像。 | 使用标准库 `unittest`，不依赖 Pymunk 随机稳定过程。 |
| `tests/test_reward_v2.py` | Reward V2 纯公式和契约测试。验证指数合成效用、potential 权重、轨迹望远镜关系、相邻分析身份以及 terminal/truncated 差异。 | 不推进真实 Pymunk 物理。 |
| `tests/test_reward_v2_integration.py` | Reward V2 环境与采集集成测试。验证 StateAnalysis 缓存、worker/episode/step 键、terminal/truncated、统计合并及 Windows spawn 边界。 | 使用标准库 `unittest`。 |
| `tests/attribution_fixtures.py` | AttributionTracker 的确定性状态、分析、支撑、接触和 merge transition 构造器。 | 人工场景不依赖随机 Pymunk 稳定过程。 |
| `tests/test_attribution_tracker.py` | 历史归因器测试。验证单/多级谱系、价值包去重、触发证据、渐进封路、恢复/确认/终局/截断、结构兑现、pickle 和 worker/episode 隔离。 | 使用标准库 `unittest`。 |
| `tests/test_engine_snapshot.py` / `tests/test_env_snapshot.py` | 完整物理快照契约与环境恢复测试。覆盖 checksum/版本/配置拒绝、Pymunk 隐藏状态、RNG/队列/ID、原动作重演和从快照构造环境。 | 反事实标签的物理根基。 |
| `tests/test_n_step.py` / `tests/test_collector_causal_nstep.py` | 3-step return 和 collector 因果输出测试。覆盖正常窗口、terminated/truncated 尾部、bootstrap horizon、规则样本与有界 proposal payload。 | 使用标准库 `unittest`。 |
| `tests/test_causal_replay.py` / `tests/test_causal_training.py` | `CausalReplayBuffer`、规则样本生成和联合 loss 测试。覆盖分层采样、去重、checkpoint、Double DQN、`gamma**steps`、规则/反事实梯度及非有限值 fail-fast。 | 主 TD replay 保持隔离。 |
| `tests/test_counterfactual.py` / `tests/test_counterfactual_runner.py` | 反事实任务契约、预算账本和物理 runner 测试。覆盖稳定 ID、原动作复现门禁、有限替代分支、target bootstrap、无效结果不生成伪标签。 | 使用标准库 `unittest`。 |
| `tests/test_counterfactual_proposal.py` / `tests/test_counterfactual_coordinator.py` | 延迟事件 proposal 与主进程协调测试。覆盖 32 边界环、跨步候选、共享 token、队列/熔断/关闭和因果 replay 写入。 | 任务不得阻塞 rollout。 |
| `tests/test_local_shapley_runner.py` / `tests/test_local_shapley_coordinator.py` | 极稀疏局部 Shapley 测试。覆盖 subset/配对排列、grand coalition 重现、效率残差、0.05% 选择上限、共享预算与 pending 重试。 | 失败路径不会生成伪标签。 |
| `tests/test_checkpointing.py` | 版本化训练 checkpoint 测试。覆盖原子替换、配置指纹、RNG、manifest、可选组件恢复、semantic drift 拒绝和 inference 兼容提取。 | 只加载可信 PyTorch checkpoint。 |
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
| `docs/rl/CAUSAL_ATTRIBUTION_V1.md` | 第一次大规模训练的完整状态归因 V1 规格。 | 固定 Reward V2、状态分析、归因事件、因果 Q 排序、反事实预算、局部 Shapley、测试和长训流程；实现步骤 1～11 已落地，烟测/标定/正式长训结果仍须按实际运行记录。 |
| `docs/rl/FIRST_500K_RUNBOOK.md` | 第一次 500k 完整因果训练的执行手册。 | 固定 Ubuntu/CUDA 安装、全量测试、preflight、三阶段命令、监控/停止阈值、恢复语义和归档顺序。 |
| `docs/rl/gnn_daxigua_design_reference.md` | GNN 状态图节点、边和特征语义参考。 | 当前 `radius` 节点特征和相关几何边特征均表示真实碰撞半径。 |
| `docs/rl/INTERFACE_V0.md` | RL v0 接口说明。 | 记录 `HeadlessGame`、`DaxiguaEnv`、状态数据和边界规则。 |
| `docs/rl/TRAINING_SPEED_OPTIMIZATION_PLAN.md` | 训练速度优化计划。 | 记录 profiling、next_graph 缓存、并行采样、fast physics、图构建优化和日志频率等优化顺序。 |
| `docs/training_runs/FIRST_500K_READINESS.md` | 第一次 500k 的阶段证据与批准清单。 | 只填写实际测试/preflight/run 产物；明确区分已通过、运行中、待运行和未获准。 |

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
- `HeadlessGame`：训练环境使用的无渲染游戏接口；稳定边界可以
  `capture_snapshot()`，并用 checksum、物理版本和配置指纹保护恢复与原动作重演。
- `DaxiguaEnv`：隔离在 `daxigua_rl` 中的 RL 环境壳层，只通过 `HeadlessGame`
  访问游戏；在 worker 内缓存前后 `StateAnalysis` 并计算 Reward V2。
- `merge_utility()` / `compute_state_potential()` / `compute_reward()`：Reward V2
  纯函数入口，分别负责指数合成效用、C/R/K potential 和相邻状态 shaping。
- `GraphBuilder`：把无渲染游戏状态和候选动作转换成框架无关 `GraphData`，供后续 GNN/Q 网络使用。
- `GraphAblator`：训练实验用的图特征消融层，通过置零特征对比不同信息组对模型的影响。
- `graph_to_tensor()`：把 `GraphData` 转成 PyTorch 张量，形成 `node_features`、`edge_index`、`edge_features` 和 `action_node_indices`。
- `collate_graph_tensors()`：把多张 `GraphTensor` 拼成不连通 `GraphBatch`，记录每张图的 action slice。
- `GNNQNetwork`：当前 GNN-Q 前向模型，输入单图输出 `[action_count]`，输入 `GraphBatch` 输出 `[total_action_count]`。
- `TensorTransition`：正式 TD 主链路使用的张量化经验记录，保存 CPU
  `GraphTensor`、n-step reward 和实际 `bootstrap_steps`；图特征用 `float16`
  降低 replay 常驻内存。
- `ReplayBuffer`：固定容量 TD 回放，支持纯内存和热内存 + 冷磁盘；其 checkpoint
  接口在 hybrid 模式只保存有界热层并显式报告 hot-resume。
- `NStepTransitionAccumulator`：每 worker 连续构造 3-step return，并正确 flush
  terminated/truncated 的 1～2 步尾部。
- `RolloutCollector`：单进程经验采集器，串联 `DaxiguaEnv`、`GraphBuilder`、Q 网络和
  `ReplayBuffer`，复用上一轮 `next_graph`，并将唯一 `TransitionKey` 交给环境完成
  Reward V2 身份对齐；每个真实 transition 还驱动 worker-local
  `AttributionTracker`、规则因果样本生成、n-step 聚合和反事实 proposal 构建。
- `ParallelRolloutCollector`：多进程经验采集器，多个 worker 并行推进 headless
  物理并分别维护 StateAnalyzer/AttributionTracker/n-step/快照环，主进程统一写
  replay 并排空 proposal；可通过 `--async-rollout` 与 DQN 更新重叠。
- `CausalReplayBuffer` / `RuleCausalSampleBuilder`：把 confirmed 历史事件与原状态图
  关联成独立动作对监督，分层采样且不修改主 replay。
- `CounterfactualCoordinator`：以冻结 target policy 和共享 token 账本异步执行
  有预算的物理分支，原动作不能复现时不生成标签。
- `LocalShapleyCoordinator`：仅选择配置比例内的高价值协同事件，和普通反事实共享
  10% 硬预算，并在 grand coalition / 效率检查后写入样本。
- `DQNTrainer`：Double DQN + n-step 更新器，联合 TD、规则排序和反事实/Shapley
  SmoothL1Loss，并记录因果 batch、正确率与额外耗时。
- `train_dqn.py`：完整训练入口，除 CSV、评估和曲线外，还保存反事实预算/重演、
  Shapley、主/因果 replay、运行指纹及 warmup/shutdown/resume/failure sidecar；
  checkpoint 支持严格配置校验和 hot-resume。
- `board_game_state()` / `board_action_candidates()`：把原 pygame `Board` 的实时局面转换成 RL 图构建所需的数据结构。
- `watch_dqn.py`：模型可视化观看入口，用真实游戏窗口检查 checkpoint 的实际操作效果。
- `export_training_catalog.py`：扫描被 Git 忽略的训练输出，生成配置快照、指标摘要、产物清单和跨实验索引，方便迁移后复盘训练数据。
- `compare_physics_modes.py`：物理模式对比入口，用已有 checkpoint 或随机策略比较 accurate 与 fast 模式的速度、分数、局长、物理帧、合成频率和截断率。
- `StateAnalysis`：worker 内的完整状态归因快照。15 位 mask 按 `action_offset`
  编位，保存 q0-q3 独立投放横坐标、真实/探针物理半径、自由空间区域、支撑/接触
  证据、伙伴分量和连锁 motif；不写入主 replay。
- `StateAnalyzer`：从稳定 `GameState`、15 个动作候选和 `TransitionKey` 生成
  `StateAnalysis`。它只做静态只读近似，不推进物理；已通过环境接入 Reward V2，
  并把相邻分析交给历史 `AttributionTracker`。
- `AttributionTracker`：每个 rollout worker 独立维护的历史归因器。它建立完整水果
  谱系、每个真实合成的唯一价值包、铺垫结构来源和 pending 负事件，并在后续真实
  合成、恢复、终局、truncated/reset/shutdown 时完成兑现或收口；完整对象不进入
  `TensorTransition`。
- `configs/train_dqn_fast30_parallel.toml`：Reward V2、Double DQN 3-step、
  CausalReplay、预算反事实、局部 Shapley 和运行资源的完整冻结基线。
- `configs/train_dqn_causal_smoke_5k.toml` /
  `train_dqn_causal_calibration_10k.toml` / `train_dqn_causal_500k.toml`：
  依次承载烟测、标定和第一次正式大规模训练，不在阶段之间静默改变算法语义。
- `tools/preflight_training.py`：正式启动前重复运行的短门禁；其 JSON 结果是运行证据，
  配置文件存在本身不代表门禁、烟测或标定已通过。
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
- `EngineSnapshot` 包含 Pymunk 的隐藏求解状态，恢复会严格检查
  `pymunk 7.3.0`、Chipmunk 构建版本、物理配置指纹和 checksum；不能把不兼容环境
  中的快照用于反事实标签。
- 版本化 checkpoint 只能从可信来源加载。resume 会拒绝改变训练语义的配置漂移；
  hybrid TD replay 是明确的 hot-resume，冷段不被悄悄声明为已经精确恢复。
- 普通反事实与局部 Shapley 共用 token 硬账本。队列满、预算不足、factual 重演失败或
  Shapley 效率检查失败时，正确行为是丢弃任务并记指标，不能伪造标签或阻塞 rollout。
- 三套因果训练 TOML 已经就位，但其存在不代表 5k 烟测、10k/25k 标定或 500k 正式
  训练已经完成；训练结论只以实际 run 产物和 `docs/training_runs/` 归档为准。
- `tools/temporary_rollout_smoke_test.py` 依赖 PyTorch，建议在 `python-torch` conda 环境中运行；它只做临时链路验证，不训练模型。
- 当前 `src/daxigua/core/board.py` 已为 `pymunk 7.3.0` 做兼容处理。
- 当前 `src/daxigua/app.py` 仍集中承载表现层细节；后续如确实需要拆分，再创建对应表现层模块。
