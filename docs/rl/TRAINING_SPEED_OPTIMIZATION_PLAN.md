# 训练速度优化计划

## 目标

当前优先目标是提高 DQN/GNN 训练吞吐，减少大规模训练等待时间。优化时需要区分两类收益：

- 每秒能完成更多次水果投放，即 `env_steps_per_second` 提升。
- 用更少样本学到更好策略，即样本效率提升。

本文先记录工程吞吐优化方案，不改变 reward 和算法目标。

## 已确认方向

- 放弃等比例缩小游戏场地和水果尺寸。该方案不会减少刚体数量、碰撞关系和物理步数，收益不稳定，而且会带来物理迁移风险。
- 保留 accurate 和 fast 两类物理模式。
- fast 模式使用更激进的 headless 物理参数做大规模训练或参数搜索。
- 正式采用 fast 模式前，先用同一模型或随机策略对比游戏分布偏移。

## 模式定义

accurate 模式默认使用当前项目真实 headless 配置：

```text
fps = 当前项目 FPS
max_physics_frames = 720
stable_frames = 15
space.iterations = 32
```

fast 候选模式历史对比范围：

```text
fps = 15 / 30 / 45
max_physics_frames = 200
stable_frames = 6
space.iterations = 8
```

经过对比后，当前大规模训练默认建议使用 `fast30`：

```text
fps = 30
max_physics_frames = 240
stable_frames = 6
space.iterations = 8
```

`max_physics_frames` 从对比脚本候选的 200 放宽到 240，用一点上限换取更稳的落稳余量。

## 优化优先级

1. 已完成：profiling 日志。记录采集、构图、张量转换、动作选择、环境 step、训练、采样、前向、target、反向、优化器、评估、保存、绘图等阶段耗时。
2. 已完成：next_graph 缓存。非终止 transition 的 `next_graph` 在下一轮作为当前图复用，减少重复构图。
3. 已完成：并行采样。多个 headless 环境在独立 worker 进程中采集经验，主进程集中写 replay 和训练。
4. 已完成：fast physics。训练脚本新增 `--physics-mode fast30`，自动使用当前确认的 30fps 快速物理参数。
5. 已完成：图构建轻量优化。`GraphBuilder` 缓存特征列索引，减少每个节点/边构造完整 dict 的开销。
6. 已完成：降低评估、绘图、保存频率。默认周期改为更适合长训的低频设置，同时保留 3 秒一次实时进度和 ETA。
7. 已完成：ReplayBuffer 内存优化。默认大容量训练使用热内存 + 冷磁盘，并在冷段写入时对共享 `GraphTensor` 做段内去重。
8. 已完成：异步采样入口。`--async-rollout` 可以让下一批并行采样在当前 DQN 参数更新时提前运行。

## 当前新增工具

`src/daxigua_rl/scripts/compare_physics_modes.py` 用于比较 accurate 与 fast 模式：

- 支持加载已有 checkpoint 做 greedy 策略对比。
- 不提供 checkpoint 时使用随机策略做物理分布基准。
- 输出单局明细、模式汇总和对比图。

输出文件：

```text
runs/physics_mode_compare_*/episode_metrics.csv
runs/physics_mode_compare_*/summary.csv
runs/physics_mode_compare_*/plots/physics_mode_comparison.png
```

## 判断标准

fast 模式是否可用于训练，不能只看速度，还要看分布是否明显偏移：

- 平均分是否大幅变化。
- 平均局长是否大幅变化。
- 平均每步物理帧是否下降。
- 合成频率是否异常。
- 物理截断率是否明显升高。
- 最终堆叠高度和水果数量是否明显不同。

如果 `fast_30` 速度提升明显，且分数、局长、合成频率、截断率没有明显异常，可以作为大规模训练默认候选；否则仅用于前期快速试验或预训练。

## 训练入口相关参数

`train_dqn.py` 当前与性能直接相关的参数：

- `--physics-mode fast30`: 使用 30fps 快速 headless 物理参数。
- `--num-envs N`: 开启 N 个并行采样 worker；建议先从 CPU 物理核心数量的一半到相等范围试起。
- `--collect-per-update M`: 每次 DQN 更新前采集多少次投放；并行采样时建议设为 `num_envs` 的整数倍。
- `--async-rollout`: 并行采样时提前提交下一批采样，使采样与训练尽量重叠。
- `--hot-replay-capacity`: 常驻内存的新经验数量；默认 `min(10000, replay_capacity)`。
- `--replay-cold-dir`: 冷 replay 磁盘目录；默认 `run_dir/replay_cold`。
- `--replay-cold-sample-ratio`: 每个 batch 期望来自冷 replay 的比例，默认 `0.25`。
- `--progress-interval`: 实时进度心跳间隔，默认 3 秒。

长训推荐直接使用 TOML 配置启动：

```bash
./scripts/train_dqn.sh
```

默认配置文件是：

```text
configs/train_dqn_fast30_parallel.toml
```

如果内存仍然紧张，优先降低 `hot_replay_capacity` 和 `batch_size`；如果 CPU 仍然是瓶颈，再调大 `num_envs` 或 `collect_per_update`。

临时覆盖少量参数时，可以在启动命令后追加普通命令行参数：

```bash
./scripts/train_dqn.sh configs/train_dqn_fast30_parallel.toml --total-updates 200000
```
