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

fast 候选模式：

```text
fps = 15 / 30 / 45
max_physics_frames = 200
stable_frames = 6
space.iterations = 8
```

其中 `fast_30` 是当前倾向的训练候选配置，`fast_15` 和 `fast_45` 用于观察速度和行为偏移边界。

## 优化优先级

1. profiling 日志：记录采集、构图、物理、训练、评估、绘图等阶段耗时，避免凭感觉优化。
2. next_graph 缓存：非终止 transition 的 `next_graph` 可在下一轮作为当前图复用，减少重复构图。
3. 并行采样：多个 headless 环境独立采集经验，主进程集中训练，让 CPU 多核和 GPU 更充分工作。
4. fast physics：降低 `max_physics_frames`、`stable_frames`、`space.iterations` 和训练物理 `fps`。
5. 图构建优化：减少 `GraphData -> GraphTensor` 的中间对象和重复 Python list/tuple 构造。
6. 降低评估、绘图、保存频率：减少非训练计算，但保留实时进度输出和 ETA。

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
