# 训练事实与派生监督扩展架构

- 状态：基础框架已实现；具体优化任务未实现
- 更新日期：2026-08-08
- 当前基线：5 层 Fast epsilon、`score_v1`、1-step Dueling Double DQN

## 1. 目标与边界

本架构让现有 GNN-DQN 训练以后能够接入辅助动作效果学习、反事实行为段归因、最优性能
差距课程、分层强化学习、BMCTS teacher 和异步多 GPU worker，同时保持当前训练核心可以
独立运行。

当前只实现：

- 稳定的训练决策身份；
- 不可变事实批次；
- 带代次校验的 Replay 只读引用；
- 当前已识别的状态恢复候选 sidecar；
- 可插拔的两阶段决策选择接口；
- 可选 GPU 有界事实缓冲；
- 有界异步 CPU 分片归档；
- 通用派生监督信封；
- 现有 `BaselineTrainer` 接线和监控字段。

当前明确不实现：

- 关键决策触发条件；
- 行为段边界算法；
- 动作效果预测头和辅助 loss；
- 反事实状态恢复、动作分支和归因；
- 最优上下界求解器与课程调度；
- 高层规划器、计划类型和 latent 尺寸；
- BMCTS；
- 多 GPU 任务传输和权重同步。

这些功能只能作为独立 producer、consumer 或 selector 接入，不能反向成为当前 TD 训练的
运行依赖。

## 2. 分层结构

```text
现有训练核心
├─ CUDA TensorVectorSimulator
├─ GNN-DQN actor
├─ GpuReplayBuffer
└─ DqnLearner

训练事实扩展
├─ ActionSelectionBatch
├─ ReplayStateReference
├─ BatchDecisionSidecar
├─ KeyDecisionCollector
└─ DecisionFactBatch

可选事实出口
├─ GpuDecisionBuffer
└─ AsyncDecisionArchive

未来派生任务
├─ auxiliary action producer
├─ counterfactual producer
├─ oracle / curriculum producer
├─ hierarchical planner producer
└─ BMCTS producer
        ↓
DerivedSupervisionBatch
        ↓
未来训练 sample assembler
```

事实记录生成后不可被派生任务覆盖。反事实、搜索、课程或高层计划结果通过原始
`decision_id`、`segment_id`、`plan_id` 和 `policy_version` 追加关联。

## 3. 当前训练热路径

开启有效 selector 后，主循环的数据顺序为：

```text
observe
→ 一次 GNN-DQN 前向得到 21 维 Q
→ epsilon-greedy ActionSelectionBatch
→ Replay.begin_append
→ selector.prepare + 决策 sidecar
→ simulator.step
→ reward + Replay.finish_append
→ selector.select 固定容量候选
→ DecisionFactBatch
→ 可选 GPU buffer / 异步 archive
→ episode reset
→ learner update
```

采集器复用 actor 已计算的 Q 和 Replay 已复制的决策前模型状态，不增加模型前向。选择器
返回固定容量的 `DecisionSelectionBatch`，用 `valid_mask` 标记实际记录，避免公共契约依赖
动态 Python 对象数量。

## 4. 稳定事实身份

一个事实批次至少保存：

```text
run_id
decision_id
episode_id
segment_id（当前为 -1）
plan_id（当前为 -1）
environment_row
replay_index + replay_generation
policy_version
producer_version
information_scope
```

`decision_id` 由 run 内全局 transition 位置确定。`episode_id` 由采集器在每个环境首次看到
一局时分配，并在 reset 后失效。跨 run 身份使用 `run_id + decision_id`。

采集器有效激活时，Replay 槽位启用单调 generation。未来上下文若引用了已经覆盖的槽位，
设备端 `valid` 会变为 false，不会静默读取另一条 transition。默认关闭路径不分配
generation Tensor，也不在每次 Replay 写入时维护代次。

## 5. 事实状态范围

`DecisionFactBatch` 包含：

- 决策前和决策后的 `TensorState`；
- 实际动作、greedy 动作、探索标志和完整 21 维 Q；
- reward 与训练 stage；
- `BatchDropResult`；
- `BatchPhysicsResult` 及完整定长 `BatchMergeEvents`；
- 候选 priority、reason bits 和 valid mask；
- `BatchDecisionSidecar`。

Sidecar 只补充不应进入模型输入、但当前判断未来状态恢复可能需要的字段：

```text
angles
fruit_ids
score / last_score
step_count / physics_frame
fail_frames
next_fruit_id
rng_state
episode_count
terminated
```

位置、速度、角速度、等级、半径、年龄、活动掩码、队列和危险状态继续由主 Replay 保存，
不在 sidecar 重复常驻。当前不实现状态恢复，因此不把这组字段宣称为已证明
完备；未来实现恢复器时必须用同动作轨迹 parity 测试补充或精简字段。

## 6. 可插拔选择器

选择器使用两阶段接口：

```text
prepare(pre_context)
select(post_context) -> DecisionSelectionBatch
```

`prepare` 可以在物理 step 前读取模型状态与动作选择并生成小型设备端摘要；不得长期保留
随后会被模拟器修改的观察视图。`select` 在真实物理结果可用后返回固定数量的行、有效掩码、
优先级和原因位。

仓库只提供 `EmptyDecisionSelector`。它故意保持 inactive，表示配置框架并不等于已经定义
“什么是关键决策”。未来优化项必须单独实现并验证自己的 selector。

## 7. 双事实出口

### 7.1 GPU 出口

`GpuDecisionBuffer` 保存有界 `DecisionFactBatch`，不发生 CPU 往返。未来同进程辅助任务或
设备端任务调度器可以消费这些批次。当前 trainer 不消费它。

### 7.2 CPU 归档出口

`AsyncDecisionArchive` 使用：

- 有界任务队列；
- CUDA 专用复制 stream；
- pinned CPU 目标 Tensor；
- 后台 writer；
- 原子 `.pt.tmp` → `.pt` 重命名；
- `manifest.json`、记录数、字节数和 SHA-256；
- 最大归档字节数和队列满丢弃统计。

训练线程不执行 `torch.save`。队列满、存储达到上限或 writer 已失败时，后续任务不阻塞
训练；失败状态通过 metrics 暴露。正式 selector 接入后仍需在目标 GPU 上做成对吞吐门禁。

## 8. 派生监督契约

`DerivedSupervisionBatch` 只规定公共信封：

```text
task_type
producer_version
information_scope
decision_ids
segment_ids
plan_ids
policy_versions
valid_mask
confidence
status
payload
```

`payload` 允许任务自定义带 batch 第一维的 Tensor，因此不会提前固定局部对象数、轨迹精度、
反事实 horizon、搜索宽度、计划 latent 或最优解证明形式。具体任务成熟后再添加自己的严格
payload 子契约。

## 9. 配置与兼容性

`TrainingConfig` 新增 `[decision_data]`：

```toml
[decision_data]
enabled = false
max_candidates_per_step = 8
gpu_retention_capacity = 0
archive_enabled = true
archive_subdirectory = "analysis/decision_facts"
archive_shard_records = 1024
archive_queue_size = 8
max_archive_bytes = 0
```

现有 TOML 没有该 section 时使用默认值。默认 `enabled = false`；即使设为 true，但没有注入
有效 selector，Collector 仍保持 inactive，不创建 writer、GPU buffer 或归档目录。

未来入口通过构造参数注入：

```text
BaselineTrainer(..., decision_selector=selector, decision_sinks=(...))
```

这使辅助动作、反事实、BMCTS 或多 GPU producer 不需要进入基线 trainer 的配置解析和模型
实现。

## 10. 后续开发约束

- 辅助动作学习先消费事实 batch；关键状态的多动作标签以后作为派生监督追加；
- 客观事件行为段可以离线标注，不要求 Collector 在线完成状态机；
- 反事实任务必须携带事实 decision/segment 身份和使用的冻结 policy version；
- privileged oracle、同信息搜索和 BMCTS 必须区分 `information_scope`；
- 高层计划的结构化事实可以关联 `plan_id`，模型特定 latent 不能冒充跨版本真值；
- 单 GPU 和多 GPU 只能改变 sink/producer 的部署，不能改变事实数据语义；
- 新 selector、producer 或训练 consumer 必须可单项关闭并做吞吐与效果消融。
