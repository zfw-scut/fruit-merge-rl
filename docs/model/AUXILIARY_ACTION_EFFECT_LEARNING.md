# 辅助动作效果学习与策略不确定性

- 状态：训练框架与监督链路已实现；排名主动学习结构已就绪，等待下一轮正式训练验证
- 更新日期：2026-08-09
- 推荐配置：`configs/gnn_dqn_auxiliary_action_l5_24m.toml`
- 对照边界：历史模型与历史配置保持单策略头、无辅助损失，仍可加载

首轮与第二轮正式训练分别验证过6.4M和18M降至0.05的epsilon曲线。第二轮影子诊断表明
固定bonus存在量纲依赖且实际改动作率有限，因此下一轮推荐配置恢复6.4M最低点，并把主动
学习改成无bonus的排名融合。历史运行身份和历史报告保持不变。

## 1. 目标与非目标

本轮实现让模型在学习 Q 值的同时，学习“一次实际投放到下一个决策边界之间发生了
什么”。它包含 CUDA 事件采集、GPU Replay 标签、多策略头 bootstrap、辅助预测头、
分组损失、低 epsilon 主动探索和云端可视化。

本轮没有实现：未执行动作的反事实标签、行为段归因、关键决策触发器、世界模型、课程
学习、高层规划或 BMCTS。关键决策事实框架仍作为稀有样本和未来反事实任务的独立出口，
不是本次密集辅助监督的运行依赖。

## 2. 标签语义

辅助监督只对应实际执行动作。新水果事件按一次投放内的全局合成事件顺序取前三个，
不要求位于 q0 谱系；由 q0 碰撞间接触发的合成也因此可进入标签。

| 分组 | 标签 |
| --- | --- |
| 合成 | 是否合成、合成次数、`score_delta`、水果总数变化 |
| q0 谱系 | q0 是否参与、谱系深度、最终后代等级 |
| 首次接触 | 地面/左墙/右墙/水果多标签、主接触类型、位置、接触水果等级差、法向量、时间、法向接近速度 |
| 新水果 | 前三次新水果是否存在、位置、等级；按全局事件顺序，不限 q0 谱系 |
| 决策边界 | q0 最终后代是否存在及 `x/y/vx/vy/角速度` |
| 稳定与风险 | 稳定、等待超时、终止、模拟时长、危险进度变化、决策边界是否越线 |

“首次接触类型”允许同一最早物理帧内多个类型同时为真；另保存一个按法向接近速度选择
的主接触，供位置、等级差和法向量使用。无接触、非水果接触、新水果不存在、q0 最终
消失均通过显式有效掩码区分，不用全零值冒充有效标签。

## 3. 热路径

```text
actor 一次前向
→ 多策略头均值 Q + 模型分歧
→ 互斥三级分支：排名主动学习 / epsilon 随机 / 贪心
→ Replay.begin_append：复制决策前状态并固化 bootstrap mask
→ CUDA simulator.step：固定大小事件累计
→ 动作效果标签压缩
→ Replay.finish_append：状态、TD 字段、bootstrap mask、辅助标签全部留在 GPU
→ learner：各策略头 TD loss + 实际动作的辅助 loss
```

事件累计没有逐帧 Python 对象、CPU 回传或同步归档。CUDA 每个环境只维护固定大小的
首次接触与 q0 谱系摘要；已有定长合成事件数组用于提取全局前三个新水果。

## 4. 多策略头与 bootstrap

共享 GNN、队列编码和动作探针后接多个表述相同 21 动作决策的 Dueling 策略头。默认
推荐 5 个头。前向兼容输出是各头 Q 值均值，详细输出保留 `[B, H, 21]`。

每条 transition 写入 Replay 时只生成一次 Bernoulli bootstrap mask，并保证至少一个头
有效；以后反复采样该 transition 都使用同一 mask。Double DQN 的动作选择、Target 估值
和 Huber loss 按头计算，再只在该头有效样本上聚合。这样分歧来自不同训练子集，而不是
每次 update 临时随机屏蔽造成的噪声。

动作不确定性使用去除每个头动作均值后的优势偏好标准差：

```text
centered_q[h, a] = q[h, a] - mean_a(q[h, a])
uncertainty[a] = std_h(centered_q[h, a])
```

它减少不同 value 偏置对决策分歧的污染。

## 5. 主动学习动作

主动学习与epsilon随机探索是两个独立、互斥的分支。默认概率为：

```text
p_active = 0                              , epsilon >= 0.50
p_active = 0.40 * (0.50-epsilon)/0.45    , 0.05 < epsilon < 0.50
p_active = 0.40                           , epsilon <= 0.05

if u < p_active:              排名主动学习
elif u < p_active + epsilon:  epsilon均匀随机
else:                         argmax(mean_Q)
```

主动分支对21个动作分别计算`mean_Q`降序名次和策略头分歧降序名次，以“名次和、两者
较差名次、价值名次、动作下标”的顺序确定候选列表，再从前K项均匀随机选择；默认K=4。
这里没有Q与不确定性的加权相加，也没有量纲变换参数。默认最低epsilon阶段三分支分别
占40%、5%、55%。这些比例是下一轮待验证配置，不是已经证明最优的参数。

## 6. 辅助头和 loss

辅助头读取共享动作效果表示，但 learner 先按实际动作 gather，再计算监督。其他 20 个未
执行动作不会接收真实结果标签。损失按五组等权聚合后乘总权重：

- `aux_loss_merge`；
- `aux_loss_q0_lineage`；
- `aux_loss_first_contact`；
- `aux_loss_generation`；
- `aux_loss_outcome`。

分类使用交叉熵或二元交叉熵，连续量使用 Smooth L1，并按接触、新水果、最终后代等有效
掩码聚合。总训练损失为：

```text
loss = bootstrap_double_dqn_loss
     + auxiliary_loss_weight * mean(auxiliary_group_losses)
```

## 7. 配置与兼容性

历史兼容默认值是：

```toml
[model]
policy_head_count = 1
action_effect_enabled = false

[dqn]
bootstrap_probability = 0.8
auxiliary_loss_weight = 0.2
active_learning_enabled = false
```

因此旧 TOML 和旧 checkpoint 的模型参数名保持兼容。启用新配置后会增加策略头、辅助头
和 Replay 标签，属于新的模型结构，不能把它误写成与旧 checkpoint 完全相同的续训。

首轮正式24M配置实际使用5层GNN、5个策略头、`score_v1`、batch 256、1536个环境和
5组辅助监督。正式运行前的CUDA编译、短smoke、显存和吞吐门禁使配置从初稿收敛到该档；
多头和辅助Replay确实提高了learner成本。

## 8. 监控与产物

实时面板保留总/DQN/辅助损失、策略头分歧、三级动作分支比例和主动/贪心重叠率。主动
学习新增指标只保留“被选动作价值名次与不确定性名次的窗口相关系数”；不保存逐动作
排名、Q代价、分歧增益或多组影子候选。相关系数为正表示两种名次同向，负值表示当前
Top-K选择更多体现价值与不确定性的权衡；样本不足或方差为零时记为空值。

面板旁路进程每2秒采样GPU利用率和显存，保存轻量时间序列；实时面板与4×2归档曲线均
展示GPU/显存占用率。训练正常完成后写入`run_status.json`，原训练入口自动在同一端口
切换为最终只读面板，明确显示“已正常完成”并保持可访问，直到用户停止该进程。可用
`--exit-after-completion`显式关闭这一行为。切换前会释放trainer对象和CUDA缓存，面板
存活不等于训练仍在运行，也不会继续占用训练期显存。
高等级密度按“每千次投放生成数”归一化，避免不同评估局长和局数直接比较原始计数。

接口预览图使用确定性合成数据，只用于检查字段、图例和布局，不是模型效果证据：

- `docs/assets/auxiliary_action_learning/training_curves_preview.png`；
- `docs/assets/auxiliary_action_learning/dashboard_panels_preview.png`。

## 9. 验证与结论边界

已验证事实：CPU 标签链路、全局前三个新水果、bootstrap mask、多策略头均值、实际动作
辅助 loss、一次 learner 更新、CUDA 扩展编译、CUDA 首次接触与 q0 谱系，以及现有回归
测试均通过。

首轮正式训练的完整配置在同4096评估seed上，相对5层Fast基线的30/120 FPS均分分别
提高13.62%/14.86%，高等级水果密度与L11消除率也提高；详见
`../model_evaluations/model-auxiliary-action-r1.md`。这证明完整配置有效，不证明收益来自辅助
任务、策略多头或主动学习中的某一个，也不证明分歧已校准、5个头和当前loss权重最优。
