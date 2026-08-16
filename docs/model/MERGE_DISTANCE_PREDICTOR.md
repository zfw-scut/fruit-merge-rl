# 轻量逐水果合成步距预测器

- 状态：模型、按场景数据契约、离线标签、训练与验证入口已实现；尚未执行云端正式采集和训练
- teacher：`SAB-128 final.pt`
- 当前阶段边界：只训练并冻结独立预测器，不接入RL模型

## 1. 任务语义

对每个稳定决策状态中的活动水果，预测它在`SAB-128` greedy策略继续运行时，距离下一次
参与合成所需的投放次数区间。合成后的新水果具有新的`fruit_id`，作为新的生命周期。

默认类别为：

```text
1, 2, 3-4, 5-8, 9-16, 17-32, 33-64,
65-128, 129-256, 257-512, 513-1024, >1024后合成, 自然终局未合成
```

模型对全部类别执行一次softmax；前11个区间概率的累加即为各时间范围内的单调合成概率。
自然终局未合成是独立竞争结果，不伪装成一个很大的`T_merge`。显式投放边界、模拟器截断
和采集停止产生的未来未知水果不进入监督loss。

这里的teacher只负责定义未来动作策略和场景分布。标签来自模拟器真实合成事件，不蒸馏
`SAB-128`的Q值或隐藏表示。

## 2. 按场景数据契约

旧Merge Potential统计表只保存被抽中水果的少量因素，不能还原完整水果图。本任务使用新
数据集，每条场景记录包含：

- 固定64槽的位置、速度、角速度、等级、物理半径、年龄、活动掩码和`fruit_id`；
- q0～q3队列、危险进度和越线状态；
- `episode_id`与当前投放次数。

采集器同时保存全局合成来源事件和对局终止类型。离线标签阶段通过
`episode_id + fruit_id`关联每个槽位未来首次合成，并按同一水果在全部场景中的观察次数
生成`1 / N`生命周期权重。

训练/验证/测试严格按完整episode划分为90%/5%/5%，同一水果的不同场景不会跨集合。
训练可按等级×结果频率做温和平衡；固定验证和测试仍保留自然策略分布。

## 3. 模型结构

预测器复用`BaselineGnnDqn`现有的水果节点特征、混合物理邻居、边特征、边界消息和队列
编码语义。默认使用64隐藏维、2层物理消息和32维队列编码，不构建动作探针、Dueling Q、
多策略头或动作效果头；默认配置共123,979个可训练参数。

水果图编码后，全局sum/max、有效水果数量、队列和危险状态形成场景上下文；场景上下文
广播回每个水果节点，由共享小型读出头输出13类离散生存分布。预测器checkpoint保存完整
结构、几何、teacher数据身份、指标和权重，不依赖RL checkpoint才能加载。

当前实现没有Voronoi、Contact/Mobility/Rigidity扩展，也没有RL融合层。后续融合阶段只有
在预测精度、校准与性能通过门禁后另行设计。

## 4. 云端流程

入口统一为`tools/train_merge_distance_predictor.py`。首次部署先运行小规模性能预检：

```bash
PYTHONPATH=src python tools/train_merge_distance_predictor.py collect \
  --output-dir runs/analysis/sab128_merge_distance_pilot \
  --episodes 1024 \
  --parallel-envs 512 \
  --physics-fps 30 \
  --max-drops 0
```

采集器默认强制核对`SAB-128 final.pt` SHA-256：

```text
fc40b9019c65ecba8502f4334d1418b4f93c0e54e984d42ccc4d0b477bddca07
```

并要求当前30 FPS完整逐帧、无自由下落加速、无真实投放上限环境。正式采集前在新实例上
测试并行环境数、场景分片规模、GPU回传间隔和磁盘吞吐。

采集完成后关联标签：

```bash
PYTHONPATH=src python tools/train_merge_distance_predictor.py label \
  runs/analysis/sab128_merge_distance_pilot
```

训练独立预测器：

```bash
PYTHONPATH=src python tools/train_merge_distance_predictor.py train \
  runs/analysis/sab128_merge_distance_pilot/predictor \
  --output-dir runs/merge_distance/sab128_teacher_r1
```

训练默认使用BF16 autocast、AdamW、生命周期权重和等级×结果平方根平衡；按验证NLL保存
`best.pt`，早停后在固定测试集上生成最终指标和`final.pt`。Linux云端默认启用
`torch.compile`；本地Windows只运行小型CPU接口预检，不执行正式训练。

采集、标签和训练进度会进入Xigua Atlas的“实时训练”页。训练阶段默认每10秒原子更新
运行清单，每个epoch追加训练loss和验证指标；既有8765遥测服务只读取这些小型文件，因此
不需要为预测器单独启动面板，也不会把门户请求放进训练热路径。

独立复核命令为：

```bash
PYTHONPATH=src python tools/train_merge_distance_predictor.py evaluate \
  runs/merge_distance/sab128_teacher_r1/checkpoints/final.pt \
  runs/analysis/sab128_merge_distance_pilot/predictor \
  --split test
```

## 5. 验证指标与阶段门禁

训练产物至少报告：

- 未加权NLL和按水果生命周期加权NLL；
- 精确区间准确率与相邻区间准确率；
- 自然终局未合成Brier分数；
- 各投放范围累计合成概率的Brier分数；
- L1～L11分等级样本量、NLL和精确区间准确率。

正式归档前还要在同一云实例记录独立预测器前向耗时、显存和吞吐。当前本地验证只证明
数据关联、模型前后向、分片训练、checkpoint保存和严格加载成立，不代表预测器已经获得
有效精度，也不代表加入RL后会提高分数。
