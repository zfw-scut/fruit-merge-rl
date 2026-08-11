# 模型评估知识库

> 物理身份说明：除最后登记的 `structured-128m-to-120fps-transfer-r1` 外，本页其余已归档
> 模型均在“合成水果继承来源动量”规则下训练和评估。当前代码已改为合成水果创建瞬间
> 线速度、角速度归零；新规则模型必须单列物理身份（含训练物理帧率），不得与继承动量
> 规则的表项作未标注的同分布比较。

- 状态：当前模型代际评估的统一入口
- 目标：让后续 Agent 在修改模型、奖励或训练流程前，先理解已有证据、失败表现和待验证原因
- 原始产物：保留在本机 `runs/` 或外部训练存储，不提交 checkpoint、轨迹和大体积日志

## 必读顺序

1. 本文；
2. [`COMPARISON_MATRIX.md`](COMPARISON_MATRIX.md)；
3. 准备继续开发的模型报告；
4. `../LEGACY_PROJECT_EXPERIENCE_INDEX.md`；
5. 对应的模型设计文档和本机原始 run。

## 当前报告

| 模型代号 | 训练奖励 | 代码身份 | 报告 | 当前结论 |
| --- | --- | --- | --- | --- |
| `baseline-r1` | 游戏合成分数 | `0836ffd` | [`model-baseline-r1.md`](model-baseline-r1.md) | 3层历史基线，保留为容量对照 |
| `baseline-scale-v1-l4-r1` | 游戏合成分数 | `f6edba2` | [`model-baseline-scale-v1-l4-r1.md`](model-baseline-scale-v1-l4-r1.md) | 30/120 FPS局均分提高14.10%/13.68%，保留为上一代效果基准 |
| `baseline-scale-v1-l5-epsilon-r1` | 游戏合成分数 | `56c2e3d` | [`model-baseline-scale-v1-l5-epsilon-r1.md`](model-baseline-scale-v1-l5-epsilon-r1.md) | Fast在30/120 FPS达到4268.14/3897.29，成为当前效果基准；Slow未证明延长随机探索有效 |
| `baseline-scale-v1-l5-fast-128m-r2` | 游戏合成分数，五层Fast从24M续训至128M | `fd65767` | [`model-baseline-scale-v1-l5-fast-128m-r2.md`](model-baseline-scale-v1-l5-fast-128m-r2.md) | 30/120 FPS达到5050.52/4794.81，成为绝对均分基准；高于辅助首轮但使用5.33倍transition且中位分更低 |
| `auxiliary-action-r1` | 游戏合成分数 + 实际动作辅助监督 | `feb10ec` | [`model-auxiliary-action-r1.md`](model-auxiliary-action-r1.md) | 30/120 FPS达到4849.65/4476.35，仍是24M预算效果基准；单模块因果仍待消融 |
| `auxiliary-action-epsilon18m-r2` | 同上，延长epsilon探索 + bonus影子诊断 | `0e9c1da` | [`model-auxiliary-action-epsilon18m-r2.md`](model-auxiliary-action-epsilon18m-r2.md) | 低于首轮5.42%/6.64%，但仍高于5层Fast 7.46%/7.23%；18M日程不升级为默认，下一对照优先bonus=2 |
| `auxiliary-action-rank-active-r3` | 同上，排名Top-4主动学习；24M后续训至34M | `faa1377` / `c01cedb` | [`model-auxiliary-action-rank-active-r3.md`](model-auxiliary-action-rank-active-r3.md) | 34M相对自身24M提高6.32%/5.93%，但仍低于辅助首轮7.34%/6.54%；证明追加预算有效，不证明排名主动优于旧方案 |
| `auxiliary-action-single-step-branch-r4` | 同上，24M父轨迹 + 4M隔离单步旁路样本 | `bdf04ae` | [`model-auxiliary-action-single-step-branch-r4.md`](model-auxiliary-action-single-step-branch-r4.md) | 30/120 FPS为4668.89/4347.91；低于辅助首轮3.73%/2.87%，但高于5层Fast 9.39%/11.56%，旁路框架保留但不升级为效果默认 |
| `auxiliary-action-structured-branch-128m-r5` | `score_v1`，128M父轨迹 + 12M隔离单步旁路 + 结构化首次接触头 | `2e0f836` | [`model-auxiliary-action-structured-branch-128m-r5.md`](model-auxiliary-action-structured-branch-128m-r5.md) | **零初速度新物理、30 FPS训练**；30/120 FPS局均分7161.42/6551.90，完整来源run与最终索引已归档，作为120 FPS迁移起点保留 |
| `structured-128m-to-120fps-transfer-r1` | `score_v1`，128M结构化旁路 → 120 FPS weights-only物理域迁移 | `4dd76b4` | [`model-structured-128m-to-120fps-transfer-r1.md`](model-structured-128m-to-120fps-transfer-r1.md) | **零初速度新物理、120 FPS训练**；30/120 FPS局均分8006.41/7568.18，L11消除64.04%/59.52%，同seed显著高于来源，单独登记 |
| `reward-v2-r1` | 纯可投放空间 | `model-reward-v2-r1` / `8235ef9` | [`model-reward-v2-r1.md`](model-reward-v2-r1.md) | 吞吐门禁通过，但游戏效果明显低于基线，不能直接替代 |
| `reward-v2.1-r1` | 状态相关无合成参考空间 | `4c8ce18` | [`model-reward-v2-1-r1.md`](model-reward-v2-1-r1.md) | 奖励偏正已修复，但只小幅超过V2且仍低于基线；墙边投放增加属于常见策略，不能单独判为缺陷 |

新报告从 [`TEMPLATE.md`](TEMPLATE.md) 复制结构，不得删除证据边界、原因假设或反证实验栏目。

## 证据标签

- `【事实】`：可由已保存的配置、指标、checkpoint 清单或代码直接复核；必须给出路径、样本数或提交身份。
- `【观察】`：由场景实验室或回放重复看到的行为；必须说明场景和重复次数。未保存场景时不能升级为事实。
- `【假设】`：对原因的解释；必须标注置信度、支持证据、反证条件和下一项实验。
- `【决策】`：已经据证据确认的保留、停止或下一轮实验边界。

同一段不得把事实和假设混写。训练链路可运行、loss 下降或吞吐达标，均不能单独证明策略变好。

## 跨代比较规则

- 首要效果指标使用真实游戏分数、投放数、合成数、最高等级和终局水果数；不同奖励版本的 episode reward 不直接比较。
- 30 FPS 和 120 FPS 分开报告；训练效果与物理帧率迁移效果不能混成一个均值。
- 记录评估 seed base、episode 数、动作数、地图、代码提交和实际训练 transition 数。
- 不同模型若训练量、评估种子或终止规则不一致，必须注明限制；不能只报告百分比提升。
- 原因判断优先使用成对消融。只有单次训练时，结论写成候选假设，不宣称因果关系。

## 每轮更新要求

正式训练或具有决策价值的评估结束后，在关闭本轮开发前完成：

1. 把原始配置、指标、曲线、评估结果、资源数据和必要 checkpoint 迁回 `runs/<run_id>/`；
2. 核对 `run_identity.json` 与 `artifact_manifest.json`；
3. 新增或更新一份模型报告；
4. 更新 `COMPARISON_MATRIX.md`；
5. 把人工场景观察保存为可复现的场景与模型动作记录，否则标记为证据缺口；
6. 只提交轻量报告，不提交 `runs/`、模型权重或大体积轨迹。

## 当前证据缺口

- 两代模型只有单个训练 seed，不能估计训练过程的跨 seed 方差；
- 场景实验室人工观察尚未形成固定场景集和结构化结果；
- 当前没有同训练量、同评估频率的完整 `score_v1` / `spatial_v2` 成对长训；
- 辅助动作run已保存带q0谱系的真实标签；但尚未建立跨run固定标签集和离线预测校准评估；
- Reward V2.1已完成正式训练和按阶段/动作/水果等级的离线统计，但固定场景动作排序集仍未建立；
- V2.1只完成一个训练seed，约1%的相对V2增益尚未验证跨seed可复现性；
- 3层、4层和5层扩容模型的batch、环境数、seed和实际训练量不同，尚无严格同配置层数消融；
- 5层Fast/Slow已完成同seed、同2400万transition的epsilon成对对照，Fast最终效果更好；
  仍需第二个训练seed判断该结论能否跨训练随机性复现；
- 辅助动作完整配置已在同一组4096评估seed上显著超过5层Fast，但训练seed不同，且多策略头、
  辅助监督、主动探索和batch同时变化；仍需第二训练seed与最小模块消融；
- 同训练seed的辅助动作6.4M/18M epsilon对照中，18M版本在双帧率和L11事件上均下降；
  当前不采用18M为默认，但仍需第二训练seed判断该差异能否跨训练随机性复现；
- 排名Top-4主动学习24M结果偏低，续训到34M后双帧率均提高，但只有一个训练seed且续训
  重建Replay；尚无“排名主动40%对主动关闭0%”的同checkpoint后期消融；
- 单步旁路24M+4M已完成，但与辅助首轮同时改变父主动选择、环境数、旁路Replay和联合loss；
  尚无同父checkpoint的旁路loss 0%/20%成对消融，不能量化旁路数据的单独贡献；
- 结构化辅助旁路128M+12M及其120 FPS适应模型已经完整归档，并完成同4096 seed成对比较；
  迁移后双帧率均明显提高，但两个阶段都只有一个训练seed，且尚无固定离线标签集上的
  各辅助预测头准确率与校准报告，不能把净提升归因于旁路、结构化头或训练帧率中的单项；
- 五层Fast从24M续训至128M后双帧率均分显著提高并超过辅助首轮，但只有一个训练seed，
  恢复时Replay重新预热；它使用5.33倍transition且中位分仍低于辅助首轮，不能据此宣称
  简洁基线具有更高样本效率或辅助动作无效；
- 训练完成事件与最后一次资源采样仍发生在内部产物清单之后，使`monitoring.jsonl`和
  `resources.jsonl`比内部清单多一小段；外层迁移归档哈希已独立核对，不影响迁移完整性。
