# 当前模型命名登记表

本表是模型正式ID与日常短称的唯一映射入口。命名规则见
[`NAMING_CONVENTION.md`](NAMING_CONVENTION.md)；效果、物理边界和证据强度仍以
[`../model_evaluations/COMPARISON_MATRIX.md`](../model_evaluations/COMPARISON_MATRIX.md)及对应
报告为准。

## 1. 当前已归档模型

| 短称 | 既有正式ID或权重身份 | 关键区分 | 状态 | 评估报告 |
| --- | --- | --- | --- | --- |
| `B3` | `baseline-r1` | 3层历史分数基线，实际14.481M | 历史对照 | [`model-baseline-r1.md`](../model_evaluations/model-baseline-r1.md) |
| `B4` | `baseline-scale-v1-l4-r1` | 4层扩容，16M | 历史对照 | [`model-baseline-scale-v1-l4-r1.md`](../model_evaluations/model-baseline-scale-v1-l4-r1.md) |
| `B5F` | `baseline-scale-v1-l5-epsilon-r1`的Fast权重 | 5层，Fast epsilon，24M | 历史对照 | [`model-baseline-scale-v1-l5-epsilon-r1.md`](../model_evaluations/model-baseline-scale-v1-l5-epsilon-r1.md) |
| `B5S` | `baseline-scale-v1-l5-epsilon-r1`的Slow权重 | 5层，Slow epsilon，24M | 已结束对照 | 同上 |
| `B5F-128` | `baseline-scale-v1-l5-fast-128m-r2` | `B5F`续训到128M | 历史效果基准 | [`model-baseline-scale-v1-l5-fast-128m-r2.md`](../model_evaluations/model-baseline-scale-v1-l5-fast-128m-r2.md) |
| `AUX` | `auxiliary-action-r1` | 5层、5策略头、动作效果辅助，24M | 历史辅助基准 | [`model-auxiliary-action-r1.md`](../model_evaluations/model-auxiliary-action-r1.md) |
| `AUX-E18` | `auxiliary-action-epsilon18m-r2` | epsilon在18M降至最低，24M | 已结束对照 | [`model-auxiliary-action-epsilon18m-r2.md`](../model_evaluations/model-auxiliary-action-epsilon18m-r2.md) |
| `AUX-R24` | `auxiliary-action-rank-active-r3`的24M权重 | 排名Top-4主动学习，24M | 已结束阶段 | [`model-auxiliary-action-rank-active-r3.md`](../model_evaluations/model-auxiliary-action-rank-active-r3.md) |
| `AUX-R34` | `auxiliary-action-rank-active-r3`的34M权重 | `AUX-R24`续训到34M | 已结束对照 | 同上 |
| `AUX-B4` | `auxiliary-action-single-step-branch-r4` | 24M父轨迹加4M旁路 | 已结束对照 | [`model-auxiliary-action-single-step-branch-r4.md`](../model_evaluations/model-auxiliary-action-single-step-branch-r4.md) |
| `SAB-128` | `auxiliary-action-structured-branch-128m-r5` | 零初速度新物理；128M父轨迹加12M旁路，30 FPS训练 | 当前来源模型 | [`model-auxiliary-action-structured-branch-128m-r5.md`](../model_evaluations/model-auxiliary-action-structured-branch-128m-r5.md) |
| `SAB-T120` | `structured-128m-to-120fps-transfer-r1` | `SAB-128`权重迁移；120 FPS适应16M | 当前代表模型 | [`model-structured-128m-to-120fps-transfer-r1.md`](../model_evaluations/model-structured-128m-to-120fps-transfer-r1.md) |
| `RV2` | `reward-v2-r1` | `spatial_v2`奖励实验 | 已结束实验 | [`model-reward-v2-r1.md`](../model_evaluations/model-reward-v2-r1.md) |
| `RV21` | `reward-v2.1-r1` | `spatial_v2_1`奖励实验 | 已结束实验 | [`model-reward-v2-1-r1.md`](../model_evaluations/model-reward-v2-1-r1.md) |

`B5F/B5S`和`AUX-R24/AUX-R34`分别共享一份评估报告，但对应不同checkpoint，因而各自拥有
独立短称。表中的“当前”只表示代码与模型谱系位置，不表示跨物理身份的无条件效果排序。

## 2. 计划或训练中

当前没有已经确认正式ID的待训练模型。新训练方案只有在训练边界和来源checkpoint确认后才
登记；尚处于讨论阶段的想法不占用短称。

## 3. 使用约定

- 日常讨论优先使用短称，例如“用`SAB-T120`跑32局终局诊断”；
- 报告首次出现时写“`SAB-T120`（`structured-128m-to-120fps-transfer-r1`）”；
- 命令或工具若尚未支持短称解析，继续传入checkpoint路径，短称只用于说明身份；
- 新增、停训或归档模型时更新本表；不要在多个评估报告中维护另一套短称映射。
