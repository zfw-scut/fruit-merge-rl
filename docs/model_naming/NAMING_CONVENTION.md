# 模型命名规范

本目录只管理模型身份和日常称呼，不替代模型评估、训练配置或run产物清单。当前模型的
短称映射见[`MODEL_REGISTRY.md`](MODEL_REGISTRY.md)。

## 1. 两层名称

每个模型使用两层名称：

- **正式ID**：用于报告、配置登记和长期引用，要求唯一、稳定、可搜索；
- **短称**：用于讨论、图表和临时工具参数，要求简短且在登记表中唯一。

既有报告、run目录、checkpoint和Git标签中的历史ID不得追溯重命名。登记短称不改变模型
权重身份，也不表示该模型成为默认方案。

## 2. 新模型正式ID

新模型使用小写kebab-case：

```text
<family>-<variant>-<stage>-r<revision>
```

只有能区分模型身份的字段才进入名称：

| 字段 | 规则 | 示例 |
| --- | --- | --- |
| `family` | 奖励或主要模型谱系 | `base`、`aux`、`sab`、`reward-v2` |
| `variant` | 关键结构或单一实验变量；无必要时省略 | `l5-fast`、`rank-active` |
| `stage` | 有效训练阶段或迁移身份 | `24m`、`128m`、`128m-b12m`、`t120-16m` |
| `revision` | 同一方案的正式训练实例，从1开始 | `r1`、`r2` |

示例：

```text
base-l5-fast-128m-r2
aux-rank-active-34m-r1
sab-128m-b12m-r1
sab-t120-16m-r1
```

`t120-16m`表示从既有权重迁移到120 FPS并训练16M；从零进行120 FPS训练时必须使用不同
variant，例如`120fps-scratch`，不得也写成`t120`。父轨迹和额外旁路预算需要同时表达时，
分别使用`128m-b12m`、`24m-b4m`。

## 3. 短称

短称使用大写ASCII字母、数字和连字符：

```text
<FAMILY><MAJOR>[-<DISTINGUISHER>]
```

当前保留的族前缀：

| 前缀 | 含义 |
| --- | --- |
| `B` | `score_v1`简洁基线 |
| `AUX` | 动作效果辅助学习谱系 |
| `SAB` | 结构化辅助旁路谱系 |
| `RV2` / `RV21` | Reward V2 / V2.1谱系 |

短称只编码日常区分所需的最少信息，例如`B5F-128`、`AUX-B4`、`SAB-T120`。训练seed、
日期、GPU、提交号、得分以及`best/final/latest`不得写入短称；这些信息属于run身份或
checkpoint选择。

短称一经用于已归档模型，不得改指向另一组权重。若同一方案产生新的正式训练实例，优先
增加`-R2`；若结构、奖励、训练物理或迁移来源改变，则创建新的区分词。

## 4. 状态和文件名

“计划中、训练中、已完成、中止”是登记状态，不是永久名称的一部分。需要口头表达时写成
“计划中的`SAB-T120-R2`”，不要创建`planned-sab-...`之类的ID。

建议文件与目录使用正式ID：

```text
configs/<formal-id>.toml
runs/<formal-id>_seed<seed>/
docs/model_evaluations/model-<formal-id>.md
```

checkpoint仍使用`best.pt`、`final.pt`和`latest.pt`；引用模型时必须同时明确正式ID或短称，
不能只说“best模型”。

## 5. 登记流程

1. 训练排期确认后，在`MODEL_REGISTRY.md`的“计划或训练中”登记正式ID、短称和来源；
2. 启动训练时补充配置与run路径，但不因状态变化改名；
3. 正式归档后补充报告、代码身份和完成状态；
4. 若训练未形成可保留checkpoint，状态标为中止，名称不回收给其他模型；
5. 正式效果仍以`docs/model_evaluations/`为准，登记表不复制或裁决效果结论。
