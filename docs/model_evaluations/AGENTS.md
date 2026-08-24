# 模型评估时效性

默认入口是 `COMPARISON_MATRIX.md` 的相关行，不是按文件名顺序阅读全部报告。

- 当前完整逐帧18维边从零128M基线：先读
  `model-sab-full-fall-edge18-128m-b16m-r1.md`；需要120 FPS迁移分叉对照时再读
  `model-sab-full-fall-t120-16m-b2m-r2.md`。
- 当前时刻几何堵塞代理：读`model-pair-blockage-geometry-50k-r1.md`；它是独立监督模型，
  不与Policy游戏分数模型横向排序，也不再作为反事实行为段的主边界。需要理解弃用原因时
  读`../development_tracks/01_pair_risk_to_blockage/README.md`。
- 禁用快进前的零初速度谱系：先读`model-structured-128m-to-120fps-transfer-r1.md`；需要来源对照时再读`model-auxiliary-action-structured-branch-128m-r5.md`。
- 五层 Fast 128M 及更早报告属于继承动量旧物理，只在比较对应历史方案时读取。
- Reward V2/V2.1 报告属于已结束奖励实验，不是当前默认训练方案。
- 报告中的“当前最佳、下一轮、默认”冻结在报告日期；最新排序和身份以比较矩阵为准。
- 不同物理身份、训练帧率、transition 预算或 seed 不得被写成无条件同分布比较。

只有新研究方案或正式评估需要读 `README.md` 的完整证据规则；普通参数、UI、工具和 bug 修复不读本目录。
