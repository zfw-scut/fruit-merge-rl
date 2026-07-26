# 新增 StateAnalyzer 状态分析器

## 修改原因

`StateAnalysis` 已经固定动作下标、状态时点和只读数据边界，但此前还没有从真实
`GameState` 生成这些字段的算法。Reward V2 和后续 `AttributionTracker` 都需要先得到
一致的投放容量、水果可达性、封路对象、自由空间和局部合成结构，因此本次完成实现顺序
第 4 步的 worker-local 静态分析器。

## 主要修改

- 新增 `StateAnalyzer` 和可哈希配置指纹，从动作前 `GameState`、15 个动作候选及
  `TransitionKey` 生成完整只读 `StateAnalysis`。
- q0～q3 分别按自身直接投放半径和合法横坐标运行解析圆形竖直投放列，输出第一落点、
  安全动作、blocker 和 Reward V2 顶部连通容量。
- 同一等级的 15 条有序投放列只扫描一次，再为各目标水果派生 blocker；避免按每颗
  水果重复全场扫描，降低后续 worker-local 集成成本。
- 每颗水果使用同级直接投放探针计算 15 位可达 mask、逐动作 blocker、关键 blocker、
  埋藏深度和高级水果倒置阻挡。
- 使用等级 1 的直接投放半径建立规范自由空间探针栅格，通过四邻域 BFS 区分顶部连通
  区域与封闭空腔，并保存面积、质心、包围盒、边界水果和墙/地板接触。
- 区域 ID 只保证单状态内确定，不冒充跨步永久身份；后续 tracker 应组合区域几何、
  边界水果和包围盒重叠关系完成跨步匹配。
- 从静态接触关系生成地板支撑、墙约束、水果支撑、盖压和桥接边；同时建立同级伙伴
  分量、`merge_pair` / `level_ladder` motif、`recoverability` 和
  `chain_readiness`。
- 只有目标失去全部可达列时才生成最终 `critical_blocker_ids` / `caps`；只有队列
  存在兼容等级且仍有实际触发动作时才生成正 motif，避免给局部受阻或完全封死结构
  错误归因。
- 明确几何近似边界：竖直列不求解沿圆滚动、弹跳或后续碰撞重排，规范栅格只用于自由
  空间连通性；ladder 距离使用真实合成后半径但合成位置仍以 pair 中点近似；分析诊断
  会记录近似算法和不稳定边界降级。
- 保持主训练链路不变。分析器尚未接入 `RolloutCollector`、主 `ReplayBuffer`、
  Reward V2 或 `AttributionTracker`，也不会自行收集逐帧接触日志。

## 涉及文件

- `src/daxigua_rl/attribution/state_analyzer.py`: 新增状态分析配置、解析投放列、规范
  自由空间、支撑/伙伴/motif 和 potential 分量计算。
- `src/daxigua_rl/attribution/schema.py`: 增加 `FreeSpaceRegionAnalysis` 及区域引用
  校验，schema 版本升为 2。
- `src/daxigua_rl/attribution/__init__.py`: 导出分析器、配置和自由空间区域类型。
- `tests/test_state_analyzer.py`: 新增人工状态几何测试。
- `tests/test_attribution_schema.py`: 增加自由空间区域和跨对象引用不变量测试。
- `docs/rl/CAUSAL_ATTRIBUTION_V1.md`: 固定 V1 实际采用的解析列/规范栅格算法、
  近似边界和区域身份语义。
- `docs/rl/INTERFACE_V0.md`: 登记公开分析接口及 worker-local 边界。
- `docs/project_map/PROJECT_FILE_INDEX.md`: 登记源码、测试和可复用入口。
- `docs/README.md`、`src/daxigua_rl/README.md`: 更新当前实现进度。

## 验证方式

在 `python-torch` conda 环境中执行：

```text
PYTHONPATH=src python -m unittest tests.test_attribution_schema -v
PYTHONPATH=src python -m unittest tests.test_state_analyzer -v
PYTHONPATH=src python -m unittest discover -s tests -v
```

人工场景不依赖 Pymunk 的稳定过程，而是直接构造 `GameState`，以便准确验证预期的
动作 mask、并列 blocker、封闭空腔、伙伴/ladder motif、支撑方向和镜像性质。当前
schema 与 analyzer 两组共 35 项定向测试全部通过；全量 68 项测试全部通过。

## 备注

- `StateAnalyzer` 提供的是第一次大规模训练所需的可解释静态证据，不是连续空间路径
  规划器；需要动态滚动才能到达的路线可能被保守地判为不可达。
- `contact_influence_edges` 仍由后续物理采集/tracker 提供。当前分析器只负责校验并把
  已压缩的前一动作证据挂到正确状态边界。
- 本步骤产生 potential 的三个原始分量，但尚未修改环境 scalar reward。下一实现步骤
  是 Reward V2，不能把当前分析器落地误解为奖励机制已经启用。
