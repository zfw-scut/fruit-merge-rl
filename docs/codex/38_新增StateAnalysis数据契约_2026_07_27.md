# 新增 StateAnalysis 数据契约

## 修改原因

完整状态归因的可达性、封路、埋藏和连锁分析会同时被 Reward V2、规则归因和后续
因果 replay 使用。在实现几何算法前，需要先固定只读结果结构、动作下标、状态时点和
跨进程序列化边界，避免下一阶段出现同名字段语义不同或重复真值互相矛盾。

## 主要修改

- 新建 `daxigua_rl.attribution` 包，提供八类 frozen、slots、深 tuple 数据结构：
  `StateAnalysis`、`FruitAnalysis`、`QueueLaneAnalysis`、`SupportEdge`、
  `ContactInfluenceEdge`、`PartnerComponent`、`ChainMotif` 和诊断结构。
- 动作 mask 固定按 15 个 `action_offset` 编位，并同时保存真实 `action_index` 和
  从左到右严格递增的投放横坐标。
- 固定状态时间语义：键 `t` 表示动作 `t` 前边界；接触边属于产生当前边界的
  `t-1` 动作，不能被误当成当前静态几何。
- 区分场上水果真实 `physics_radius` 与未来同级直接投放使用的
  `probe_physics_radius`；q0 到 q3 各自保存独立投放横坐标。
- 支撑边统一为 supporter -> supported fruit，并显式区分水果、地板和左右墙，
  不使用负水果 ID 作为边界哨兵。
- 为低级水果埋藏和封路归因保留 `critical_blocker_ids`、
  `inversion_blocker_ids` 和 `reachable_partner_ids`。
- 构造时校验 15 位 mask/count、15 项动作数组、比例/有限值、游戏等级范围、
  直接投放探针半径、当前水果引用、支撑缓存一致性、接触时间窗和确定性排序。
- 按 Reward V2 固定公式复算每个队列槽容量及 q0-q3 衰减聚合容量，拒绝彼此矛盾的
  重复真值。
- 完整分析对象继续保持 worker-local，不加入主 `TensorTransition`、`ReplayBuffer`
  或 `RolloutStats`。

## 涉及文件

- `src/daxigua_rl/attribution/schema.py`: 新增状态归因纯数据契约和不变量。
- `src/daxigua_rl/attribution/__init__.py`: 提供 attribution 包公开导出。
- `tests/test_attribution_schema.py`: 新增构造、引用、公式、pickle 和 Windows spawn
  回归测试。
- `docs/rl/CAUSAL_ATTRIBUTION_V1.md`: 补充动作对齐、探针半径、队列槽位和实现进度。
- `docs/rl/INTERFACE_V0.md`: 登记当前可用的 schema 接口和边界。
- `docs/project_map/PROJECT_FILE_INDEX.md`: 登记新包、测试和可复用入口。
- `docs/README.md`、`src/daxigua_rl/README.md`: 更新当前进度和下一实现步骤。

## 验证方式

在 `python-torch` conda 环境中执行：

```text
PYTHONPATH=src python -m unittest tests.test_attribution_schema -v
PYTHONPATH=src python -m unittest discover -s tests -v
```

结果分别为 17 项 schema 测试和 50 项全量测试全部通过。schema 测试包含真实
Windows `spawn` 子进程中的完整对象 pickle 往返。

## 备注

- 本次只实现数据契约，没有伪造栅格可达性、支撑检测、伙伴分量或 potential 算法。
- `contact_influence_edges` 目前通过 `incoming_transition_key` 固定时间归属；后续
  tracker 接线时仍应将动态证据组织在 worker-local step record 中。
- 封闭空腔需要稳定的区域面积、顶部连通和边界水果表示；该结构将在下一步确定实际
  栅格/可达算法时补充，避免 schema 阶段先固化错误的区域模型。
- `partner_ids` 与 `partner_components` 的精确图论关系、非 `merge_pair` motif 的
  有序角色语义，也在下一步 StateAnalyzer 输出规则确定后收紧。
