# 实现 Merge Potential 大规模对局统计工具

## 原因

后续计划增加 Merge Potential 辅助任务，需要先了解不同等级水果距离下一次参与合成的
投放次数`T_merge`、终局仍未合成的比例，以及少量场景和局部状态因素的解释能力。正式
数据来自`SAB-128`在当前无自由落体快进环境中的大规模greedy对局，当前阶段先准备工具，
不在本地运行大规模评估或性能测试。

## 实现

- 新增`merge_potential_stats.py`，利用模拟器已有的局内`fruit_id`和合成事件
  `source_ids/new_fruit_id`建立水果生命周期，不修改物理或模型输入；
- 新增`collect_merge_potential_stats.py`，默认核对`SAB-128 final.pt`精确哈希，在30 FPS
  完整逐帧CUDA环境批量采集greedy对局；
- 原始数据拆分为水果快照、合成来源和对局结局三类分片张量表，采集阶段不固定可视化时间
  窗口或分箱；
- 首次观察始终记录，周期快照采用随水果寿命逐档扩大的间隔并限制单水果数量，避免超长寿命
  水果只留下早期状态；GPU结果按多个决策轮次合并回传，磁盘分片可在后台写入；
- 离线关联区分合成、自然终局未合成和截断未知。生命周期每颗水果只计一次；状态快照按
  同一水果实际快照数归一化，使单水果总权重为1；
- 记录空间占用率、水果高度、同等级水果数量、最近同等级中心距离和表面间隙；输出单因素
  表及“占用率×最近同级距离”“高度×同级数量”两类轻量交互表；
- 暴露并行环境数、特征分块、快照间隔、单水果上限、GPU回传间隔、分片规模、后台写入、
  模型编译和BF16候选开关，供云实例启动后进行局部性能调节；
- 将规范目录中的采集清单接入云端训练遥测API和Xigua Atlas实时训练页，只读展示进度、
  吞吐、样本量、峰值显存、物理身份和离线汇总状态，不读取原始张量分片；
- 新增`MERGE_POTENTIAL_DATA.md`说明数据口径、运行命令、输出表和建议样本量。

## 主要文件

- `src/daxigua/rl/merge_potential_stats.py`
- `src/daxigua/rl/merge_potential_status.py`
- `tools/collect_merge_potential_stats.py`
- `tests/test_merge_potential_stats.py`
- `portal/app/TrainingWorkspace.tsx`
- `docs/MERGE_POTENTIAL_DATA.md`

## 验证

- Python静态编译通过；
- `tests.test_merge_potential_stats`共5项通过，覆盖首次/周期采样、年龄自适应间隔及上限、最近同级距离、
  合成事件展开、重复快照归一化、自然终局未合成和截断未知；
- 与`tests.test_rl_baseline`合并回归共37项通过；
- 门户清单发现和归一化测试通过；门户后端与RL基线合并回归共39项通过；
- 门户`npm run lint`和`npm run build`通过；
- 使用纯合成张量表验证离线输出的生命周期、时间范围、单因素和交互CSV；
- 未加载checkpoint，未在本地运行模型对局或性能测试；云端吞吐和正式参数均待服务器启动
  后测量。

## 当前边界

- 当前工具只记录和分析标签，不修改observation、Replay、Q网络或训练loss；
- 统计结果属于`SAB-128`策略和当前物理环境下的条件分布，不自动推广到所有策略；
- 默认20,000局是正式基线建议，是否扩到50,000局由云端吞吐、高等级样本量和交互格有效
  权重决定。
