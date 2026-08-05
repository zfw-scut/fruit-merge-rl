# 文档索引

accelerated-v1 当前只维护最小仓库治理文档：

- `GIT_WORKFLOW.md`：分支、提交语言和提交前检查规则；
- `LEGACY_PROJECT_EXPERIENCE_INDEX.md`：旧分支关键提交、证据、经验和禁止照搬边界；
- `model/STRATEGIC_VIEW_CONNECTIVITY.md`：新模型战略视角、水果可联通性、多时间尺度直接合成机会和 pair survival 算法设计；
- `model/STRATEGIC_LEVEL_AGGREGATION.md`：水果个体与等级节点、数量敏感聚合、理论进位、合并机会和合并债务算法设计；
- `model/STRATEGIC_REGION_ANCHORS.md`：7 个固定横向战略锚点、软归属、垂直区域摘要和高层计划位置语义；
- `model/HIGH_LEVEL_PLAN_LIFECYCLE.md`：事件驱动的可变时长显式计划、完成、中止、抢占和分类最长时域；
- `model/ACTION_CONDITIONED_EFFECT_VIEW.md`：共享物理水果 GNN、21 个单向动作探针、反事实隔离和 Dueling 读出；
- `model/MODEL_DESIGN_STATUS_OVERVIEW.md`：新模型组件、节点、边、输出、弃用项、设计状态和实现基础总览；
- `model/GNN_DQN_BASELINE.md`：第一版 GNN-DQN 基线的模型范围、1-step Double DQN、纯分数奖励和排除项；
- `model/GNN_DQN_TRAINING_SYSTEM.md`：30 FPS 独占训练、30/120 FPS 隔离评估、GPU Replay、性能标定、动态扩容、实时面板和云端归档规格；
- `model/LOCAL_MODEL_VIEWER.md`：checkpoint 本地加载、120 FPS greedy 游玩、逐帧浏览器页面和 Q 值展示；
- `codex/RULES.md`：较大修改的记录规则；
- `codex/01_建立accelerated_v1最小基线_2026_08_01.md`：本分支首个结构记录；
- `codex/02_对齐标准合成规则_2026_08_01.md`：当前水果、计分和西瓜相消规则；
- `codex/03_新增CUDA多环境并行模拟_2026_08_04.md`：批量物理、兼容层和性能门禁。
- `codex/04_新增CUDA物理抽样回放_2026_08_04.md`：指定环境逐帧追踪和离线回放。
- `codex/05_诊断CUDA稳定截断_2026_08_04.md`：截断环境逐帧抽样、稳定阻塞证据和原因。
- `codex/06_修正CUDA静止残余速度_2026_08_04.md`：逐水果静止速度修正、性能与长局验证。
- `codex/07_将物理等待上限改为决策边界_2026_08_04.md`：超时后保留运动状态并允许下一次投放。
- `codex/08_增加长局终局抽样回放_2026_08_04.md`：4096 环境长局测试和完整局复跑机制。
- `codex/09_让合成水果继承动量_2026_08_04.md`：合成后的线动量、角动量继承与验证。
- `codex/10_重构纹理化长局回放_2026_08_04.md`：水果贴图、压缩长局、调试控制和单页多局目录。
- `codex/11_增加训练帧率与稳定投放快进_2026_08_04.md`：30 FPS 显式训练档、稳定场景自由下落快进与 4096 环境对比。
- `codex/12_增加30帧超时分层回放_2026_08_05.md`：逐环境超时统计、分层完整局回放、超时跳转与快进追踪容量修复。
- `codex/13_修复30帧碰撞与静止语义_2026_08_05.md`：角阻尼、低速恢复、自适应碰撞子步、时间步一致静止语义与完整局对比。
- `codex/14_记录新模型总体技术方向_2026_08_05.md`：新模型的分层决策、状态编码、计划条件化、辅助任务和搜索增强总体草案。
- `codex/15_实现第一版GNN_DQN训练系统_2026_08_05.md`：第一版物理 GNN-DQN、30 FPS 正式训练、隔离双帧率评估、性能标定、面板和本机验证。
- `codex/16_升级Win11训练面板与效果曲线_2026_08_05.md`：中文 Win11 Fluent 训练面板、可靠数字格式和分数/损失/吞吐历史曲线。
- `codex/17_新增本地模型观看器_2026_08_05.md`：复用旧观看器原则重新实现的 checkpoint 浏览器游戏演示链路。

旧模型、训练、因果归因、云服务器、Android 和 UI 文档没有迁入本分支。它们仍可按
`LEGACY_PROJECT_EXPERIENCE_INDEX.md` 从旧分支只读查阅，但不属于 accelerated-v1 的
实现依据。

当前已实现第一版 GNN-DQN 基线和训练系统。后续战略、计划、动作辅助和搜索模块继续按
专项文档逐项验证，以本基线作为速度与效果比较对象。
