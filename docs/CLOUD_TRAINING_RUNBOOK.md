# 云端正式训练与自动接力规范

本文是云服务器模型训练、排队和自动接力的当前统一入口。历史部署细节可按需查阅
`docs/codex/`，但不得用旧记录中的服务器、PID、路径或“当前状态”替代本规范与现场检查。

## 1. 开始前必须读取

1. 读取本机 `docs/CLOUD_SERVER_LOCAL.md`，只连接其中登记为当前可用的实例；
2. 读取目标配置附近的 `configs/AGENTS.md`、训练实现附近的
   `src/daxigua/rl/AGENTS.md`；
3. 检查服务器现有训练进程、`runs/training_queue.json`、`/root/autodl-tmp/training-relay/`
   和占用中的面板端口；
4. 若服务器已有正式训练，默认加入现有串行队列或部署自动接力，不得并行争用同一张GPU，
   也不得在未检查接力状态时另起一套调度；
5. 只有确认服务器为空闲且无等待任务时，才直接启动新的正式训练。

`src/daxigua/rl/training_queue.py`只实现队列文件的校验、合并与原子写入契约；真正启动任务的
是云端接力或调度守护进程。不能因为该模块本身不启动训练，就判断项目不存在训练队列机制。

## 2. 队列与接力状态

统一面板队列文件为`runs/training_queue.json`，格式见`docs/PROJECT_PORTAL.md`。面板只读，
接力或调度程序负责写入。常用状态为：

- `queued`：已排期，尚未进入依赖等待；
- `waiting`：正在等待前序任务或其它明确条件；
- `preflight`：门禁已经释放，正在执行启动前校验；
- `running`：训练主阶段运行中；
- `evaluating`：主预算已到，仍在执行最终评测或产物收尾；
- `completed`：训练、要求的评测和产物校验均完成；
- `failed / stopped / cancelled`：不得自动视为成功并继续后继任务。

判断真实状态时应同时检查：训练PID及命令行、`run_status.json`、面板`/api/status`、
接力目录中的`relay_status.json`与`relay.log`。实时进度可能来自`metrics.jsonl`和面板聚合，
因此不能只凭一个长时间未刷新的简化`run_status.json`断言训练停滞。

## 3. 正式部署门禁

部署必须固定并记录以下身份：Git提交、配置路径、目标run、训练seed、物理身份、训练FPS、
初始化方式、来源checkpoint路径及SHA-256。代码优先通过校验过的Git bundle或其它固定提交
方式进入独立clean工作树，不让新部署修改正在运行任务的代码目录。

启动前至少完成：

1. 解析目标TOML并核对模型、物理、Replay、旁路、评测和墙钟预算；
2. 核对目标run不存在或为空，禁止覆盖已有产物；
3. 校验来源checkpoint完整SHA和训练进度；
4. 明确使用`--resume`还是`--init-checkpoint`：前者恢复模型、目标网络、优化器、RNG与
   训练进度，后者只迁移在线权重并重置训练状态；
5. 在目标GPU上完成适当的CUDA preflight或smoke；若已有训练占用同一GPU，预检也应进入
   串行接力阶段，不得提前争用GPU；
6. 启动后复核PID、`run_identity.json`、checkpoint来源、transition增长、GPU/显存、面板和
   定期checkpoint；这些只证明部署正确，不代表策略变强。

## 4. 自动接力必须fail-closed

后继训练只有在前序任务满足全部门禁后才可启动：

1. `run_status.json`明确为`phase=completed`；
2. 计划中的父轨迹与旁路transition预算均已达到；
3. `checkpoints/final.pt`、要求的最终评测索引和`artifact_manifest.json`完整落盘；
4. 来源checkpoint、目标配置和固定代码身份再次校验通过；
5. 只终止已完成任务留下的只读面板进程，并核对PID的命令行和工作目录；
6. 新训练通过启动门禁并接管面板后，队列才更新为`running`。

前序训练若失败、停止、提前退出、产物缺失或身份不匹配，接力必须写入`failed`并停止；不得
自动改用`best.pt`、中间checkpoint或更改配置继续训练。接力守护应保存PID、状态和日志，且
部署后至少连续观察一次心跳，确认前序训练未被修改、后继训练没有提前占用GPU。

## 5. 完成与归档

训练完成后校验最终checkpoint可加载、产物清单中的文件及SHA一致，再把可复核产物迁回本地。
新模型、新训练方法或有决策价值的正式评估按`docs/AGENTS.md`更新模型登记、评估报告和比较
矩阵。单次训练完成、loss下降、吞吐达标或实时窗口分数提高都不能单独写成策略提升结论。

相关现有说明：

- 队列文件契约：`docs/PROJECT_PORTAL.md`；
- 已跑通的接力实例：`docs/codex/48_部署辅助旁路至128M基线自动接力_2026_08_10.md`；
- 完整fail-closed实例：`docs/codex/60_部署128M至120FPS适应训练自动接力_2026_08_11.md`。
