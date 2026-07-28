# 第一次 250k 完整因果训练就绪记录

最后更新：2026-07-28

正式配置：`configs/train_dqn_causal_250k.toml`

运行手册：`../rl/FIRST_250K_RUNBOOK.md`

## 1. 决策冻结

- 第一次结构感知 V2 大规模训练的最终长度冻结为 `250000` updates。
- 必须从 update 0 使用独立 250k 配置启动；不得把 500k 配置临时覆盖、跑到一半
  停止，也不得从 10k checkpoint 延长。
- fresh run 的 `epsilon_schedule_total_updates` 必须为 `250000`；smooth 探索率
  锚点为 75k≈0.50、125k≈0.20、175k≈0.07、200k 起为 0.05。
- 用户在 2026-07-28 明确批准：已独立完成的 V2 10k 校准替代缺失的 V2 5k
  顺序门禁，并授权在最终检查通过后直接启动 250k。无需重复 5k 或额外 25k。

## 2. 已有 10k 证据

| 项目 | 结果 |
| --- | --- |
| 训练源码 commit | `ddcb249` |
| 更新与启动方式 | 从 update 0 独立完成 10000 updates；不是 resume |
| 运行时长 / 吞吐 | 55m25s；约 3.047 updates/s |
| 完成局数 | 1175 |
| 分数趋势 | 训练窗口约 350→640 |
| greedy eval | update 5000 mean 560.2；update 10000 mean 577.6 |
| 正式检查 | readiness `ready=true`，required failures 为空 |
| 结构/归因链路 | 六维结构 target、集中 actor、反事实、Shapley、共享预算、checkpoint/replay 均通过 |
| 资源安全 | cgroup OOM 事件为 0；监控正常看到目标进程并在其结束后退出 |

该 10k 证明同一模型与训练链路能够跨越 5k 观察窗口稳定运行。最终 250k 使用新的
冻结提交，因此仍必须在该提交上重新执行全量测试和正式 preflight；10k 不能替代
这两项。

## 3. 服务器容量核算

| 资源 | 启动前核对值 | 门禁 |
| --- | ---: | ---: |
| GPU | RTX 5090，32607 MiB，启动前空闲 | CUDA 正常，训练前无其它占用 |
| 有效 CPU | 25 核 | ≥24 核 |
| cgroup 内存上限 | 约 90 GiB | ≥64 GiB |
| 可回收工作集余量 | 约 84.2 GiB | ≥16 GiB |
| `/root/autodl-tmp` 可用磁盘 | 约 135 GiB | 启动时 ≥80 GiB，运行硬停止线 40 GiB |

10k 的 cold replay 约 17.2 GiB 已接近配置容量上限，而不是按训练长度线性增长：
主 replay 容量为 100000、hot 为 8000、segment 为 1024，最多保留约 90 个 cold
segments，超过容量后删除最旧段。因此当前磁盘余量能够承载 250k；旧 10k 产物保留，
不通过删除证据换取空间。

## 4. 最终启动硬门禁

- [x] 250k 独立配置与 run 目录已冻结。
- [x] 500k 配置被明确标为可选延长实验，不能通过当前正式配置契约。
- [x] 用户批准 10k 替代缺失 5k，并批准最终检查后启动。
- [x] 最终启动提交工作树干净，且本地、云端 HEAD 完全一致。
- [x] 最终提交上的全量单测与 compile/py_compile 零失败。
- [x] 正式 250k preflight：`ready=true`、required failures 为空、32/32 snapshot、
  正式配置契约和 CUDA actor 链路通过。
- [x] 正式 run、monitor、control 目录在启动前不存在。
- [x] 普通资源监控和 cgroup 监控先于训练启动；两者均存活并具有独立退出码路径。
- [x] 训练以 `configs/train_dqn_causal_250k.toml` 从 update 0 启动；命令不含
  `--resume`、`--total-updates` 或 `--overwrite`。
- [x] dashboard 切换到正式 run，进度、资源与 ETA 非 stale。
- [x] 启动后 update/env_steps 递增，无 Traceback、CUDA OOM、Xid、NaN/Inf，
  `memory.events` 无新增 OOM。

## 5. 实际启动身份

| 项目 | 实际值 |
| --- | --- |
| 训练源码 commit | `83026cad5b43a8d39a7e224445f28091a5f9c785`，`codex/work-1`，云端 clean |
| 本地测试 | 376 项通过，1 项仅因 Windows 无符号链接权限跳过；0 failure/error |
| 云端测试 | 376 项全部通过；compileall 通过 |
| preflight | `runs/preflight/first_250k_prelaunch_83026ca_20260728T000628Z.json` |
| preflight SHA-256 | `99598a5146408f7f131473109222448f0a334216ede0c21f94cbe31db8c86e22` |
| 启动时间 | `2026-07-28T08:18:51+08:00` |
| 正式 run | `runs/dqn_causal_structure_h256_l4_n3_250k` |
| 资源监控 | `runs/resource_monitor/formal_250k_83026ca_20260728T001851Z` |
| 阶段控制 | `runs/stage_control/formal_250k_83026ca_20260728T001851Z` |
| supervisor | PID `40568`，PID start-ticks 身份核对通过 |
| 面板 | `127.0.0.1:8765`，health `status=running`、`data_fresh=true` |

首次稳定快照在 update 103：目标为 250000，env steps 6648，训练和四个 supervisor/
监控进程均存活且 start-ticks 匹配；GPU utilization 约 55%，显存约 30871/32607 MiB，
有效 CPU 利用约 91.7%，可用内存约 73 GiB，磁盘可用约 135 GiB。cgroup 的
`low/high/max/oom/oom_kill` 全部为 0，未生成 `train.exit`，说明 run 正在运行而非
提前结束。

启动时 `config.json` 已确认 `total_updates=250000`、`epsilon_schedule=smooth`、
`resume=None`。fresh-run 代码将 epsilon horizon 直接取自 `args.total_updates`；
首个 20k checkpoint 落盘后，再由 checkpoint 明确复核
`epsilon_schedule_total_updates=250000`。

## 6. update 42348 故障与 40k 恢复

原 fresh 进程于 `2026-07-28T12:42:14+08:00` 在 update 42348 退出。直接异常为
读取 cold replay 时的 `OSError: [Errno 24] Too many open files`；服务器当时
`nofile` soft/hard 为 1024/65535。没有 CUDA OOM、Xid、cgroup OOM、磁盘不足或
模型非有限值证据，故障 segment 也可在独立进程重复加载且 FD 恒定。

根因是局部 Shapley 把含 2～12 个 `GraphTensor` 的 task 直接提交给 PyTorch
multiprocessing。默认 `file_descriptor` 共享策略让每个 tensor storage 占一个 FD，
可信样本又把原 graph 长期保存在因果 replay；最小复现实测每个双候选任务稳定留下
约 10 个 FD，约 90 个任务后与固定进程句柄共同触及 1024。修复提交
`dec106b2003135f35763259f13dfa4831f7f4863` 改为 task bytes 传输，并让启动器把
soft limit 提升至 65535。

恢复前完成：

- 本地 377 项测试通过，0 failure/error，1 项 Windows 权限预期跳过；
- 云端 Linux 377 项测试全部通过，编译检查通过；
- 12 个连续独立 Shapley task 的 FD 压测为 10→12 后保持不变；
- 修复提交正式 preflight `ready=true`、0 warning、32/32 snapshot，路径为
  `runs/preflight/first_250k_resume_dec106b_20260728T083700Z.json`，SHA-256 为
  `33ae10de89383c6ac2813bc3d237f8d6afa4329b50d9c0ea805c8a9cc8eb43a4`；
- 不可变 `step_00040000.pt` 完整恢复审计通过，SHA-256 为
  `900baf2329f72a8901506052ae47c6f9448915d674fb86c64aea437426d3df44`。

实际于 `2026-07-28T16:43:38+08:00` 从 update 40000 恢复。旧 monitor/control
和失败 cold generation均原样保留；新身份为：

| 项目 | 恢复值 |
| --- | --- |
| 训练源码 | `dec106b2003135f35763259f13dfa4831f7f4863`，云端 clean |
| checkpoint | `runs/dqn_causal_structure_h256_l4_n3_250k/checkpoints/step_00040000.pt` |
| monitor | `runs/resource_monitor/formal_250k_resume40k_dec106b_20260728T083800Z` |
| control | `runs/stage_control/formal_250k_resume40k_dec106b_20260728T083800Z` |
| cold 现场 | `runs/dqn_causal_structure_h256_l4_n3_250k/replay_cold.failed_at_42348_20260728T083800Z` |
| replay 恢复 | TD hot-only 8000/99176，省略旧 cold 91176；causal replay 20000 |
| 日志收口 | checkpoint 后 4 行 metrics、293 行 episode metrics 自动移入 orphan 文件 |
| 资源保护 | learner `nofile=65535/65535`，新双监控存活，面板重新绑定且数据新鲜 |

恢复快照已经越过 40k 并持续增加，但本节只记录“恢复成功且当前运行中”；250k
最终完成、最终 readiness 和模型效果仍须等待实际产物，不能提前打勾。
