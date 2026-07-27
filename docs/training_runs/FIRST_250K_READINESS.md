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
- [ ] 最终启动提交工作树干净，且本地、云端 HEAD 完全一致。
- [ ] 最终提交上的全量单测与 compile/py_compile 零失败。
- [ ] 正式 250k preflight：`ready=true`、required failures 为空、32/32 snapshot、
  正式配置契约和 CUDA actor 链路通过。
- [ ] 正式 run、monitor、control 目录在启动前不存在。
- [ ] 普通资源监控和 cgroup 监控先于训练启动，并写入独立退出码。
- [ ] 训练以 `configs/train_dqn_causal_250k.toml` 从 update 0 启动；命令不含
  `--resume`、`--total-updates` 或 `--overwrite`。
- [ ] dashboard 切换到正式 run，进度、资源与 ETA 非 stale。
- [ ] 启动后 update/env_steps 递增，无 Traceback、CUDA OOM、Xid、NaN/Inf，
  `memory.events` 无新增 OOM。

启动身份、preflight 哈希、进程 PID、监控目录和首次稳定心跳在训练实际启动后写入
单独的启动记录；不能在运行前预填为已通过。
