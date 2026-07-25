# 训练实验索引

生成时间：`2026-07-26T01:38:30+08:00`

本索引由 `tools/export_training_catalog.py` 从本地 `runs/` 自动提取。
数值只代表 CSV 中已经落盘的最后状态，不代表后台进程退出前尚未写入的数据。

## 训练实验

| Run | 状态 | Updates | Env Steps | Episodes | 平均分 | 中位数 | 最高分 | Eval 最佳 | 数据体积 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [debug_reward_breakdown_smoke](runs/debug_reward_breakdown_smoke/summary.md) | 已完成 | 4 | 16 | 4 | 0.500 | 0 | 2 | 未记录 | 844.0 KiB |
| [dqn_20260701_040141](runs/dqn_20260701_040141/summary.md) | 已完成 | 200 | 400 | 0 | 未记录 | 未记录 | 未记录 | 未记录 | 19.0 MiB |
| [dqn_baseline_h128_l3_10k](runs/dqn_baseline_h128_l3_10k/summary.md) | 未完成或中断 | 5,700 | 6,700 | 0 | 未记录 | 未记录 | 未记录 | 未记录 | 12.9 MiB |
| [dqn_baseline_h128_l3_10k_eps10k](runs/dqn_baseline_h128_l3_10k_eps10k/summary.md) | 已完成 | 10,000 | 11,000 | 0 | 未记录 | 未记录 | 未记录 | 未记录 | 19.1 MiB |
| [dqn_cuda_h128_l3_100k_smooth](runs/dqn_cuda_h128_l3_100k_smooth/summary.md) | 已完成 | 100,000 | 105,000 | 810 | 469.514 | 455.500 | 1,173 | 1,008 | 1.1 GiB |
| [dqn_fast30_parallel_h128_l3_100k](runs/dqn_fast30_parallel_h128_l3_100k/summary.md) | 未完成或中断 | 116,000 | 933,000 | 7,846 | 409.817 | 398 | 1,037 | 991 | 11.9 GiB |
| [smoke_perf_parallel](runs/smoke_perf_parallel/summary.md) | 已完成 | 3 | 20 | 0 | 未记录 | 未记录 | 未记录 | 未记录 | 1.2 MiB |
| [smoke_train_script](runs/smoke_train_script/summary.md) | 已完成 | 1 | 6 | 0 | 未记录 | 未记录 | 未记录 | 未记录 | 757.5 KiB |
| [smoke_train_toml_launcher](runs/smoke_train_toml_launcher/summary.md) | 已完成 | 1 | 6 | 0 | 未记录 | 未记录 | 未记录 | 未记录 | 760.9 KiB |

## 非训练输出

这些目录没有标准 `config.json + metrics.csv` 训练组合，因此不参与模型得分比较。

| 目录 | 类型 | 本地相对路径 | 数据体积 |
| --- | --- | --- | ---: |
| `cuda_stress` | 资源或 CUDA 诊断 | `runs/cuda_stress` | 70.1 KiB |
| `debug_logs` | 日志集合 | `runs/debug_logs` | 99.6 KiB |
| `debug_physics_compare_smoke` | 物理模式或评估对比 | `runs/debug_physics_compare_smoke` | 224.2 KiB |
| `launcher_logs` | 日志集合 | `runs/launcher_logs` | 1.7 MiB |
| `physics_compare_model` | 物理模式或评估对比 | `runs/physics_compare_model` | 251.7 KiB |
| `resource_monitor` | 资源或 CUDA 诊断 | `runs/resource_monitor` | 175.3 KiB |

## 阅读说明

- 比较模型效果时以真实 `episode score` 为准，不以 shaped reward 代替游戏分数。
- `未完成或中断` 仅表示最后一行 update 小于配置目标，不能判断进程是手动停止还是异常退出。
- Smoke/debug 运行主要验证代码链路，不应与正式长训直接比较。
- 本目录没有复制 checkpoint 和 ReplayBuffer；需要时按各 Run 的 `artifacts.md` 单独迁移。
