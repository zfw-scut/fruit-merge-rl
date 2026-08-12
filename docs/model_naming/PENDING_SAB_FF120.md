# 待部署模型登记：SAB-FF120

| 项目 | 值 |
| --- | --- |
| 短称 | `SAB-FF120` |
| 正式ID | `sab-full-fall-t120-16m-r1` |
| 状态 | 待部署 |
| 来源 | `SAB-T120 best.pt`，weights-only |
| 目标物理 | `tensor_cuda_v3_full_fall`，120 FPS完整逐帧下落 |
| 预算 | 16M transition |
| 配置 | `configs/sab-full-fall-t120-16m-r1.toml` |
| run | `runs/sab-full-fall-t120-16m-r1_seed20260813` |

该独立登记用于系统崩溃后 `MODEL_REGISTRY.md` 无法被标准补丁工具读取期间保留已确认身份。
服务器启动后、正式部署前应将本行合并进 `MODEL_REGISTRY.md` 的“计划或训练中”，再删除本
临时登记文件。不得因为完成本地预检而提前把状态改为“训练中”。
