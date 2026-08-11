# 模拟器导航

先按目标选文件，不要从 `vector.py` 通读。

| 修改目标 | 文件 |
| --- | --- |
| 几何、帧率、物理参数 | `config.py` |
| Batch 数据契约 | `types.py` |
| Tensor 状态和物理编排 | `vector.py` |
| CUDA 加载 | `cuda_backend.py` |
| Python/CUDA 参数桥接 | `cuda/vector_step.cpp` |
| CUDA 碰撞、积分、合并 | `cuda/vector_step_kernel.cu` |
| 普通奖励接口 | `reward.py` |
| Reward V2/V2.1 | `spatial_reward.py` |
| 回放 | `replay.py`、`replay_web.py` |
| 场景实验室 | `scenario_lab.py`、`scenario_lab_live.py`、`scenario_lab_service.py`、`scenario_lab_server.py` |

## 边界

- 训练、评估和场景实验室共享 `TensorVectorSimulator`；不要建立第二套物理规则。
- 30 FPS 训练档来自 `SimulatorConfig.training_fast()`；默认配置是 120 FPS。修改时明确影响哪种物理身份。
- CUDA 调用签名或参数变化必须同步 Python、C++、CUDA 三层。
- 修改 Batch 类型、动作效果或物理结果时，用符号搜索 RL 和场景实验室消费者。
- 仅改场景 API/UI 行为时，不阅读碰撞 Kernel；仅改回放时，不阅读训练代码。

## 最小测试

| 范围 | 测试 |
| --- | --- |
| 物理、CUDA、配置 | `tests.test_vector_simulator` |
| Reward V2/V2.1 | `tests.test_spatial_reward` |
| 回放 | `tests.test_replay` |
| 场景实验室 | `tests.test_scenario_lab` |

先运行相关单项；CUDA 改动再运行对应 CUDA 用例和必要 benchmark。新物理或奖励语义才按 `docs/AGENTS.md` 扩展证据阅读。
