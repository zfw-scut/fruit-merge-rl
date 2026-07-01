# 优化 ReplayBuffer 图特征内存占用

## 修改原因

正式 CUDA 训练期间，Python 进程内存曾上涨到约 9GB，整机 16GB 内存接近耗尽并伴随黑屏。结合训练日志中 `buffer=30000` 的状态，主要风险点是 ReplayBuffer 长期保存大量 `TensorTransition`，而每条 transition 同时保存当前图和下一状态图，CPU 常驻图特征内存与 Python 对象开销较高。

## 主要修改

- `RolloutCollector` 写入 ReplayBuffer 时固定把图特征转为 `float16`。
- 删除 `train_dqn.py` 中的 `--replay-storage-dtype` 兼容参数，避免后续误用高内存的 float32 replay 存储。
- `GNNQNetwork` 在前向入口把 `GraphTensor` / `GraphBatch` 移动到模型所在 device 时，同时转成模型参数 dtype。这样 ReplayBuffer 固定半精度存储，但模型仍按 float32 参数正常训练。
- 删除旧的框架无关 `Transition` 结构和导出，训练主链路只接受 `TensorTransition`。
- 新增测试覆盖：ReplayBuffer 固定以 float16 保存图特征时，float32 模型仍能完成 DQN 更新。

## 涉及文件

- `src/daxigua_rl/training/collector.py`: 固定以 `float16` 写入 replay 图特征。
- `src/daxigua_rl/scripts/train_dqn.py`: 删除 replay 存储 dtype 参数。
- `src/daxigua_rl/models/gnn_q.py`: 模型前向时统一将输入图特征转成模型参数 dtype。
- `src/daxigua_rl/training/dqn.py`: 要求 ReplayBuffer 中必须是 `TensorTransition`。
- `src/daxigua_rl/training/tensor_transition.py`: 删除旧 `Transition` 转换入口。
- `src/daxigua_rl/training/transition.py`: 删除旧经验结构。
- `tests/test_graph_batch_training.py`: 增加半精度 replay 存储训练测试。
- `docs/project_map/PROJECT_FILE_INDEX.md`: 更新训练入口和可复用组件说明。

## 验证方式

- `PYTHONPATH=src conda run --no-capture-output -n python-torch python -m unittest tests.test_graph_batch_training`
- `PYTHONPATH=src conda run --no-capture-output -n python-torch python -m unittest discover -s tests`
- `PYTHONPATH=src conda run --no-capture-output -n python-torch python -m daxigua_rl.scripts.train_dqn --run-dir /tmp/daxigua_train_half_storage_smoke --device cpu --hidden-dim 32 --message-layers 1 --total-updates 3 --warmup-steps 8 --collect-per-update 1 --batch-size 4 --replay-capacity 32 --log-interval 1 --save-interval 0 --eval-interval 0 --plot-interval 0 --progress-interval 0 --action-count 5 --max-physics-frames 120 --stable-frames 4`

## 备注

- 该优化主要降低图特征张量内存，不会消除 Python 对象、索引张量以及当前图/下一图双份保存带来的全部开销。
- 在 16GB 内存机器上继续训练时，仍建议将 `--replay-capacity` 控制在 `15000` 到 `30000` 之间，并配合外部资源监控观察实际 RSS。
