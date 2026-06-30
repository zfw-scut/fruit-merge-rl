# 新增 ReplayBuffer 经验回放池

## 修改原因

在 `Transition` 经验结构完成后，需要一个固定容量的经验回放池，用于保存 rollout 产生的经验，并为后续 DQN 训练提供随机采样能力。

## 主要修改

- 新增基础版 `ReplayBuffer`。
- 默认容量设为 `100_000`，即十万条经验。
- 容量满后通过环形写入覆盖最旧经验。
- `sample(batch_size)` 使用均匀随机无放回采样。
- 第一版采样结果返回 `tuple[Transition, ...]`，不在 buffer 层做 tensor batch。
- 暂不实现优先经验回放、磁盘缓存、多进程共享或压缩存储。

## 涉及文件

- `src/daxigua_rl/training/replay_buffer.py`: 新增 `ReplayBuffer`。
- `src/daxigua_rl/training/__init__.py`: 导出 `ReplayBuffer`。
- `src/daxigua_rl/__init__.py`: 顶层导出 `ReplayBuffer`。
- `src/daxigua_rl/README.md`: 补充训练侧接口说明。
- `docs/rl/INTERFACE_V0.md`: 记录 `ReplayBuffer` 接口和当前约定。
- `docs/project_map/PROJECT_FILE_INDEX.md`: 更新项目文件索引。

## 验证方式

- `conda run -n python-torch python -m py_compile src/daxigua_rl/training/replay_buffer.py src/daxigua_rl/training/__init__.py src/daxigua_rl/__init__.py`
- 构造真实 `Transition`，写入 `ReplayBuffer`，验证 `len()`、`sample()`、容量覆盖和时间顺序输出。
- `git diff --check`

## 备注

- `ReplayBuffer` 不依赖 PyTorch，仍然保存框架无关 `Transition`。
- 后续 DQN trainer 从 buffer 采样后，再负责把图转换为模型输入张量。
