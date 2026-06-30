# 新增 RolloutCollector 经验采集器

## 修改原因

在 `Transition` 和 `ReplayBuffer` 完成后，需要一个训练侧采集器，把无渲染环境、图构建、动作选择和经验写入串成完整采样链路，为后续 DQN 更新逻辑提供经验来源。

## 主要修改

- 新增单进程 `RolloutCollector`。
- 新增 `EpsilonGreedyPolicy`，支持随机探索和模型 argmax。
- 新增 `RolloutStats`，返回采集步数、episode 统计、动作选择统计和 buffer 大小。
- `collect_steps(step_count, epsilon)` 会自动：
  - 从当前状态构建图。
  - 用 epsilon-greedy 选择 `action_offset`。
  - 调用 `DaxiguaEnv.step(action_offset)`。
  - 构建 `Transition`。
  - 写入 `ReplayBuffer`。
  - episode 结束后自动 reset 并继续采集。
- collector 依赖 PyTorch，因此只通过 `daxigua_rl.training` 懒加载导入，不放进 `daxigua_rl` 顶层导出。

## 涉及文件

- `src/daxigua_rl/training/collector.py`: 新增 `RolloutCollector`、`EpsilonGreedyPolicy`、`RolloutStats`。
- `src/daxigua_rl/training/__init__.py`: 增加 collector 相关对象的懒加载导出。
- `src/daxigua_rl/README.md`: 补充当前训练侧接口说明。
- `docs/rl/INTERFACE_V0.md`: 记录 collector 接口和采集流程。
- `docs/project_map/PROJECT_FILE_INDEX.md`: 更新项目文件索引。

## 验证方式

- `conda run -n python-torch python -m py_compile src/daxigua_rl/training/collector.py src/daxigua_rl/training/__init__.py`
- 使用 `DaxiguaEnv`、`GraphBuilder`、`ReplayBuffer` 和 `RolloutCollector` 采集随机 transition。
- 使用 `GNNQNetwork` 验证 `epsilon=0.0` 的 greedy 分支可以正常采集。
- `git diff --check`

## 备注

- 本次仍不实现 DQN loss、target network、optimizer 或模型参数更新。
- 当前 collector 是单进程同步采集器，后续多进程采样可以在此基础上拆 worker。
