# 合成大西瓜 accelerated-v1

这是模型与训练系统重构分支。当前已在稳定领域契约之上加入
支持 CUDA 的多环境并行物理模拟器，但仍不包含模型和正式训练入口。

## 当前能力

- 水果等级、半径、合成计分等稳定领域规则；
- 不依赖 pygame、pymunk 或 PyTorch 的状态数据契约；
- PyTorch Tensor CPU 回退后端和单 Kernel CUDA 高吞吐后端；
- 批量重置、批量动作、独立 RNG、碰撞、滚动、合成、连锁、稳定和终止；
- 合成水果继承两颗来源水果的总线动量和关于合成中点的总角动量；
- 零拷贝 Tensor 状态读取、Python 单环境适配器和可插拔奖励接口；
- 指定 CUDA 环境的逐物理帧抽样录制和浏览器交互回放；
- 按当前规则重新实现的 Pymunk 行为参考环境，只用于对照和回退。

`daxigua.core` 仍只依赖 Python 标准库。模拟器的独立依赖见
`requirements-simulator.txt`。

## 当前明确不包含

- 桌面或 Android 游戏；
- GNN、DQN 或其他模型结构；
- replay、rollout、trainer 和训练脚本；
- Reward V1/V2、状态分析、因果归因、反事实或 Shapley；
- 正式 RL 奖励和 CUDA 训练管线。

这些能力只有在新设计被明确确认后才会逐项加入。

## 使用 python-torch

```powershell
$python = 'C:\Users\yefan\miniconda3\envs\python-torch\python.exe'
$env:PYTHONPATH = 'src'
& $python -B -m unittest discover -s tests -v
```

CUDA 后端首次运行会用 Ninja、MSVC 和本机 CUDA Toolkit 编译扩展，
之后复用 `.torch_extensions/` 缓存。

```python
import torch

from daxigua.simulator import SimulatorConfig, TensorVectorSimulator

simulator = TensorVectorSimulator(
    4096,
    config=SimulatorConfig(max_fruits=128),
    device='cuda',
)
actions = torch.randint(0, 21, (4096,), device='cuda')
result = simulator.step(actions)

reset_mask = result.physics.done | result.physics.truncated
simulator.reset(reset_mask)
```

`max_physics_frames` 是两次模型决策之间等待稳定的上限，不是回合上限。
若跑满上限仍未稳定，`result.physics.settle_timeout` 为真，但
`done`、`truncated` 和 `needs_reset` 都保持为假；下一次 `step` 会保留所有
位置、线速度、角度和角速度，在仍运动的场景中继续投放。

调试物理效果时，可以随机预热一批环境、抽取其中若干环境并记录下一次
完整投放。输出包含可直接打开的 HTML 回放和保留完整 Tensor 的 `.pt` 文件：

```powershell
& $python tools\record_cuda_replay.py `
    --num-envs 256 --warmup-drops 4 --record-drops 12 `
    --samples 3 --frame-stride 2
```

训练或模型评估代码也可以直接调用
`simulator.step_with_trace(actions, env_indices, frame_stride=2)`，只记录指定环境；
没有启用追踪时不会分配逐帧缓冲。

正式奖励尚未定义。需要类 Gymnasium 返回值时，必须显式为 `VectorEnv` 提供
`RewardComputer`，不能把游戏分数静默当成 RL 奖励。

## 验证和性能

PowerShell：

```powershell
$env:PYTHONPATH = 'src'
& $python benchmarks\benchmark_vector_simulator.py --num-envs 4096 --steps 20
& $python benchmarks\compare_reference_distribution.py
& $python benchmarks\run_random_until_stop.py --num-envs 4096 --max-drops 1000
```

需要在正式长局计时后抽取完整局回放时，可以启用确定性复跑。下面的命令会均匀抽取
20 个环境，每条从第一次投放保存到失败或投放上限；录像复跑不计入正式模拟耗时：

```powershell
& $python benchmarks\run_random_until_stop.py `
    --num-envs 4096 --max-drops 1000 `
    --replay-samples 20 --replay-full-episodes --replay-frame-stride 2
```

完整局会分别生成独立 HTML，并提供一个轻量索引页，避免浏览器一次加载 20 条
长轨迹。只需查看终局前短片段时，可以不传 `--replay-full-episodes`，再通过
`--replay-tail-drops` 控制尾段投放数。

## 分支来源

`codex/accelerated-v1` 从提交
`1a95a3d9a47bc6774144061f05ef067098b43544` 分叉。完整旧训练体系和 Android 成品
继续保留在原项目分支中，不作为本分支的运行依赖。
