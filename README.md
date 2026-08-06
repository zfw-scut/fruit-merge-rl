# 合成大西瓜 accelerated-v1

这是模型与训练系统重构分支。当前已在稳定领域契约和 CUDA 多环境并行物理之上，
实现第一版可正式训练的 GNN-Dueling Double DQN 基线。

## 当前能力

- 水果等级、半径、合成计分等稳定领域规则；
- 不依赖 pygame、pymunk 或 PyTorch 的状态数据契约；
- PyTorch Tensor CPU 回退后端和单 Kernel CUDA 高吞吐后端；
- 批量重置、批量动作、独立 RNG、碰撞、滚动、合成、连锁、稳定和终止；
- 合成水果继承两颗来源水果的总线动量和关于合成中点的总角动量；
- 零拷贝 Tensor 状态读取、Python 单环境适配器和可插拔奖励接口；
- 指定 CUDA 环境的逐物理帧抽样录制、纹理化离线播放器和多局回放目录；
- 按当前规则重新实现的 Pymunk 行为参考环境，只用于对照和回退。
- 定长混合局部物理图、q0～q3 队列图和 21 个单向动作探针；
- 1-step Dueling Double DQN、GPU Replay、分阶段预热和原子 checkpoint；
- 只使用 30 FPS 采集训练数据，隔离执行 30/120 FPS greedy 评估；
- 云端 CUDA 门禁、端到端性能标定、动态环境扩容和低开销 Web 面板；
- 最终 Replay 抽样、完整决策边界轨迹和 SHA-256 产物清单。
- 从任意当前格式 checkpoint 重建在线 GNN，并生成带逐帧物理、Q 值和动作解释的本地模型观看页面。
- 纯空间 Reward V2：21列可投放面积、队列对齐、标准占用补偿和相邻状态缓存；
- 鼠标交互式场景实验室，可调用真实物理并可视化21动作空间奖励与投放后局面。

`daxigua.core` 仍只依赖 Python 标准库。模拟器的独立依赖见
`requirements-simulator.txt`；正式训练的附加依赖见
`requirements-training.txt`。

## 当前明确不包含

- 桌面或 Android 游戏；
- 重型状态分析、因果归因、反事实或 Shapley；
- 战略 Pair Encoder、等级/区域聚合、高层计划器、FiLM、动作效果辅助任务和 BMCTS。

这些后续能力只有在新设计被明确确认后才会逐项加入；旧分支模型不作为运行依赖。

## 第一版 GNN-DQN 训练

本机或云服务器先执行正确性门禁，再在彼此隔离的子进程中扫描环境数、batch、BF16、
FP32 和可用时的 `torch.compile`，最后按 UTD=1 端到端吞吐和显存余量自动选择正式配置：

```bash
export PYTHONPATH="$PWD/src"
python tools/preflight_training.py
python tools/benchmark_training_pipeline.py
python tools/compare_reward_throughput.py
python tools/run_autotuned_training.py --max-wall-hours 12
```

也可以直接运行 `scripts/run_cloud_training.sh --max-wall-hours 12`。面板默认监听
`127.0.0.1:8765`，应通过 SSH 端口转发访问。训练只调用 30 FPS 物理；120 FPS 只在里程碑
和最终评估中使用，相关状态不会写入 Replay 或 loss。面板旁路默认每 120 秒原子更新
`<run_dir>/plots/training_curves.png`，评估完成和训练收尾时也会立即刷新；页面会自动显示
最新图片，最终 PNG 与 `training_curves.json` 一并作为训练产物保留。

快速验证完整主链：

```powershell
$env:PYTHONPATH = 'src'
& $python tools\train_gnn_dqn.py --smoke --device cuda `
    --run-dir runs\local-formal-smoke
```

## 本地观看训练模型

模型观看器默认使用 120 FPS 精确物理和 greedy 策略。它从 checkpoint 冻结配置重建
在线 GNN，记录真实逐帧运动，并生成不依赖服务端的浏览器游戏页面：

```powershell
& $python tools\watch_gnn_dqn.py `
    runs\gnn_dqn_baseline\checkpoints\best.pt `
    --device cuda --physics-fps 120 --open
```

页面提供播放、暂停、倍速、逐帧、逐投放和合成跳转，并显示 q0～q3、模型选择的落点、
21 个动作的 Q 值分布和推理耗时。增加 `--episodes 5` 会生成多局目录；增加
`--physics-fps 30` 可检查同一模型在训练物理下的表现。观看器不会把任何状态写回训练
Replay，也不会修改 checkpoint。

## 自定义场景实验室

使用真实物理和 Reward V2 后端启动交互式场景实验室：

```powershell
$env:PYTHONPATH = 'src'
& $python tools\open_scenario_lab.py --serve --device cuda --open
```

页面支持鼠标放置与拖动、右键删除、滚轮切换等级、撤销重做、q0～q3 编辑、21 动作
投放探针、30/120 FPS 参数选择、极端场景预设和 JSON 导入导出。“评估 21 个动作”
会批量执行同一场景的21个真实投放，显示势能前后值、原始空间变化、占用补偿、终局
动作和投放后水果，并在画布叠加每个未来水果的21列空间增减。省略 `--serve` 时仍可
生成完全自包含的离线编辑页面，但不会伪造物理或奖励。

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
完整投放。输出包含可直接打开的 HTML 回放和保留完整 Tensor 的压缩 `.pt.gz`
文件：

```powershell
& $python tools\record_cuda_replay.py `
    --num-envs 256 --warmup-drops 4 --record-drops 12 `
    --samples 3 --frame-stride 2
```

训练或模型评估代码也可以直接调用
`simulator.step_with_trace(actions, env_indices, frame_stride=2)`，只记录指定环境；
没有启用追踪时不会分配逐帧缓冲。

新版回放会内嵌 11 级水果贴图和 gzip 状态数据，生成后不需要服务器或外部资源。
播放器提供智能浏览、真实物理时间和逐次投放三种模式，并可按投放或合成事件跳转。
碰撞半径、角度、线速度和水果 ID 都可以独立开关；贴图按显示半径绘制，白色调试圈
始终表示追踪中的真实物理半径。

已有 `.pt` 或 `.pt.gz` 追踪无需重新运行 CUDA，可以直接升级为当前播放器。多个输入
会同时生成一个只按需加载当前回放的目录页：

```powershell
& $python tools\render_replay_trace.py `
    recordings\old-replays\env-1-full-episode.pt `
    recordings\old-replays\env-2-full-episode.pt `
    --output-dir recordings\rendered-replays
```

压缩播放器使用现代浏览器原生的 `DecompressionStream`。需要兼容不支持该接口的旧浏览器
时，重新渲染时增加 `--no-payload-compression`；文件会更大，但物理记录完全相同。

第一版 RL 基线配置仍显式使用 `score_delta / 66`，作为 Reward V2 的固定对照；
`configs/gnn_dqn_reward_v2.toml` 改用纯空间奖励。通用
`VectorEnv` 仍必须显式提供 `RewardComputer`，不能把游戏分数静默当成其它任务的奖励。

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

完整局会分别生成独立 HTML，并提供一个带筛选、排序和内嵌播放器的目录页；浏览器
始终只加载当前选择的一局，不再需要为 20 局打开 20 个标签页。只需查看终局前短片段
时，可以不传 `--replay-full-episodes`，再通过
`--replay-tail-drops` 控制尾段投放数。

## 分支来源

`codex/accelerated-v1` 从提交
`1a95a3d9a47bc6774144061f05ef067098b43544` 分叉。完整旧训练体系和 Android 成品
继续保留在原项目分支中，不作为本分支的运行依赖。
