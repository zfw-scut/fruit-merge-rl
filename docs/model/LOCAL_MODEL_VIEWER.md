# 本地 GNN-DQN 模型观看器

- 状态：第一版已实现并通过本机 CUDA smoke checkpoint 验证
- 用途：下载云端 checkpoint 后，在本地浏览器游戏页面观看模型的真实物理游玩
- 默认：120 FPS、greedy、单局、完整 Q 值和逐帧运动记录

## 1. 工作边界

观看器从 checkpoint 内的冻结 `ModelConfig` 重建当前 `BaselineGnnDqn`，只加载
`learner.online_model` 权重。模型每次只在模拟器完成一次投放等待后决策，CUDA 模拟器
本身负责连续稳定窗口、等待上限、正式终局和下一次投放语义。

第一版采用“先在本地运行真实模型和物理，再在浏览器按物理时间播放完整记录”的方式。
浏览器不重新计算模型或物理，也不是 WebGPU/ONNX 运行时；因此页面可离线保存、复制和
复查，同一 seed 的结果不受播放速度影响。后续若需要人类实时对战，再在此稳定契约上
增加流式服务，不改变 checkpoint 和模型输入。

## 2. 与旧分支的复用边界

保留旧 `watch_dqn.py` 已验证的三项原则：

1. checkpoint 决定模型结构，观看器不得猜测隐藏维度；
2. 模型只在稳定决策边界执行 greedy 推理；
3. 画面控制层不进入训练包，也不修改 checkpoint。

旧版 pygame `Board`、动态 `GraphBuilder`、`StateAnalyzer` 和旧 checkpoint 提取器没有
迁入。当前实现直接使用定长 `TensorState`、新 GNN 和 CUDA 逐帧追踪，不依赖旧分支
运行时。

## 3. 使用方法

```powershell
$python = 'C:\Users\yefan\miniconda3\envs\python-torch\python.exe'
& $python tools\watch_gnn_dqn.py `
    runs\gnn_dqn_baseline\checkpoints\best.pt `
    --device cuda --physics-fps 120 --open
```

常用选项：

| 参数 | 含义 |
| --- | --- |
| `--physics-fps 120` | 默认精确评估物理 |
| `--physics-fps 30` | 训练物理对照，不产生训练数据 |
| `--episodes N` | 连续生成 N 个独立 seed 的完整页面和目录 |
| `--max-drops N` | 未终局时的观看投放上限 |
| `--frame-stride N` | 每 N 个物理帧保留一个采样，终止帧始终保留 |
| `--output-dir PATH` | 页面与追踪输出目录 |
| `--no-trace` | 只保留 HTML，不保存可重新渲染的 PT.GZ |
| `--open` | 完成后使用系统默认浏览器打开 |

页面显示游戏分数、水果数、合成、终局状态、真实投放位置、q0～q3、21 个动作 Q 值、
选定 Q、危险进度与推理耗时。水果贴图使用显示半径，白色调试圈继续表示真实碰撞半径。

## 4. 当前限制

- 完整逐帧追踪使用 CUDA 扩展；第一版观看命令要求本机可用 CUDA GPU；
- 页面展示的是已经真实运行完成的对局，不在浏览器中实时执行 PyTorch；
- 本地 smoke checkpoint 只验证加载和显示链路，不能用于评价策略水平；
- 正式比较仍以固定种子的 30/120 FPS 批量 greedy 评估为准，观看页面只作定性分析。
