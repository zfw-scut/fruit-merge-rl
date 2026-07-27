# 合成大西瓜与完整因果归因训练

这是一个基于 `pygame`、`pymunk` 和 PyTorch 的《合成大西瓜》项目。仓库同时包含：

- 可直接游玩的 pygame 游戏本体；
- 无渲染物理环境、GNN-Q 模型和并行 rollout；
- 面向首轮大规模训练的 Reward V2、Double DQN + 3-step return；
- 完整状态分析、历史贡献归因、独立 `CausalReplayBuffer`、预算反事实和极稀疏局部 Shapley。
- 显式支撑/遮挡/可达关系、连锁 motif、关系感知 dueling GNN、六维结构辅助监督和
  集中式 GPU actor。

游戏本体位于 `daxigua`，训练代码位于 `daxigua_rl`；前者不得反向依赖后者。旧实验代码已经移除，当前 RL 主链路是重新设计后的实现。

当前手动游戏窗口为固定尺寸 `400x800`，顶部独立信息层会显示分数和待投放水果队列，方便提前规划投放顺序。

## 运行手动游戏

先安装依赖：

```bash
python -m pip install -r requirements.txt
```

启动游戏：

```bash
python Main.py
```

## 操作

- 鼠标移动：调整当前水果的投放位置。
- 鼠标左键：投放水果。
- `A` / `Left`：向左调整投放位置。
- `D` / `Right`：向右调整投放位置。
- `Space` / `Enter`：投放水果。
- `R`：重新开始。
- `Esc`：退出。

顶部 `QUEUE` 区域从左到右显示当前水果和后续 3 颗水果；每次投放后队列会向前推进，并在末尾补充新水果。该区域和当前悬浮水果处于不同高度层，避免移动水果时遮挡队列。

## 训练准备

当前正式服务器是 RTX 5090，训练环境固定为 PyTorch 2.12.1 + CUDA runtime
13.0。依赖版本记录在 `requirements-training.txt`，CUDA wheel 从官方 cu130 源
安装：

```bash
python -m pip install -r requirements.txt
python -m pip install matplotlib==3.11.1
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip check
```

大规模训练前应先运行门禁。它会验证正式结构感知配置、CUDA 前后向、六维结构
optimizer step、固定 Pymunk/Chipmunk 版本、`EngineSnapshot` 原动作重演、完整因果
optimizer step、局部 Shapley 物理链路、磁盘与 CPU 余量，并真实执行一次 16-worker
集中式 CUDA actor 的同步/异步采集与关闭链路，但不会启动持续训练：

```bash
python tools/preflight_training.py \
  --config configs/train_dqn_causal_500k.toml \
  --min-free-gb 80 \
  --min-memory-gb 64 \
  --min-available-memory-gb 16
```

当前提供三套同源配置：

- `configs/train_dqn_causal_smoke_5k.toml`：5000 次更新烟测；
- `configs/train_dqn_causal_calibration_10k.toml`：10000 次更新标定；样本不足时从
  update 0 另起独立 25000 次更新校准是可比性首选；也可连续延长，但会保持
  checkpoint 冻结的 epsilon horizon，并必须单独标记；
- `configs/train_dqn_causal_500k.toml`：第一次 500000 次更新正式训练。

当前三套配置属于结构感知 V2：正式基线采用 H256/L4、batch 128、16 个 rollout
worker、6 个反事实物理 worker 和集中式 GPU actor，并以
`lambda_structural=0.15` 训练六维一步结构结果。旧 H128/L3 checkpoint、旧 hot/cold
replay 和旧 causal replay 不能续训新架构；三个 V2 阶段都必须使用空目录从 update 0
开始。完整设计与兼容表见
`docs/rl/STRUCTURE_AWARE_GNN_V2.md`。

完整安装、阶段门禁、监控、停止阈值和恢复流程见
`docs/rl/FIRST_500K_RUNBOOK.md`；当前阶段证据只以
`docs/training_runs/FIRST_500K_READINESS.md` 为准。

训练入口：

```bash
PYTHONPATH=src python -u -m daxigua_rl.scripts.train_dqn \
  --config configs/train_dqn_causal_smoke_5k.toml
```

从版本化 checkpoint 恢复时使用 `--resume`，但只允许恢复同一 V2 run 的可信
checkpoint。正式 hybrid replay 采用明确记录的 hot-resume：恢复模型、target、
optimizer、更新计数、RNG、因果 replay 和主 replay 热层，不宣称恢复已经省略的
冷层。

## 同步云端基础训练数据

查看云端曲线时不需要下载大型 checkpoint/replay。阶段结束后运行：

```bash
python tools/sync_cloud_training_artifacts.py \
  --host <SSH主机> --port <SSH端口> --user <SSH用户> \
  --remote-run-dir /absolute/path/to/runs/<run_id> \
  --local-dir runs/cloud_evidence/<本地名称> \
  --require-complete
```

工具只同步标准配置/指标/归因 JSON 白名单和三张 `plots/*.png`，校验后写入
`sync_manifest.json`；认证由 OpenSSH 的密钥、agent 或密码提示负责。训练中途同步时
去掉 `--require-complete`。完整说明见
`docs/rl/FIRST_500K_RUNBOOK.md` 的“把基础分析数据同步回本地”。

## 实时查看云端训练

云服务器可运行只读训练面板，汇总训练 update、预计剩余时间、loss、episode、
CPU/GPU/显存和训练生成的评估曲线。面板只读取既有产物，不提供启动、停止、恢复或
修改训练的控制接口。

在服务器项目根目录启动：

```bash
scripts/training_dashboard.sh start \
  --run-dir runs/<run_id> \
  --monitor-dir runs/resource_monitor/<monitor_id> \
  --control-dir runs/stage_control/<control_id>
```

服务默认只监听服务器的 `127.0.0.1:8765`。在本地电脑建立 SSH 隧道：

```bash
ssh -N -L 8765:127.0.0.1:8765 \
  -p <SSH端口> <SSH用户>@<服务器地址>
```

保持隧道连接后访问 `http://127.0.0.1:8765/`。脚本也支持自动发现最新训练产物；
服务器保留多个实验时建议显式指定三个目录。启动、停止、状态检查、安全边界和排障
说明见 `docs/operations/TRAINING_DASHBOARD.md`。

Windows 本地电脑可运行
`scripts/windows/install_training_dashboard_shortcut.ps1`，创建带项目图标的
`合成大西瓜训练面板` 桌面入口。双击后会自动认证、建立回环 SSH 隧道并打开浏览器。
仓库和快捷方式不保存明文密码；可选的一键入口只在本机以当前 Windows 用户 DPAPI
加密保存凭据。

## 项目说明

- `Main.py`: 兼容旧启动方式的薄入口。
- `src/daxigua/`: 游戏本体包。
- `src/daxigua/app.py`: 游戏应用入口和当前表现层实现。
- `src/daxigua/core/`: 游戏核心逻辑，负责物理世界、边界、碰撞合成、计分和水果定义。
- `src/daxigua/core/engine.py`: 无渲染游戏引擎和可校验、可恢复的 `EngineSnapshot`。
- `src/daxigua_rl/`: 环境、结构关系图、关系感知 GNN、Reward V2、因果归因、
  反事实和训练主链路。
- `requirements-training.txt`: 训练侧 PyTorch 和绘图依赖版本。
- `tools/preflight_training.py`: 第一次大规模训练前门禁。
- `tools/sync_cloud_training_artifacts.py`: 只读同步云端轻量指标和曲线。
- `tools/training_dashboard.py`: 只读汇总训练进度、资源状态和评估曲线的 HTTP 服务。
- `scripts/training_dashboard.sh`: 云端面板的安全启动、停止、重启和状态检查入口。
- `scripts/windows/`: Windows 训练面板桌面入口的安装、自动隧道和 ASKPASS 源码。
- `configs/`: 烟测、标定和正式训练配置。
- `assets/fruits/`: 水果图片资源。
- `assets/dashboard/`: Windows 训练面板桌面入口的 PNG 源图和多尺寸 ICO 图标。
- `assets/fruits.zip`: 原始水果图片压缩包归档。
- `docs/`: 项目文档和 Codex 修改记录。
