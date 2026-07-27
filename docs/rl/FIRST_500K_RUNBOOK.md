# 第一次 500k 完整因果训练运行手册

状态：首轮长训前执行基线

适用分支：`codex/work-1`

最后更新：2026-07-27

## 1. 目标与边界

本手册用于把同一套冻结语义依次跑过：

```text
静态门禁
-> 5k 冒烟
-> 10k 校准（必要时另起一个从 update 0 开始的 25k 校准）
-> 500k 首轮正式训练
```

三份训练配置都继承 `configs/train_dqn_fast30_parallel.toml`。5k 和 10k 只缩短
训练长度并提高日志、保存和评估频率，不关闭 Reward V2、Double DQN、3-step、
完整状态归因、规则排序、稀疏物理反事实或局部 Shapley。短跑的用途是排除实现和
数值错误，不是重新决定是否启用完整归因。

不要在 5k 和 10k 门禁完成前启动 500k。当前进度和证据以
`docs/training_runs/FIRST_500K_READINESS.md` 为准。

## 2. 无秘密部署原则

- 仓库凭据、云平台密钥和对象存储令牌不写入命令、TOML、`config.json`、shell
  历史或本文档。私有仓库使用云主机已经配置好的只读 SSH key，或在本地打包后传输。
- 训练本身不需要网络服务密钥。正式启动前可以断开不必要的外网权限。
- 只从可信来源恢复 `.pt`。PyTorch checkpoint 使用 pickle 反序列化，不能加载
  来历不明的文件。
- `runs/` 不提交 Git。需要迁移时单独复制 checkpoint、CSV、JSON 和必要日志。

### 2.1 保留 Git 身份的离线迁移

正式 preflight 会校验 commit、分支和工作树是否干净，因此云端训练根目录必须保留
完整 `.git` 元数据。优先在云端从可信远端克隆 `codex/work-1`；不能访问远端时，在
干净工作树中生成 Git bundle：

```bash
git status --short
git bundle create fruit-merge-rl.bundle codex/work-1
git bundle verify fruit-merge-rl.bundle
sha256sum fruit-merge-rl.bundle
```

通过已经批准的安全传输通道上传 bundle 和单独记录的 SHA-256，在云端先核对哈希，
再克隆和复核身份：

```bash
sha256sum -c fruit-merge-rl.bundle.sha256
git clone --branch codex/work-1 fruit-merge-rl.bundle fruit-merge-rl
cd fruit-merge-rl
git rev-parse HEAD
git status --short
```

普通 ZIP/源码归档没有 `.git`，不能作为正式 5k、10k 或 500k 的项目根目录；否则
preflight 无法证明 commit 与 `dirty=false`。ZIP 只可用于人工阅读或附带非 Git
产物。每次 HEAD 变化后都必须重新生成、验证并计算 bundle 哈希，不能复用旧包。

## 3. Ubuntu 云服务器安装

### 3.1 系统前置条件

建议准备：

- Ubuntu 22.04 或更新版本；
- Python 3.11；
- NVIDIA GPU 和能支持 CUDA 12.6 PyTorch wheel 的驱动；
- 至少 80 GiB 可用磁盘空间；因果 replay 精确 checkpoint 与 100k 冷 TD replay
  会同时占盘，不能只按模型权重估算；
- 至少 24 GiB 内存，建议 32 GiB 或更多；3 个 rollout worker、20k 因果 replay、
  8k 热 TD replay 和 2k 冷缓存会并存；
- 至少 6 个经 affinity/cgroup 限制后仍实际可用的 CPU 逻辑核。正式配置固定使用
  3 个 rollout worker 和 2 个反事实/局部 Shapley 共享物理 worker，并为主进程
  保留 1 核；preflight 按有效核数而不是宿主机名义核数检查。

安装基础工具：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential
```

如果服务器尚无 Conda，可以安装 Miniconda；已有 Conda 时跳过：

```bash
curl -fsSLo /tmp/miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

创建隔离环境：

```bash
conda create -n python-torch python=3.11 pip -y
conda activate python-torch
python -m pip install --upgrade pip
```

进入已经安全传输或克隆的项目根目录，然后安装固定依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install matplotlib==3.11.1
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip check
```

`requirements-training.txt` 记录训练依赖的冻结版本；上面把 PyTorch 单独安装，是为了
明确使用官方 CUDA 12.6 wheel，避免普通 PyPI 解析意外得到 CPU 构建。首轮云端 GPU
是 compute capability 7.0 的 V100；实测 2.12.1+cu130 只包含 `sm_75` 及以上内核，
虽然 `torch.cuda.is_available()` 为真，真实运算仍会报
`cudaErrorNoKernelImageForDevice`。因此首轮环境契约固定为 2.12.1+cu126，而不是
cu130。若云服务器驱动不支持该 wheel，不要临时改成 CPU 构建；先升级驱动并重新执行
本节验证。

## 4. CUDA 与依赖验证

先检查驱动和 GPU：

```bash
nvidia-smi
```

再检查实际 Python 环境：

```bash
conda run --no-capture-output -n python-torch python -c \
  "import torch; print('torch=', torch.__version__); print('cuda_runtime=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); assert torch.cuda.is_available(); x=torch.randn(1024,1024,device='cuda'); print('probe=', float((x@x).mean()))"

conda run --no-capture-output -n python-torch python -c \
  "import pymunk; print('pymunk=', pymunk.version); print('chipmunk=', pymunk.chipmunk_version); assert pymunk.version == '7.3.0'; assert pymunk.chipmunk_version == '2.0.1-ade7ed72849e60289eefb7a41e79ae6322fefaf3'"
```

预期关键值：

```text
torch = 2.12.1+cu126
cuda_runtime = 12.6
available = True
pymunk = 7.3.0
chipmunk = 2.0.1-ade7ed72849e60289eefb7a41e79ae6322fefaf3
```

不要只根据 `nvidia-smi` 判断可训练；必须让当前 Conda 环境实际完成 CUDA 张量运算。

## 5. 代码与静态测试门禁

确认当前源码身份：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

正式 500k 必须记录明确的 commit，并保持工作树干净。然后执行：

```bash
PYTHONPATH=src:tests conda run --no-capture-output -n python-torch \
  python -m unittest discover -s tests -p "test_*.py" -v

conda run --no-capture-output -n python-torch \
  python -m compileall -q src tools tests
```

任一测试失败都不允许进入 5k。测试数会随代码变化，因此以“零失败、零错误”为硬门禁，
并把当次总数、commit 和命令写入 readiness 记录。

## 6. 正式 preflight

preflight 不启动训练，但会验证正式 500k 解析后的配置、CUDA 前后向、完整因果
optimizer step、物理反事实、局部 Shapley、32 次 EngineSnapshot 原动作复现、磁盘和
CPU 余量：

```bash
conda run --no-capture-output -n python-torch \
  python tools/preflight_training.py \
  --config configs/train_dqn_causal_500k.toml \
  --output runs/preflight/first_500k_pre_smoke.json \
  --snapshot-audits 32 \
  --min-free-gb 80 \
  --min-memory-gb 24 \
  --min-available-memory-gb 8
```

允许进入短跑的条件：

- 进程返回码为 0；
- JSON 中 `ready=true`；
- `required_failures=[]`；
- `warnings=[]`，或每条 warning 已有人为解释；
- `engine_snapshot_reproduction.matches=32`；
- `formal_500k_config_contract` 通过，确认输入确实是冻结的 500k/CUDA/3 env/
  9 steps per update/3-step/完整因果与 Shapley 配置，而不是误传 5k 或关闭归因的
  配置；
- `full_causal_optimizer_step` 和 `local_shapley_physical` 均通过；
- cgroup 可用物理内存上限至少 24 GiB，启动时可用内存至少 8 GiB；
- JSON 中 commit 与准备运行的源码 commit 一致，`dirty=false`。

preflight 通过只表示静态和小规模物理链路闭合，不等于 500k 已经获准启动。

## 7. 启动旁路资源监控

在独立终端先启动 Linux 旁路监控器：

```bash
conda run --no-capture-output -n python-torch \
  python -u tools/monitor_training_resources.py \
  --interval 3 \
  --stop-after-target-exits \
  --warn-mem-available-mb 4096 \
  --warn-swap-used-mb 1024 \
  --warn-target-rss-mb 16384 \
  --warn-gpu-memory-used-mb 7600
```

监控输出默认位于 `runs/resource_monitor/<时间戳>/`。它与训练进程隔离，即使训练被
OOM killer 终止，也能留下崩溃前的 CPU、内存、GPU 和进程日志。

每个短跑与正式 run 都必须独占一个监控目录。阶段结束时至少核对：

- `events.jsonl` 出现 `target_process_started` 和 `target_exited_stop_requested`；
- `process_metrics.csv` 至少包含一条目标训练进程记录；
- `summary.json` 的 `samples > 0`、`min_mem_available_mb >= 4096`、
  `peak_swap_used_mb <= 1024`、`peak_target_rss_mb <= 16384`；
- `gpu_metrics.csv` 存在有效 GPU 样本，显存没有连续 3 次超过总量 95%；
- `events.jsonl` 没有在结束时仍未恢复的 `low_system_memory`、
  `swap_usage_high`、`target_rss_high`、`gpu_memory_high` 或
  `gpu_query_failed`。

训练目录内的 readiness 报告不会自动读取这个旁路目录，因此即使其
`ready=true`，也必须把上述结果和监控目录路径人工登记到 readiness 记录；两部分
证据缺一不可。

另开一个终端启动训练。启动器会设置 `PYTHONPATH`、使用 `python-torch` Conda 环境，
并把终端输出同时写入 `runs/launcher_logs/`：

```bash
chmod +x scripts/train_dqn.sh
```

## 8. 5k 冒烟

```bash
CONDA_ENV=python-torch \
  ./scripts/train_dqn.sh configs/train_dqn_causal_smoke_5k.toml
```

输出目录固定为：

```text
runs/dqn_causal_smoke_5k/
```

若该目录已经非空，训练入口会拒绝静默覆盖。不要为图省事使用
`--overwrite-run-dir` 覆盖有价值的结果；应先归档目录，或显式换一个新的
`--run-dir`。

5k 只检查：

- 无崩溃、NaN/Inf、死锁、ID 串线和重复预算记账；
- 主 TD、规则排序和反事实监督都能进入真实 optimizer step；
- 规则因果 replay 同时出现正、负样本；
- 物理反事实存在可复现且可生成标签的结果；
- 预算始终不超过 10% 硬上限；
- checkpoint、CSV、曲线和 shutdown sidecar 能完整落盘。

模型分数在 5k 时没有否决权。局部 Shapley 的累计上限只有 0.05%，5k 中没有选中
任务也不自动判失败；其物理实现先由 preflight 保证，真实选择率在 10k/25k 校准。

## 9. 10k 校准与可选独立 25k 校准

5k 通过后，新启动 10k：

```bash
CONDA_ENV=python-torch \
  ./scripts/train_dqn.sh configs/train_dqn_causal_calibration_10k.toml
```

输出目录：

```text
runs/dqn_causal_calibration_10k/
```

10k 除了稳定性，还要校准：

- shaping p95 是否处于任务奖励的次要尺度；
- 归因事件数量、正负样本比例、确认/取消/中断分布；
- 规则 loss、反事实 loss 与 TD loss 的数量级；
- 反事实复现率、标签产量、实际 token 比例和队列积压；
- 局部 Shapley 的选择、物理复现、效率门和样本落盘；
- `env_steps_per_second`、内存、显存和 checkpoint 体积是否可承受 500k。

若 10k 结束时只有 Shapley 的真实样本量不足，而 selector、预算和 preflight 均正常，
可以另起一个从 update 0 开始的 25k 校准 run：

```bash
CONDA_ENV=python-torch \
  ./scripts/train_dqn.sh \
  configs/train_dqn_causal_calibration_10k.toml \
  --run-dir runs/dqn_causal_calibration_25k \
  --total-updates 25000
```

这条命令故意不带 `--resume`，是为了得到从 update 0 开始、探索曲线一致且便于横向
比较的 25k 校准证据。训练入口现在会把首次 checkpoint 的
`epsilon_schedule_total_updates` 冻结下来；若技术上把 10k checkpoint 延长到 25k，
探索率会沿用原 10k horizon，已经降到的低值不会重新放大。此时
`resume_<时间戳>.json` 必须记录
`epsilon_schedule_extended_without_reexpansion=true`。这种 run 应明确标记为
“10k 后连续延长”，不能与从零开始的独立 25k 混为同一校准证据。

不要为了制造 Shapley 样本提高正式的 `shapley_event_ratio_max`。若需要强制触发验证，
只能另开明确标记的诊断 run，不得把其指标混入正式校准。

## 10. 500k 正式启动

只有 readiness 文档中所有硬门禁均为通过，且正式配置 commit、preflight commit 和
准备启动的 commit 完全一致时，才执行：

```bash
CONDA_ENV=python-torch \
  ./scripts/train_dqn.sh configs/train_dqn_causal_500k.toml
```

正式输出目录由基配置冻结为：

```text
runs/dqn_causal_fast30_h128_l3_n3_500k/
```

启动后立即保存以下身份信息到运行记录：

- 完整 commit SHA 和分支；
- `runs/.../config.json` 中的训练、StateAnalyzer、反事实和 Shapley 指纹；
- preflight JSON 路径及 SHA-256；
- launcher log 和 resource monitor 目录；
- GPU 型号、驱动、PyTorch/CUDA、Pymunk/Chipmunk 版本；
- 启动时间和操作者。

## 11. 训练中应监控的指标

核心文件：

```text
metrics.csv
episode_metrics.csv
attribution_warmup.json
attribution_shutdown.json
counterfactual_shutdown.json
checkpoints/latest.pt
plots/training_curves.png
failure_latest.json                 # 仅异常时出现
```

每个日志窗口至少检查：

| 维度 | 关键字段 | 正常含义 |
| --- | --- | --- |
| 主训练 | `loss`、`td_loss`、`mean_q`、`mean_target`、`grad_norm` | 全部有限，长期不单向爆炸 |
| 因果训练 | `rule_rank_loss`、`counterfactual_loss`、各 batch size | 非零样本到来后能进入更新 |
| 因果 replay | 正/负、rule/cf/Shapley 数量和 cause type | 不被单一类别永久饿死 |
| 状态分析 | degraded rate、shaping p95、cache hit | 降级稀少，shaping 不压过任务效用 |
| 反事实 | admitted/completed/failed、reproduction、samples | 失败被丢弃，不产生伪标签 |
| 预算 | actual/projected token ratio、hard budget respected | 共享硬预算始终成立 |
| Shapley | observed/selected/completed/reproduced/samples | 极稀疏、可复现、效率门通过 |
| 性能 | updates/s、env steps/s、各阶段 seconds | 无持续退化或队列死锁 |
| 游戏表现 | episode score、terminated/truncated | 以真实 score 解释，不比较旧 shaped reward |

Reward V2 的 5 级合成效用为 `2**1.5 = 2.828`。设计门禁要求单步 shaping 绝对值
p95 不超过其 25%，即约 `0.707`。

## 12. 停止与暂停阈值

### 12.1 立即停止

出现任一项，停止当前 run 并保留现场，不继续等待“自行恢复”：

- `failure_latest.json` 出现，或终端报出 NaN、Inf、非有限梯度/target；
- `counterfactual_hard_budget_respected=0`，或实际 token 比例超过 `0.10`；
- 训练进程仍在但 10 分钟没有心跳、`update_step` 和 `env_steps` 都无变化，且不是
  已知的 checkpoint/评估阶段；
- checkpoint 写入失败、输出盘可用空间低于 40 GiB，或系统发生 OOM/Xid；
- `metrics.csv` 的 update/env step 倒退，或 episode/worker 身份冲突；
- 反事实/局部 Shapley 向 replay 写入了未通过原动作复现门的标签。

### 12.2 暂停并审查

以下项目允许偶发，但连续 3 个日志窗口或达到统计样本下限后仍存在，就不能进入下一阶段：

- shaping p95 大于 `0.707`；
- StateAnalyzer degraded rate 大于 `1%`；
- 至少 50 局后 truncated 比例大于 `2%`；
- `abs(mean_q)`、`abs(mean_target)` 或 mean absolute TD error 大于 `100`；
- 反事实累计完成至少 100 个任务后，原动作复现失败率大于 `1%`；
- `counterfactual_circuit_open=1`，或 snapshot failure 非零且持续增加；
- counterfactual pending 达到队列上限并持续 10 分钟没有完成结果；
- pending 归因事件持续增长，同时至少 5 个窗口没有确认或取消事件；
- 内存低于 4 GiB、swap 持续增长，或 GPU 显存连续 3 次超过 95%。

这些阈值是“停止扩大训练规模”的工程门禁，不是自动修改奖励权重的依据。任何阈值调整都
必须写入 readiness 记录，并在启动 500k 前重新冻结配置和 preflight。

### 12.3 阶段结束的最低信号

5k 结束至少应看到：

- TD 更新完成 5000 次且无异常；
- 因果 replay 有规则正、负样本；
- 至少一次规则 batch 和一次物理反事实 batch 实际进入 loss；
- 反事实硬预算始终成立；
- 最终 checkpoint 和 shutdown sidecar 均存在。

10k/25k 结束还应看到：

- 反事实有完成、复现通过和样本写入，不只是 proposal 被预算拒绝；
- 局部 Shapley 若被 selector 选中，则任务必须复现并通过效率门；若仍未选中，必须用
  observed/eligible/quota/drop reason 解释，并决定是否另起独立 25k 校准；
- 吞吐、内存和 checkpoint 保存时间可以外推到 500k；
- 所有暂停阈值均已排除或有书面解释。

每个阶段正常结束后运行只读门禁，不手工挑选中间行：

```bash
conda run --no-capture-output -n python-torch \
  python tools/check_training_readiness.py \
  --run-dir runs/dqn_causal_smoke_5k \
  --stage 5k \
  --output runs/preflight/dqn_causal_smoke_5k_readiness.json

conda run --no-capture-output -n python-torch \
  python tools/check_training_readiness.py \
  --run-dir runs/dqn_causal_calibration_10k \
  --stage 10k \
  --output runs/preflight/dqn_causal_calibration_10k_readiness.json
```

门禁要求配置/manifest/replay/RNG/模型 optimizer 可真实恢复，按 resume sidecar 分段
核对累计反事实与 Shapley 账本，并检查最终 shutdown 已排空。报告自身不读取旁路
resource monitor 的主机 RSS/GPU 峰值；这些仍必须从第 7 节监控目录人工补入
readiness 记录。

## 13. 安全停止方法

训练入口会定期原子更新 `checkpoints/latest.pt`。准备人工停止时：

1. 先确认最近一次 `latest.pt` 已完成写入且文件大小稳定；
2. 在训练终端发送一次 `Ctrl+C`；
3. 等待 worker、反事实、Shapley、replay 和日志清理结束；
4. 保存 launcher log、resource monitor 和 `failure_latest.json`；
5. 不使用 `kill -9`，除非进程已经无法响应且系统安全受到影响。

当前 `Ctrl+C` 不承诺在中断瞬间新建 checkpoint；恢复点是最后一次成功的周期
`latest.pt`。非有限数值触发的 fail-fast 会额外尝试保存
`failure_last_normal.pt`，但它仍只应作为诊断/恢复候选，先检查
`failure_latest.json`。

## 14. Checkpoint 恢复语义

从可信的同一 run checkpoint 恢复：

```bash
CONDA_ENV=python-torch \
  ./scripts/train_dqn.sh \
  configs/train_dqn_causal_500k.toml \
  --resume runs/dqn_causal_fast30_h128_l3_n3_500k/checkpoints/latest.pt
```

manifest 允许增加 `--total-updates`。`smooth` epsilon 使用 checkpoint 首次创建时
冻结的 `epsilon_schedule_total_updates`，因此延长总步数不会把已经下降的探索率突然
抬高；`resume_<时间戳>.json` 会同时记录冻结 horizon、请求总步数和
`epsilon_schedule_extended_without_reexpansion`。10k→25k 仍优先按第 9 节从
update 0 独立启动，以获得可比较的校准曲线；从 10k 连续延长是允许的不同证据类型。

日志、保存、评估、绘图和进度频率中被 manifest 明确列为 mutable 的字段可以改变。
模型结构、seed、Reward、DQN、物理、归因、反事实、Shapley、replay 容量和设备等
训练语义不能改变；指纹不一致会在日志打开前拒绝恢复。`--overwrite-run-dir` 不能与
`--resume` 同时使用。

如果没有显式传 `--run-dir`，入口会从
`<run>/checkpoints/latest.pt` 推导原 run 目录；若显式传入，则必须正好是 checkpoint
所属目录，不能把一个 run 的状态写进另一个 run。

恢复内容：

- online/target 模型、Adam optimizer 和 trainer update step；
- env step、epsilon、最好评估值和最新指标；
- Python、PyTorch CPU 和全部 CUDA RNG；CUDA 设备数不一致会拒绝恢复；
- `CausalReplayBuffer` 的样本、游标、统计和独立 RNG，精确恢复；
- 正式 hybrid TD replay 的最近 hot layer（当前最多 8000 条）和 replay RNG。

有意不恢复：

- hybrid TD replay 已落盘的旧 cold layer；checkpoint 会明确记录
  `resume_policy=hot_only` 和 `omitted_cold_count`；
- rollout worker 内正在进行的物理轨迹、pending n-step 尾部、正在排队的物理
  反事实和 Shapley 任务。

因此恢复不是逐物理帧的 bit-for-bit 续跑。worker 会从新的 episode 边界开始，并使用
历史 env step 之后的 episode ID 防止和已恢复因果样本串线；如果 hot replay 不足一个
batch，入口会执行 `resume_warmup` 补足。每次恢复会生成：

```text
resume_config_<时间戳>.json
resume_<时间戳>.json
attribution_resume_warmup_<update>.json   # 需要补 warmup 时
```

若 CSV 中存在晚于 checkpoint 的行，恢复器会将其移到
`*.orphaned_after_<update>_<时间戳>.csv`，再从 checkpoint 对应 update 继续追加，避免
日志看似前进而模型实际回退。

异常时会同时写带时间戳的 `failure_<时间戳>.json` 和活动指针
`failure_latest.json`。成功恢复并完整保存最终 checkpoint 与曲线后，入口只自动
删除活动指针，保留历史诊断。历史失败由 readiness 记录为告警；当前仍存在
`failure_latest.json` 才是硬失败，不要人工删除它来伪造成功。

## 15. 训练结束与归档

每个阶段结束后：

1. 记录退出码，确认没有 `failure_latest.json`；
2. 检查 `latest.pt`、最终 metrics、episode metrics、曲线和 shutdown sidecar；
3. 从 CSV 计算 readiness 表中的门禁指标，不复制终端滚动均值充当最终统计；
4. 保存 config 指纹、commit 和 preflight 身份；
5. 更新 `docs/training_runs/FIRST_500K_READINESS.md`；
6. 5k/10k 结果通过后再进入下一阶段，500k 结果另建正式训练总结。

大型 checkpoint、cold replay 和原始监控日志继续保留在 `runs/` 或外部训练存储中；
Git 只保存配置、运行手册、轻量结论和必要的结构化摘要。
