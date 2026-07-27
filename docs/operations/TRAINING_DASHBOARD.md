# 训练监控面板运维手册

训练面板只读取既有训练、资源监控和阶段控制产物，用于远程查看 update、吞吐、
loss、episode、CPU、GPU、显存以及训练进程状态。它不是训练启动器，启动或停止面板
不会启动、停止或修改 DQN 训练。

## 1. 部署前检查

在云服务器项目根目录执行：

```bash
cd /root/autodl-tmp/fruit-merge-rl
test -f tools/training_dashboard.py
test -f scripts/training_dashboard.sh
conda run -n python-torch python -c \
  "import http.server, json, pathlib; print('stdlib ok')"
chmod +x scripts/training_dashboard.sh
```

面板后端只使用 Python 标准库，不需要 Flask、Django 或额外 Web 依赖。若标准库检查
失败，应先修复 `python-torch` 环境，不要改用来源不明的系统 Python。

## 2. 当前 V2 10k 示例

当前 10k 的训练目录为：

```text
runs/dqn_structure_v2_calibration_10k
```

```bash
RUN_DIR=runs/dqn_structure_v2_calibration_10k
MONITOR_DIR=runs/resource_monitor/v2_10k_ddcb249_20260728T0510
CONTROL_DIR=runs/stage_control/v2_10k_ddcb249_20260728T0510
```

先用 `test -d` 核对服务器上的实际目录，再启动：

```bash
test -d "${RUN_DIR}"
test -d "${MONITOR_DIR}"
test -d "${CONTROL_DIR}"
scripts/training_dashboard.sh start \
  --run-dir "${RUN_DIR}" \
  --monitor-dir "${MONITOR_DIR}" \
  --control-dir "${CONTROL_DIR}"
```

三个目录参数都可以省略，面板会自动发现最新产物。正式运维建议显式指定，避免服务器
同时保留多个历史 run 时看错对象。

## 3. 通过 SSH 隧道访问

面板默认只监听服务器回环地址 `127.0.0.1:8765`。在本地电脑建立 SSH 端口转发：

```bash
ssh -N -L 8765:127.0.0.1:8765 \
  -p 17899 root@connect.westb.seetacloud.com
```

OpenSSH 会自行提示认证；命令和本文都不保存密码。保持该 SSH 会话打开，然后在本地
浏览器访问：

```text
http://127.0.0.1:8765/
```

如本地 8765 已占用，可只修改本地端口，例如
`-L 18765:127.0.0.1:8765`，浏览器相应访问 `http://127.0.0.1:18765/`。

### Windows 一键桌面入口

Windows 本地电脑可以把认证、SSH 隧道和浏览器启动包装成一个桌面快捷应用。在项目
根目录使用 Windows PowerShell 5.1 运行一次安装器：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\windows\install_training_dashboard_shortcut.ps1
```

安装器只在首次安装时提示输入 SSH 密码，并完成以下操作：

- 使用 Windows DPAPI 以当前 Windows 用户身份加密密码；
- 把启动器、ASKPASS 辅助程序和图标安装到
  `%LOCALAPPDATA%\FruitMergeRL\TrainingDashboard`；
- 在当前用户桌面创建 `合成大西瓜训练面板.lnk`；
- 双击时隐藏建立仅绑定 `127.0.0.1` 的 SSH 隧道，确认面板 API 后打开默认浏览器；
- 已有由本入口为同一 SSH 目标建立的健康隧道时直接复用；默认端口被占用时依次尝试
  18765、28765，不复用未知面板，也不终止占用端口的其他程序。

密码不会进入仓库文件、PowerShell 脚本、快捷方式、SSH 命令行或日志。
`credential.bin` 只能由同一台电脑上的同一 Windows 用户通过 DPAPI 解密，并额外把
文件 ACL 限制为当前用户和 SYSTEM。它仍属于短期本地凭据：服务器释放后应卸载入口
或更新凭据。

服务器密码变化时重新录入：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\windows\install_training_dashboard_shortcut.ps1 `
  -UpdateCredential
```

卸载会安全核对保存的 SSH PID 与进程启动时刻，只结束该入口自己创建的隧道，然后
删除桌面快捷方式和本地安装目录：

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\windows\install_training_dashboard_shortcut.ps1 `
  -Uninstall
```

## 4. 生命周期操作

`start` 默认在后台启动，并把 PID 和日志分别保存到
`runs/dashboard/dashboard.pid`、`runs/dashboard/dashboard.log`：

```bash
scripts/training_dashboard.sh start
scripts/training_dashboard.sh status
scripts/training_dashboard.sh restart \
  --run-dir runs/dqn_structure_v2_calibration_10k
scripts/training_dashboard.sh stop
```

查看实时日志：

```bash
tail -f runs/dashboard/dashboard.log
```

可用参数：

```text
--run-dir PATH
--monitor-dir PATH
--control-dir PATH
--host HOST
--port PORT
--poll-history-limit N  # 10..5000，默认 600
--conda-env NAME
```

默认 Conda 环境为 `python-torch`，也可通过 `CONDA_ENV` 环境变量覆盖。除非已有受控
的反向代理、认证和防火墙策略，否则保持默认 `--host 127.0.0.1`，不要把面板直接
绑定到公网地址。

## 5. 安全边界

- 面板是只读观察入口，不提供训练 start、stop、resume 或删除产物的按钮。
- `stop` 不按进程名模糊杀进程。它同时校验 PID、Linux 进程启动时刻、命令行中的
  `tools/training_dashboard.py` 和项目工作目录；任一不一致都会拒绝发送信号。
- 面板通过独立的 `setsid` 进程组运行。停止时只向已经通过上述身份校验的面板进程组
  发信号，不会向 `train_dqn.py`、rollout worker 或资源监控进程发信号。
- PID 文件指向存活但身份不符的进程时，不要直接执行 `kill` 或删除 PID 文件后重试；
  先用 `ps -fp <PID>`、`tr '\0' ' ' </proc/<PID>/cmdline` 人工确认。
- 仓库文档和脚本不保存明文 SSH 密码。手动隧道继续交给 OpenSSH、密钥或云平台的
  安全入口；可选 Windows 桌面入口只在本机保存当前用户 DPAPI 加密后的
  `credential.bin`。
- `/api/health` 同时返回服务状态和数据新鲜度；训练/监控心跳超过 30 秒未更新时，
  页面会标记为 stale，不会继续把旧的进程计数显示成“正在运行”。

## 6. 故障排查

面板启动失败时先检查：

```bash
scripts/training_dashboard.sh status
tail -n 100 runs/dashboard/dashboard.log
conda run -n python-torch python tools/training_dashboard.py --help
```

常见情况：

- `找不到 conda`：先初始化当前 shell 的 Conda，或使用安装 Conda 后的登录 shell。
- 缺少 Python 依赖：在 `python-torch` 环境中安装项目要求的依赖，再重新启动。
- 页面能打开但没有数据：确认 `--run-dir` 下至少已有 `metrics.csv` 或
  `episode_metrics.csv`，并核对 monitor/control 路径没有指向旧 run。
- 本地浏览器无法连接：确认面板 `status` 为 running、SSH `-L` 会话仍在运行，并检查
  本地端口是否被占用；不应通过改成 `0.0.0.0` 绕过隧道问题。
- Windows 桌面入口提示认证失败：重新运行安装器并加 `-UpdateCredential`；启动器只
  尝试一次密码，不会循环锁定云账号。
- Windows 桌面入口提示主机指纹变化：不要删除 `known_hosts` 或关闭主机指纹校验；
  先从云平台核对服务器地址和新指纹，再人工更新本机记录。
- Windows 双击后没有页面：查看
  `%LOCALAPPDATA%\FruitMergeRL\TrainingDashboard\launcher.log` 和 `ssh.log`。
  日志不含密码；若隧道 PID 仍存活但面板无响应，应先在云端检查 dashboard 服务。
- `unsafe-pid-record`：PID 被复用、元数据缺失或文件被人工修改。先核实该 PID 的真实
  命令；只有确认进程已经不存在后，才清理 `runs/dashboard/dashboard.pid` 和
  `dashboard.start_ticks` 并重新启动。
- 指定目录不存在：面板会保留页面并显示数据源错误；修正目录参数后执行
  `restart`。`restart` 不会沿用上次命令参数，需重新传入需要固定的目录。
