#!/usr/bin/env bash
set -euo pipefail

# 训练面板的生命周期只由这个薄启动器管理。PID 校验同时检查启动时刻、命令行和
# 工作目录，避免陈旧 PID 被复用后误伤正在运行的训练进程。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND="${PROJECT_ROOT}/tools/training_dashboard.py"
DASHBOARD_DIR="${PROJECT_ROOT}/runs/dashboard"
PID_FILE="${DASHBOARD_DIR}/dashboard.pid"
START_FILE="${DASHBOARD_DIR}/dashboard.start_ticks"
LOG_FILE="${DASHBOARD_DIR}/dashboard.log"

usage() {
  cat <<'EOF'
用法：
  scripts/training_dashboard.sh start [选项]
  scripts/training_dashboard.sh stop
  scripts/training_dashboard.sh status
  scripts/training_dashboard.sh restart [选项]

start/restart 选项：
  --run-dir PATH              指定训练 run；省略时由面板自动发现
  --monitor-dir PATH          指定资源监控目录；省略时自动发现
  --control-dir PATH          指定阶段控制目录；省略时自动发现
  --host HOST                 监听地址，默认 127.0.0.1
  --port PORT                 监听端口，默认 8765
  --poll-history-limit N      内存中保留的轮询历史长度，默认 600
  --conda-env NAME            Conda 环境，默认 python-torch
EOF
}

read_pid() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "${PID_FILE}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] && (( pid > 1 )) || return 1
  printf '%s\n' "${pid}"
}

process_start_ticks() {
  local pid="$1"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  awk '{print $22}' "/proc/${pid}/stat"
}

dashboard_identity_is_safe() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/cmdline" && -r "${START_FILE}" ]] || return 1

  local expected_ticks current_ticks command_line process_cwd
  expected_ticks="$(tr -d '[:space:]' < "${START_FILE}")"
  current_ticks="$(process_start_ticks "${pid}")"
  [[ -n "${expected_ticks}" && "${current_ticks}" == "${expected_ticks}" ]] || return 1

  command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
  [[ "${command_line}" == *"tools/training_dashboard.py"* ]] || return 1

  process_cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
  [[ "${process_cwd}" == "${PROJECT_ROOT}" ]]
}

clean_dead_pid_files() {
  local pid
  if ! pid="$(read_pid 2>/dev/null)"; then
    rm -f "${PID_FILE}" "${START_FILE}"
    return
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${PID_FILE}" "${START_FILE}"
  fi
}

status_dashboard() {
  local pid
  if ! pid="$(read_pid 2>/dev/null)"; then
    echo "dashboard=stopped"
    echo "log=${LOG_FILE}"
    return 1
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "dashboard=stopped (stale pid=${pid})"
    echo "log=${LOG_FILE}"
    return 1
  fi
  if ! dashboard_identity_is_safe "${pid}"; then
    echo "dashboard=unsafe-pid-record"
    echo "pid=${pid}"
    echo "refusing to treat this process as the dashboard"
    return 2
  fi
  echo "dashboard=running"
  echo "pid=${pid}"
  echo "log=${LOG_FILE}"
}

start_dashboard() {
  local host="127.0.0.1"
  local port="8765"
  local conda_env="${CONDA_ENV:-python-torch}"
  local poll_history_limit="600"
  local run_dir=""
  local monitor_dir=""
  local control_dir=""

  while (( $# > 0 )); do
    case "$1" in
      --run-dir|--monitor-dir|--control-dir|--host|--port|--poll-history-limit|--conda-env)
        (( $# >= 2 )) || { echo "缺少 $1 的参数值" >&2; return 2; }
        case "$1" in
          --run-dir) run_dir="$2" ;;
          --monitor-dir) monitor_dir="$2" ;;
          --control-dir) control_dir="$2" ;;
          --host) host="$2" ;;
          --port) port="$2" ;;
          --poll-history-limit) poll_history_limit="$2" ;;
          --conda-env) conda_env="$2" ;;
        esac
        shift 2
        ;;
      -h|--help)
        usage
        return
        ;;
      *)
        echo "未知参数：$1" >&2
        usage >&2
        return 2
        ;;
    esac
  done

  [[ "${port}" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || {
    echo "--port 必须在 1..65535 之间" >&2
    return 2
  }
  [[ "${poll_history_limit}" =~ ^[0-9]+$ ]] \
    && (( poll_history_limit >= 10 && poll_history_limit <= 5000 )) || {
    echo "--poll-history-limit 必须在 10..5000 之间" >&2
    return 2
  }
  [[ -f "${BACKEND}" ]] || { echo "找不到 ${BACKEND}" >&2; return 2; }
  command -v conda >/dev/null || { echo "找不到 conda" >&2; return 2; }
  command -v setsid >/dev/null || { echo "找不到 setsid" >&2; return 2; }

  mkdir -p "${DASHBOARD_DIR}"
  clean_dead_pid_files

  local existing_pid
  if existing_pid="$(read_pid 2>/dev/null)"; then
    if dashboard_identity_is_safe "${existing_pid}"; then
      echo "dashboard 已在运行，pid=${existing_pid}"
      return
    fi
    echo "PID 文件指向存活但无法验证为面板的进程；为保护训练进程，拒绝启动。" >&2
    echo "请人工检查 pid=${existing_pid}、${PID_FILE} 和 ${START_FILE}。" >&2
    return 3
  fi

  local -a command=(
    conda run --no-capture-output -n "${conda_env}"
    python -u "${BACKEND}"
    --host "${host}"
    --port "${port}"
    --poll-history-limit "${poll_history_limit}"
  )
  [[ -z "${run_dir}" ]] || command+=(--run-dir "${run_dir}")
  [[ -z "${monitor_dir}" ]] || command+=(--monitor-dir "${monitor_dir}")
  [[ -z "${control_dir}" ]] || command+=(--control-dir "${control_dir}")

  cd "${PROJECT_ROOT}"
  {
    echo
    echo "===== dashboard start $(date -Is) ====="
    echo "bind=${host}:${port}"
    echo "conda_env=${conda_env}"
  } >> "${LOG_FILE}"
  rm -f "${PID_FILE}" "${START_FILE}"
  # `setsid` 在调用者恰好是进程组长时可能自行 fork，因此不能把外层 `$!` 当作
  # 服务 PID。让新会话中的短 bash 先记录自己的 PID/启动时刻，再 exec conda；
  # exec 后 PID 和进程组保持不变，PID 文件始终对应可安全终止的面板进程组。
  nohup setsid bash -c '
    pid_file="$1"
    start_file="$2"
    project_root="$3"
    shift 3
    printf "%s\n" "$$" > "${pid_file}"
    awk "{print \$22}" "/proc/$$/stat" > "${start_file}"
    cd "${project_root}"
    exec "$@"
  ' dashboard-supervisor \
    "${PID_FILE}" "${START_FILE}" "${PROJECT_ROOT}" "${command[@]}" \
    >> "${LOG_FILE}" 2>&1 < /dev/null &

  local attempt pid="" stable_checks=0 probe_host="${host}" probe_url=""
  local health_payload=""
  case "${probe_host}" in
    0.0.0.0|::|"[::]") probe_host="127.0.0.1" ;;
  esac
  if [[ "${probe_host}" == *:* ]]; then
    probe_url="http://[${probe_host}]:${port}/api/health"
  else
    probe_url="http://${probe_host}:${port}/api/health"
  fi

  # 最长等待约 6 秒。存在 curl 时以真实健康接口为准；极简系统没有 curl 时，
  # 至少要求进程身份连续稳定 1.5 秒，避免端口占用或参数错误被误报为 started。
  for attempt in {1..60}; do
    pid="$(read_pid 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && dashboard_identity_is_safe "${pid}"; then
      if command -v curl >/dev/null; then
        health_payload="$(
          curl --silent --show-error --fail --max-time 1 \
            "${probe_url}" 2>/dev/null || true
        )"
        if [[ "${health_payload}" == *'"schema_version":1'* ]]; then
          sleep 0.2
          dashboard_identity_is_safe "${pid}" || continue
          echo "dashboard=started"
          echo "pid=${pid}"
          echo "url=http://${host}:${port}/"
          echo "log=${LOG_FILE}"
          return
        fi
      else
        (( stable_checks += 1 ))
        if (( stable_checks >= 15 )); then
          echo "dashboard=started (health probe unavailable; process stable)"
          echo "pid=${pid}"
          echo "url=http://${host}:${port}/"
          echo "log=${LOG_FILE}"
          return
        fi
      fi
    else
      stable_checks=0
    fi
    sleep 0.1
  done

  if [[ -n "${pid}" ]] && dashboard_identity_is_safe "${pid}"; then
    local pgid
    pgid="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
    if [[ "${pgid}" == "${pid}" ]]; then
      kill -TERM -- "-${pgid}" 2>/dev/null || true
    else
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${PID_FILE}" "${START_FILE}"
  echo "面板启动失败，最近日志：" >&2
  tail -n 30 "${LOG_FILE}" >&2 || true
  return 1
}

stop_dashboard() {
  local pid
  if ! pid="$(read_pid 2>/dev/null)"; then
    clean_dead_pid_files
    echo "dashboard 已停止"
    return
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${PID_FILE}" "${START_FILE}"
    echo "dashboard 已停止（已清理陈旧 PID）"
    return
  fi
  if ! dashboard_identity_is_safe "${pid}"; then
    echo "PID 无法验证为 training_dashboard.py；拒绝发送任何信号。" >&2
    echo "pid=${pid}" >&2
    return 3
  fi

  local pgid
  pgid="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
  if [[ "${pgid}" == "${pid}" ]]; then
    kill -TERM -- "-${pgid}"
  else
    kill -TERM "${pid}"
  fi

  local attempt
  for attempt in {1..20}; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "${pid}" 2>/dev/null && dashboard_identity_is_safe "${pid}"; then
    if [[ "${pgid}" == "${pid}" ]]; then
      kill -KILL -- "-${pgid}"
    else
      kill -KILL "${pid}"
    fi
  fi
  rm -f "${PID_FILE}" "${START_FILE}"
  echo "dashboard=stopped"
}

action="${1:-}"
[[ -n "${action}" ]] || { usage; exit 2; }
shift

case "${action}" in
  start) start_dashboard "$@" ;;
  stop) (( $# == 0 )) || { echo "stop 不接受额外参数" >&2; exit 2; }; stop_dashboard ;;
  status) (( $# == 0 )) || { echo "status 不接受额外参数" >&2; exit 2; }; status_dashboard ;;
  restart) stop_dashboard; start_dashboard "$@" ;;
  -h|--help|help) usage ;;
  *) echo "未知操作：${action}" >&2; usage >&2; exit 2 ;;
esac
