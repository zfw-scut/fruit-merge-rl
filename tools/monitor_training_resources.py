#!/usr/bin/env python3
"""训练资源旁路监控脚本。

本脚本是一个独立于 DQN 训练入口的外部监控器，用于定位长时间训练时的
系统资源问题。它不 import `daxigua` 或 `daxigua_rl`，也不修改训练参数；
只通过 Linux `/proc` 和 `nvidia-smi` 读取状态，然后把采样结果持续写入日志。

推荐使用方式是在一个终端启动本脚本，另一个终端照常启动训练。这样即使
训练进程被 OOM killer 杀掉，或者图形界面进入异常状态，监控日志仍然能保留
崩溃前最后几轮采样。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / 'runs' / 'resource_monitor'
PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')
CPU_COUNT = os.cpu_count() or 1


@dataclass(frozen=True)
class ProcessInfo:
    """从 `/proc/<pid>` 读取出来的一条进程快照。

    `cpu_jiffies` 是 Linux 内核中的 CPU tick 累计值。监控器会把相邻两次采样
    的差值换算成 CPU 百分比，因此第一轮采样的 CPU 百分比通常为 0。
    """

    pid: int
    ppid: int
    name: str
    cmdline: str
    rss_mb: float
    vms_mb: float
    cpu_jiffies: int
    cpu_percent: float
    threads: int


@dataclass(frozen=True)
class GPUInfo:
    """`nvidia-smi --query-gpu` 返回的一块 GPU 状态。"""

    index: int
    name: str
    temperature_c: float | None
    util_gpu_percent: float | None
    util_mem_percent: float | None
    memory_total_mb: float | None
    memory_used_mb: float | None
    memory_free_mb: float | None
    power_draw_w: float | None
    power_limit_w: float | None


@dataclass(frozen=True)
class GPUProcessInfo:
    """`nvidia-smi --query-compute-apps` 返回的一条 GPU 计算进程记录。"""

    pid: int
    process_name: str
    used_memory_mb: float | None


def now_iso() -> str:
    """返回带本地时区的秒级时间戳，方便和 `journalctl` 日志对齐。"""

    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def timestamp_for_path() -> str:
    """生成适合放进目录名的时间戳。"""

    return dt.datetime.now().strftime('%Y%m%d_%H%M%S')


def read_cpu_total_jiffies() -> int:
    """读取整机 CPU 累计 tick。

    `/proc/stat` 第一行的 `cpu` 汇总了所有 CPU 核心的各种时间。这里直接求和，
    后续用相邻两次的差值作为进程 CPU 百分比的分母。
    """

    with Path('/proc/stat').open('r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('cpu '):
                return sum(int(value) for value in line.split()[1:])
    raise RuntimeError('cannot read aggregate CPU line from /proc/stat')


def read_meminfo() -> dict[str, float]:
    """读取系统内存和 swap 状态，单位统一转换成 MiB。"""

    values_kib: dict[str, int] = {}
    with Path('/proc/meminfo').open('r', encoding='utf-8') as file:
        for line in file:
            key, rest = line.split(':', 1)
            parts = rest.strip().split()
            if parts:
                values_kib[key] = int(parts[0])

    mem_total = values_kib.get('MemTotal', 0) / 1024
    mem_available = values_kib.get('MemAvailable', 0) / 1024
    swap_total = values_kib.get('SwapTotal', 0) / 1024
    swap_free = values_kib.get('SwapFree', 0) / 1024

    return {
        'mem_total_mb': mem_total,
        'mem_available_mb': mem_available,
        'mem_used_mb': max(0.0, mem_total - mem_available),
        'mem_available_percent': 0.0 if mem_total <= 0 else mem_available / mem_total * 100,
        'swap_total_mb': swap_total,
        'swap_free_mb': swap_free,
        'swap_used_mb': max(0.0, swap_total - swap_free),
    }


def read_cmdline(pid: int, fallback_name: str) -> str:
    """读取进程命令行。

    某些内核线程或已经退出的进程可能没有可读 cmdline；这时退回到进程名，
    这样日志里仍然能知道是哪类进程。
    """

    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return fallback_name

    text = raw.replace(b'\x00', b' ').decode('utf-8', errors='replace').strip()
    return text or fallback_name


def read_process_info(pid: int, previous_cpu: dict[int, int], previous_total: int | None, current_total: int) -> ProcessInfo | None:
    """读取单个进程状态，并根据上一次采样估算 CPU 百分比。"""

    try:
        stat_text = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
    except OSError:
        return None

    # `/proc/<pid>/stat` 第二个字段是括号包裹的 comm，里面理论上可以包含空格。
    # 因此不能直接 split 整行，而是先定位最后一个右括号，再解析后续字段。
    close_index = stat_text.rfind(')')
    if close_index < 0:
        return None

    name = stat_text[stat_text.find('(') + 1 : close_index]
    rest = stat_text[close_index + 2 :].split()
    if len(rest) < 22:
        return None

    try:
        ppid = int(rest[1])
        utime = int(rest[11])
        stime = int(rest[12])
        threads = int(rest[17])
        vms_bytes = int(rest[20])
        rss_pages = int(rest[21])
    except ValueError:
        return None

    cpu_jiffies = utime + stime
    cpu_percent = 0.0
    if previous_total is not None and current_total > previous_total:
        previous_process_cpu = previous_cpu.get(pid)
        if previous_process_cpu is not None and cpu_jiffies >= previous_process_cpu:
            # `/proc/stat` 的总 tick 是所有 CPU 核心的总和。乘以 CPU_COUNT 后，
            # 单个多线程进程可以显示超过 100%，这更符合 top/htop 的直觉。
            cpu_percent = (cpu_jiffies - previous_process_cpu) / (current_total - previous_total) * CPU_COUNT * 100

    return ProcessInfo(
        pid=pid,
        ppid=ppid,
        name=name,
        cmdline=read_cmdline(pid, name),
        rss_mb=rss_pages * PAGE_SIZE / 1024 / 1024,
        vms_mb=vms_bytes / 1024 / 1024,
        cpu_jiffies=cpu_jiffies,
        cpu_percent=cpu_percent,
        threads=threads,
    )


def read_all_processes(previous_cpu: dict[int, int], previous_total: int | None, current_total: int) -> dict[int, ProcessInfo]:
    """扫描 `/proc` 下所有数字目录，得到当前进程快照。"""

    processes: dict[int, ProcessInfo] = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue

        info = read_process_info(int(entry.name), previous_cpu, previous_total, current_total)
        if info is not None:
            processes[info.pid] = info

    return processes


def descendants_of(root_pids: set[int], processes: dict[int, ProcessInfo]) -> set[int]:
    """根据 PPID 关系找出指定 PID 的所有子孙进程。"""

    if not root_pids:
        return set()

    children_by_parent: dict[int, list[int]] = {}
    for process in processes.values():
        children_by_parent.setdefault(process.ppid, []).append(process.pid)

    found: set[int] = set()
    stack = list(root_pids)
    while stack:
        parent = stack.pop()
        for child in children_by_parent.get(parent, []):
            if child not in found and child not in root_pids:
                found.add(child)
                stack.append(child)

    return found


def ancestor_pids(pid: int, processes: dict[int, ProcessInfo]) -> set[int]:
    """找出当前监控器自己的父进程链，避免把启动它的 shell 误判成目标。"""

    ancestors: set[int] = set()
    current = processes.get(pid)
    while current is not None and current.ppid > 0 and current.ppid not in ancestors:
        ancestors.add(current.ppid)
        current = processes.get(current.ppid)
    return ancestors


def select_target_processes(args: argparse.Namespace, processes: dict[int, ProcessInfo]) -> list[ProcessInfo]:
    """根据 `--pid` 和 `--match` 选择需要重点记录的训练进程。"""

    own_pid = os.getpid()
    ignored_pids = {own_pid} | ancestor_pids(own_pid, processes)
    selected_pids = {pid for pid in args.pid if pid in processes}

    for process in processes.values():
        if process.pid in ignored_pids:
            continue
        if any(pattern in process.cmdline for pattern in args.match):
            selected_pids.add(process.pid)

    if args.include_children:
        selected_pids |= descendants_of(selected_pids, processes)

    selected_pids -= ignored_pids
    return [processes[pid] for pid in sorted(selected_pids) if pid in processes]


def parse_float(value: str) -> float | None:
    """把 `nvidia-smi` 的字段转成 float，无法解析时返回 None。"""

    value = value.strip()
    if value in {'', 'N/A', '[N/A]'}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def run_nvidia_smi(args: list[str], timeout: float) -> tuple[int, str, str]:
    """执行 `nvidia-smi` 并返回 exit code、stdout、stderr。"""

    try:
        result = subprocess.run(
            ['nvidia-smi', *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return 127, '', 'nvidia-smi not found'
    except subprocess.TimeoutExpired:
        return 124, '', f'nvidia-smi timed out after {timeout:.1f}s'

    return result.returncode, result.stdout, result.stderr


def query_gpus(timeout: float) -> tuple[list[GPUInfo], str | None]:
    """查询 GPU 总览信息。

    返回值中的第二项是错误字符串。错误只写入日志，不让监控器退出，因为在图形
    栈异常时 `nvidia-smi` 自身失败也正是我们想记录的现场。
    """

    query = ','.join(
        (
            'index',
            'name',
            'temperature.gpu',
            'utilization.gpu',
            'utilization.memory',
            'memory.total',
            'memory.used',
            'memory.free',
            'power.draw',
            'power.limit',
        )
    )
    code, stdout, stderr = run_nvidia_smi(
        [f'--query-gpu={query}', '--format=csv,noheader,nounits'],
        timeout=timeout,
    )
    if code != 0:
        return [], (stderr or stdout or f'nvidia-smi exited with code {code}').strip()

    gpus: list[GPUInfo] = []
    for row in csv.reader(io.StringIO(stdout)):
        if len(row) < 10:
            continue
        try:
            index = int(row[0].strip())
        except ValueError:
            continue

        gpus.append(
            GPUInfo(
                index=index,
                name=row[1].strip(),
                temperature_c=parse_float(row[2]),
                util_gpu_percent=parse_float(row[3]),
                util_mem_percent=parse_float(row[4]),
                memory_total_mb=parse_float(row[5]),
                memory_used_mb=parse_float(row[6]),
                memory_free_mb=parse_float(row[7]),
                power_draw_w=parse_float(row[8]),
                power_limit_w=parse_float(row[9]),
            )
        )

    return gpus, None


def query_gpu_processes(timeout: float) -> tuple[list[GPUProcessInfo], str | None]:
    """查询当前 CUDA/compute 进程。

    图形进程不一定出现在 compute-apps 中；训练脚本如果正在用 CUDA，通常会出现在
    这里。没有计算进程时，不视为错误。
    """

    code, stdout, stderr = run_nvidia_smi(
        [
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits',
        ],
        timeout=timeout,
    )
    if code != 0:
        message = (stderr or stdout or f'nvidia-smi exited with code {code}').strip()
        if 'No running processes found' in message:
            return [], None
        return [], message

    processes: list[GPUProcessInfo] = []
    for row in csv.reader(io.StringIO(stdout)):
        if len(row) < 3:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue

        processes.append(
            GPUProcessInfo(
                pid=pid,
                process_name=row[1].strip(),
                used_memory_mb=parse_float(row[2]),
            )
        )

    return processes, None


def finite_values(values: Iterable[float | None]) -> list[float]:
    """过滤 None，便于计算 GPU 聚合指标。"""

    return [value for value in values if value is not None]


def max_or_none(values: Iterable[float | None]) -> float | None:
    """对可能为空的数值列表求最大值。"""

    filtered = finite_values(values)
    return max(filtered) if filtered else None


def sum_or_none(values: Iterable[float | None]) -> float | None:
    """对可能为空的数值列表求和。"""

    filtered = finite_values(values)
    return sum(filtered) if filtered else None


def fmt_mb(value: float | None) -> str:
    """把 MiB 数值格式化成适合终端心跳展示的文本。"""

    if value is None:
        return 'n/a'
    if value >= 1024:
        return f'{value / 1024:.1f}GiB'
    return f'{value:.0f}MiB'


def write_jsonl(file, payload: dict) -> None:
    """写一行 JSONL 并立即 flush，减少异常关机时的数据丢失。"""

    file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')
    file.flush()


def open_csv(path: Path, fieldnames: list[str]):
    """打开 CSV 文件并写入表头。"""

    file = path.open('w', encoding='utf-8', newline='')
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    file.flush()
    return file, writer


def build_log_dir(args: argparse.Namespace) -> Path:
    """确定本次监控输出目录。"""

    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
    else:
        log_dir = DEFAULT_LOG_ROOT / timestamp_for_path()

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description='独立监控 DQN 训练期间的系统内存、目标进程和 NVIDIA GPU 状态。'
    )
    parser.add_argument(
        '--log-dir',
        default=None,
        help='日志输出目录。默认写入 runs/resource_monitor/<时间戳>/。',
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=3.0,
        help='采样间隔秒数。建议和训练 progress-interval 一样使用 3 秒。',
    )
    parser.add_argument(
        '--match',
        action='append',
        default=['daxigua_rl.scripts.train_dqn'],
        help='按命令行子串匹配目标进程。可重复传入；默认匹配训练入口模块名。',
    )
    parser.add_argument(
        '--pid',
        action='append',
        type=int,
        default=[],
        help='额外指定目标 PID。适合训练已经启动后再挂监控。',
    )
    parser.add_argument(
        '--no-children',
        dest='include_children',
        action='store_false',
        help='只记录匹配到的 PID，不自动纳入其子进程。',
    )
    parser.set_defaults(include_children=True)
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='跳过 nvidia-smi 查询，只记录 CPU/内存/进程。',
    )
    parser.add_argument(
        '--nvidia-timeout',
        type=float,
        default=2.0,
        help='每次 nvidia-smi 查询最多等待多少秒。',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最多采样多少次，主要用于 smoke test。',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='最多运行多少秒。',
    )
    parser.add_argument(
        '--stop-after-target-exits',
        action='store_true',
        help='目标训练进程出现过并退出后，监控器自动停止。',
    )
    parser.add_argument(
        '--warn-mem-available-mb',
        type=float,
        default=2048.0,
        help='系统可用内存低于该值时记录 warning 事件。',
    )
    parser.add_argument(
        '--warn-swap-used-mb',
        type=float,
        default=1024.0,
        help='swap 使用量高于该值时记录 warning 事件。',
    )
    parser.add_argument(
        '--warn-target-rss-mb',
        type=float,
        default=8192.0,
        help='目标进程 RSS 总量高于该值时记录 warning 事件。',
    )
    parser.add_argument(
        '--warn-gpu-memory-used-mb',
        type=float,
        default=7000.0,
        help='GPU 显存使用量高于该值时记录 warning 事件。',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='不打印每轮心跳，只写日志。',
    )
    return parser.parse_args()


def emit_event(events_file, level: str, event: str, elapsed_sec: float, details: dict) -> None:
    """写入结构化事件日志。"""

    write_jsonl(
        events_file,
        {
            'timestamp': now_iso(),
            'elapsed_sec': round(elapsed_sec, 3),
            'level': level,
            'event': event,
            'details': details,
        },
    )


def maybe_emit_threshold_event(
    events_file,
    active_events: dict[str, bool],
    key: str,
    active: bool,
    elapsed_sec: float,
    details: dict,
) -> None:
    """只在阈值状态发生变化时写事件，避免 warning 刷屏。"""

    was_active = active_events.get(key, False)
    if active and not was_active:
        emit_event(events_file, 'warning', key, elapsed_sec, details)
    elif was_active and not active:
        emit_event(events_file, 'info', f'{key}_recovered', elapsed_sec, details)
    active_events[key] = active


def print_heartbeat(
    sample: int,
    elapsed_sec: float,
    meminfo: dict[str, float],
    target_processes: list[ProcessInfo],
    gpus: list[GPUInfo],
    gpu_error: str | None,
) -> None:
    """打印一行人能快速扫懂的实时心跳。"""

    target_rss = sum(process.rss_mb for process in target_processes)
    target_cpu = sum(process.cpu_percent for process in target_processes)
    gpu_used = sum_or_none(gpu.memory_used_mb for gpu in gpus)
    gpu_total = sum_or_none(gpu.memory_total_mb for gpu in gpus)
    gpu_util = max_or_none(gpu.util_gpu_percent for gpu in gpus)

    gpu_text = 'gpu=n/a'
    if gpu_error:
        gpu_text = f'gpu_error={gpu_error[:80]}'
    elif gpu_total is not None:
        gpu_text = f'gpu_mem={fmt_mb(gpu_used)}/{fmt_mb(gpu_total)} gpu_util={gpu_util or 0:.0f}%'

    print(
        f'[monitor] sample={sample} elapsed={elapsed_sec:.0f}s '
        f'mem_avail={fmt_mb(meminfo["mem_available_mb"])} '
        f'swap_used={fmt_mb(meminfo["swap_used_mb"])} '
        f'target_proc={len(target_processes)} '
        f'target_rss={fmt_mb(target_rss)} '
        f'target_cpu={target_cpu:.1f}% '
        f'{gpu_text}',
        flush=True,
    )


def run_monitor(args: argparse.Namespace) -> int:
    """运行监控主循环。"""

    if args.interval <= 0:
        raise ValueError('--interval must be positive')

    log_dir = build_log_dir(args)
    started_at = time.monotonic()
    should_stop = False

    def request_stop(signum, _frame) -> None:
        nonlocal should_stop
        should_stop = True
        print(f'[monitor] received signal {signum}, stopping after current sample...', flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    system_fields = [
        'timestamp',
        'elapsed_sec',
        'sample',
        'hostname',
        'load1',
        'load5',
        'load15',
        'cpu_count',
        'mem_total_mb',
        'mem_used_mb',
        'mem_available_mb',
        'mem_available_percent',
        'swap_total_mb',
        'swap_used_mb',
        'swap_free_mb',
        'target_process_count',
        'target_rss_mb',
        'target_vms_mb',
        'target_cpu_percent',
        'target_threads',
        'gpu_count',
        'gpu_memory_used_mb',
        'gpu_memory_total_mb',
        'gpu_util_percent_max',
        'gpu_mem_util_percent_max',
        'gpu_temperature_c_max',
        'gpu_power_draw_w_sum',
        'gpu_power_limit_w_sum',
        'gpu_query_error',
    ]
    process_fields = [
        'timestamp',
        'elapsed_sec',
        'sample',
        'pid',
        'ppid',
        'name',
        'rss_mb',
        'vms_mb',
        'cpu_percent',
        'threads',
        'cmdline',
    ]
    gpu_fields = [
        'timestamp',
        'elapsed_sec',
        'sample',
        'index',
        'name',
        'temperature_c',
        'util_gpu_percent',
        'util_mem_percent',
        'memory_total_mb',
        'memory_used_mb',
        'memory_free_mb',
        'power_draw_w',
        'power_limit_w',
    ]
    gpu_process_fields = [
        'timestamp',
        'elapsed_sec',
        'sample',
        'pid',
        'is_target_process',
        'process_name',
        'used_memory_mb',
    ]

    system_file, system_writer = open_csv(log_dir / 'system_metrics.csv', system_fields)
    process_file, process_writer = open_csv(log_dir / 'process_metrics.csv', process_fields)
    gpu_file, gpu_writer = open_csv(log_dir / 'gpu_metrics.csv', gpu_fields)
    gpu_process_file, gpu_process_writer = open_csv(log_dir / 'gpu_process_metrics.csv', gpu_process_fields)
    events_file = (log_dir / 'events.jsonl').open('w', encoding='utf-8')

    metadata = {
        'timestamp': now_iso(),
        'hostname': socket.gethostname(),
        'monitor_pid': os.getpid(),
        'cwd': str(Path.cwd()),
        'argv': sys.argv,
        'args': vars(args),
    }
    (log_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'[monitor] logging to {log_dir}', flush=True)
    emit_event(events_file, 'info', 'monitor_started', 0.0, metadata)

    previous_total_cpu: int | None = None
    previous_process_cpu: dict[int, int] = {}
    previous_target_pids: set[int] = set()
    active_events: dict[str, bool] = {}
    target_seen = False
    sample = 0

    peak_target_rss = 0.0
    peak_swap_used = 0.0
    min_mem_available = float('inf')
    peak_gpu_memory_used = 0.0

    try:
        while not should_stop:
            loop_started_at = time.monotonic()
            elapsed_sec = loop_started_at - started_at

            if args.duration is not None and elapsed_sec > args.duration:
                emit_event(events_file, 'info', 'duration_reached', elapsed_sec, {'duration': args.duration})
                break

            current_total_cpu = read_cpu_total_jiffies()
            processes = read_all_processes(previous_process_cpu, previous_total_cpu, current_total_cpu)
            target_processes = select_target_processes(args, processes)
            target_pids = {process.pid for process in target_processes}

            if target_pids and not target_seen:
                target_seen = True
                emit_event(events_file, 'info', 'target_process_started', elapsed_sec, {'pids': sorted(target_pids)})
            elif target_seen and previous_target_pids and not target_pids:
                emit_event(events_file, 'warning', 'target_process_missing', elapsed_sec, {})

            if target_pids != previous_target_pids:
                emit_event(
                    events_file,
                    'info',
                    'target_process_set_changed',
                    elapsed_sec,
                    {
                        'previous_pids': sorted(previous_target_pids),
                        'current_pids': sorted(target_pids),
                    },
                )
                previous_target_pids = target_pids

            meminfo = read_meminfo()
            load1, load5, load15 = os.getloadavg()

            gpu_error = None
            gpu_process_error = None
            gpus: list[GPUInfo] = []
            gpu_processes: list[GPUProcessInfo] = []
            if not args.no_gpu:
                gpus, gpu_error = query_gpus(args.nvidia_timeout)
                gpu_processes, gpu_process_error = query_gpu_processes(args.nvidia_timeout)

            if gpu_error:
                maybe_emit_threshold_event(
                    events_file,
                    active_events,
                    'gpu_query_failed',
                    True,
                    elapsed_sec,
                    {'error': gpu_error},
                )
            else:
                maybe_emit_threshold_event(events_file, active_events, 'gpu_query_failed', False, elapsed_sec, {})

            if gpu_process_error:
                emit_event(events_file, 'warning', 'gpu_process_query_failed', elapsed_sec, {'error': gpu_process_error})

            target_rss = sum(process.rss_mb for process in target_processes)
            target_vms = sum(process.vms_mb for process in target_processes)
            target_cpu = sum(process.cpu_percent for process in target_processes)
            target_threads = sum(process.threads for process in target_processes)
            gpu_memory_used = sum_or_none(gpu.memory_used_mb for gpu in gpus)
            gpu_memory_total = sum_or_none(gpu.memory_total_mb for gpu in gpus)

            peak_target_rss = max(peak_target_rss, target_rss)
            peak_swap_used = max(peak_swap_used, meminfo['swap_used_mb'])
            min_mem_available = min(min_mem_available, meminfo['mem_available_mb'])
            if gpu_memory_used is not None:
                peak_gpu_memory_used = max(peak_gpu_memory_used, gpu_memory_used)

            maybe_emit_threshold_event(
                events_file,
                active_events,
                'low_system_memory',
                meminfo['mem_available_mb'] < args.warn_mem_available_mb,
                elapsed_sec,
                {
                    'mem_available_mb': round(meminfo['mem_available_mb'], 1),
                    'threshold_mb': args.warn_mem_available_mb,
                },
            )
            maybe_emit_threshold_event(
                events_file,
                active_events,
                'swap_usage_high',
                meminfo['swap_used_mb'] > args.warn_swap_used_mb,
                elapsed_sec,
                {
                    'swap_used_mb': round(meminfo['swap_used_mb'], 1),
                    'threshold_mb': args.warn_swap_used_mb,
                },
            )
            maybe_emit_threshold_event(
                events_file,
                active_events,
                'target_rss_high',
                target_rss > args.warn_target_rss_mb,
                elapsed_sec,
                {
                    'target_rss_mb': round(target_rss, 1),
                    'threshold_mb': args.warn_target_rss_mb,
                    'pids': sorted(target_pids),
                },
            )
            maybe_emit_threshold_event(
                events_file,
                active_events,
                'gpu_memory_high',
                gpu_memory_used is not None and gpu_memory_used > args.warn_gpu_memory_used_mb,
                elapsed_sec,
                {
                    'gpu_memory_used_mb': None if gpu_memory_used is None else round(gpu_memory_used, 1),
                    'threshold_mb': args.warn_gpu_memory_used_mb,
                },
            )

            timestamp = now_iso()
            system_writer.writerow(
                {
                    'timestamp': timestamp,
                    'elapsed_sec': round(elapsed_sec, 3),
                    'sample': sample,
                    'hostname': socket.gethostname(),
                    'load1': round(load1, 3),
                    'load5': round(load5, 3),
                    'load15': round(load15, 3),
                    'cpu_count': CPU_COUNT,
                    'mem_total_mb': round(meminfo['mem_total_mb'], 1),
                    'mem_used_mb': round(meminfo['mem_used_mb'], 1),
                    'mem_available_mb': round(meminfo['mem_available_mb'], 1),
                    'mem_available_percent': round(meminfo['mem_available_percent'], 2),
                    'swap_total_mb': round(meminfo['swap_total_mb'], 1),
                    'swap_used_mb': round(meminfo['swap_used_mb'], 1),
                    'swap_free_mb': round(meminfo['swap_free_mb'], 1),
                    'target_process_count': len(target_processes),
                    'target_rss_mb': round(target_rss, 1),
                    'target_vms_mb': round(target_vms, 1),
                    'target_cpu_percent': round(target_cpu, 2),
                    'target_threads': target_threads,
                    'gpu_count': len(gpus),
                    'gpu_memory_used_mb': '' if gpu_memory_used is None else round(gpu_memory_used, 1),
                    'gpu_memory_total_mb': '' if gpu_memory_total is None else round(gpu_memory_total, 1),
                    'gpu_util_percent_max': '' if max_or_none(gpu.util_gpu_percent for gpu in gpus) is None else round(max_or_none(gpu.util_gpu_percent for gpu in gpus), 1),
                    'gpu_mem_util_percent_max': '' if max_or_none(gpu.util_mem_percent for gpu in gpus) is None else round(max_or_none(gpu.util_mem_percent for gpu in gpus), 1),
                    'gpu_temperature_c_max': '' if max_or_none(gpu.temperature_c for gpu in gpus) is None else round(max_or_none(gpu.temperature_c for gpu in gpus), 1),
                    'gpu_power_draw_w_sum': '' if sum_or_none(gpu.power_draw_w for gpu in gpus) is None else round(sum_or_none(gpu.power_draw_w for gpu in gpus), 2),
                    'gpu_power_limit_w_sum': '' if sum_or_none(gpu.power_limit_w for gpu in gpus) is None else round(sum_or_none(gpu.power_limit_w for gpu in gpus), 2),
                    'gpu_query_error': gpu_error or '',
                }
            )

            for process in target_processes:
                process_writer.writerow(
                    {
                        'timestamp': timestamp,
                        'elapsed_sec': round(elapsed_sec, 3),
                        'sample': sample,
                        'pid': process.pid,
                        'ppid': process.ppid,
                        'name': process.name,
                        'rss_mb': round(process.rss_mb, 1),
                        'vms_mb': round(process.vms_mb, 1),
                        'cpu_percent': round(process.cpu_percent, 2),
                        'threads': process.threads,
                        'cmdline': process.cmdline,
                    }
                )

            for gpu in gpus:
                gpu_writer.writerow(
                    {
                        'timestamp': timestamp,
                        'elapsed_sec': round(elapsed_sec, 3),
                        'sample': sample,
                        'index': gpu.index,
                        'name': gpu.name,
                        'temperature_c': '' if gpu.temperature_c is None else round(gpu.temperature_c, 1),
                        'util_gpu_percent': '' if gpu.util_gpu_percent is None else round(gpu.util_gpu_percent, 1),
                        'util_mem_percent': '' if gpu.util_mem_percent is None else round(gpu.util_mem_percent, 1),
                        'memory_total_mb': '' if gpu.memory_total_mb is None else round(gpu.memory_total_mb, 1),
                        'memory_used_mb': '' if gpu.memory_used_mb is None else round(gpu.memory_used_mb, 1),
                        'memory_free_mb': '' if gpu.memory_free_mb is None else round(gpu.memory_free_mb, 1),
                        'power_draw_w': '' if gpu.power_draw_w is None else round(gpu.power_draw_w, 2),
                        'power_limit_w': '' if gpu.power_limit_w is None else round(gpu.power_limit_w, 2),
                    }
                )

            for gpu_process in gpu_processes:
                gpu_process_writer.writerow(
                    {
                        'timestamp': timestamp,
                        'elapsed_sec': round(elapsed_sec, 3),
                        'sample': sample,
                        'pid': gpu_process.pid,
                        'is_target_process': gpu_process.pid in target_pids,
                        'process_name': gpu_process.process_name,
                        'used_memory_mb': '' if gpu_process.used_memory_mb is None else round(gpu_process.used_memory_mb, 1),
                    }
                )

            # 每轮采样后都 flush。这样即使后面突然黑屏或断电，已写入的数据更可能保留。
            for file in (system_file, process_file, gpu_file, gpu_process_file):
                file.flush()

            if not args.quiet:
                print_heartbeat(sample, elapsed_sec, meminfo, target_processes, gpus, gpu_error)

            previous_total_cpu = current_total_cpu
            previous_process_cpu = {pid: process.cpu_jiffies for pid, process in processes.items()}

            sample += 1
            if args.max_samples is not None and sample >= args.max_samples:
                emit_event(events_file, 'info', 'max_samples_reached', elapsed_sec, {'max_samples': args.max_samples})
                break

            if args.stop_after_target_exits and target_seen and not target_processes:
                emit_event(events_file, 'info', 'target_exited_stop_requested', elapsed_sec, {})
                break

            sleep_seconds = args.interval - (time.monotonic() - loop_started_at)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        total_elapsed = time.monotonic() - started_at
        summary = {
            'timestamp': now_iso(),
            'elapsed_sec': round(total_elapsed, 3),
            'samples': sample,
            'peak_target_rss_mb': round(peak_target_rss, 1),
            'peak_swap_used_mb': round(peak_swap_used, 1),
            'min_mem_available_mb': None if min_mem_available == float('inf') else round(min_mem_available, 1),
            'peak_gpu_memory_used_mb': round(peak_gpu_memory_used, 1),
            'log_dir': str(log_dir),
        }
        (log_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        emit_event(events_file, 'info', 'monitor_stopped', total_elapsed, summary)

        for file in (system_file, process_file, gpu_file, gpu_process_file, events_file):
            file.close()

        print(
            '[monitor] stopped '
            f'samples={sample} '
            f'peak_target_rss={fmt_mb(peak_target_rss)} '
            f'peak_swap={fmt_mb(peak_swap_used)} '
            f'peak_gpu_mem={fmt_mb(peak_gpu_memory_used)} '
            f'log_dir={log_dir}',
            flush=True,
        )

    return 0


def main() -> int:
    """命令行入口。"""

    try:
        return run_monitor(parse_args())
    except Exception as exc:
        print(f'[monitor] fatal: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
