#!/usr/bin/env python3
"""独立 CUDA 计算压力测试与日志采集脚本。

本脚本用于把问题从项目训练链路中剥离出来：不 import 游戏、不 import RL 模块，
只使用 PyTorch 在 CUDA 上做连续矩阵乘法，并同步记录 NVIDIA GPU、系统内存、
进程内存和内核日志中的 NVIDIA/Xid 事件。

如果该脚本也能稳定复现黑屏或 Xid 16，则问题更可能落在 NVIDIA 驱动、显示栈、
硬件状态、功耗/散热或高刷新率显示链路，而不是 DQN 训练代码本身。
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
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / 'runs' / 'cuda_stress'
PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')
NVIDIA_LOG_KEYWORDS = (
    'nvrm',
    'xid',
    'nvidia',
    'nvidia-modeset',
    'lost display',
    'gpu progress',
    'drm',
)


def now() -> dt.datetime:
    """返回当前本地时区时间，方便和 `journalctl` 的 `short-iso` 输出对齐。"""

    return dt.datetime.now().astimezone()


def now_iso() -> str:
    """返回秒级 ISO 时间戳。"""

    return now().isoformat(timespec='seconds')


def journal_time(value: dt.datetime) -> str:
    """把时间格式化成当前 `journalctl` 能稳定解析的本地时间字符串。"""

    return value.astimezone().strftime('%Y-%m-%d %H:%M:%S')


def timestamp_for_path() -> str:
    """生成适合目录名的时间戳。"""

    return dt.datetime.now().strftime('%Y%m%d_%H%M%S')


def run_command(command: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    """运行一个只读诊断命令，并把失败也记录为文本。"""

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return 127, '', f'{command[0]} not found'
    except subprocess.TimeoutExpired:
        return 124, '', f'command timed out after {timeout:.1f}s: {command}'

    return result.returncode, result.stdout, result.stderr


def parse_float(value: str) -> float | None:
    """把 `nvidia-smi` 的数值字段转成 float。"""

    value = value.strip()
    if value in {'', 'N/A', '[N/A]'}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def query_gpu(timeout: float) -> tuple[dict[str, object], str | None]:
    """查询单机 NVIDIA GPU 汇总信息。

    这里按当前机器一块 4070 Laptop GPU 的情况返回第一块 GPU；如果后续有多卡，
    仍然记录所有 GPU 的总显存使用量和最高温度/利用率。
    """

    fields = (
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
        'clocks.sm',
        'clocks.mem',
        'pstate',
    )
    code, stdout, stderr = run_command(
        [
            'nvidia-smi',
            f'--query-gpu={",".join(fields)}',
            '--format=csv,noheader,nounits',
        ],
        timeout=timeout,
    )
    if code != 0:
        return {}, (stderr or stdout or f'nvidia-smi exited with code {code}').strip()

    rows = list(csv.reader(io.StringIO(stdout)))
    if not rows:
        return {}, 'nvidia-smi returned no GPU rows'

    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        if len(row) < len(fields):
            continue
        parsed_rows.append(
            {
                'index': row[0].strip(),
                'name': row[1].strip(),
                'temperature_c': parse_float(row[2]),
                'gpu_util_percent': parse_float(row[3]),
                'memory_util_percent': parse_float(row[4]),
                'memory_total_mb': parse_float(row[5]),
                'memory_used_mb': parse_float(row[6]),
                'memory_free_mb': parse_float(row[7]),
                'power_draw_w': parse_float(row[8]),
                'power_limit_w': parse_float(row[9]),
                'clock_sm_mhz': parse_float(row[10]),
                'clock_mem_mhz': parse_float(row[11]),
                'pstate': row[12].strip(),
            }
        )

    if not parsed_rows:
        return {}, 'nvidia-smi rows could not be parsed'

    first = parsed_rows[0]
    first['gpu_count'] = len(parsed_rows)
    first['memory_used_total_mb'] = sum_number(row.get('memory_used_mb') for row in parsed_rows)
    first['memory_total_all_mb'] = sum_number(row.get('memory_total_mb') for row in parsed_rows)
    first['temperature_max_c'] = max_number(row.get('temperature_c') for row in parsed_rows)
    first['gpu_util_max_percent'] = max_number(row.get('gpu_util_percent') for row in parsed_rows)
    first['power_draw_total_w'] = sum_number(row.get('power_draw_w') for row in parsed_rows)
    return first, None


def sum_number(values: Iterable[object]) -> float | None:
    """对可能含 None 的数值求和。"""

    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numbers) if numbers else None


def max_number(values: Iterable[object]) -> float | None:
    """对可能含 None 的数值求最大值。"""

    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return max(numbers) if numbers else None


def read_meminfo() -> dict[str, float]:
    """读取系统内存和 swap 状态，单位统一为 MiB。"""

    values: dict[str, int] = {}
    with Path('/proc/meminfo').open('r', encoding='utf-8') as file:
        for line in file:
            key, rest = line.split(':', 1)
            parts = rest.strip().split()
            if parts:
                values[key] = int(parts[0])

    mem_total = values.get('MemTotal', 0) / 1024
    mem_available = values.get('MemAvailable', 0) / 1024
    swap_total = values.get('SwapTotal', 0) / 1024
    swap_free = values.get('SwapFree', 0) / 1024
    return {
        'mem_total_mb': mem_total,
        'mem_available_mb': mem_available,
        'mem_used_mb': max(0.0, mem_total - mem_available),
        'swap_total_mb': swap_total,
        'swap_used_mb': max(0.0, swap_total - swap_free),
        'swap_free_mb': swap_free,
    }


def read_self_memory_mb() -> dict[str, float]:
    """读取当前压力测试进程自己的 RSS/VMS。"""

    try:
        statm = Path('/proc/self/statm').read_text(encoding='utf-8').split()
        total_pages = int(statm[0])
        rss_pages = int(statm[1])
    except (OSError, ValueError, IndexError):
        return {'process_vms_mb': 0.0, 'process_rss_mb': 0.0}

    return {
        'process_vms_mb': total_pages * PAGE_SIZE / 1024 / 1024,
        'process_rss_mb': rss_pages * PAGE_SIZE / 1024 / 1024,
    }


def open_csv(path: Path, fieldnames: list[str]):
    """打开 CSV 文件并写入表头。"""

    file = path.open('w', encoding='utf-8', newline='')
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    file.flush()
    return file, writer


def write_jsonl(file, payload: dict) -> None:
    """写入一行 JSONL 并立即 flush，尽量保留崩溃前最后一条事件。"""

    file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')
    file.flush()


def append_command_report(file, title: str, command: list[str], timeout: float = 5.0) -> None:
    """把一条诊断命令的输出追加到报告文件。"""

    code, stdout, stderr = run_command(command, timeout=timeout)
    file.write(f'\n===== {title} =====\n')
    file.write(f'$ {" ".join(command)}\n')
    file.write(f'exit_code={code}\n')
    if stdout:
        file.write(stdout)
        if not stdout.endswith('\n'):
            file.write('\n')
    if stderr:
        file.write('[stderr]\n')
        file.write(stderr)
        if not stderr.endswith('\n'):
            file.write('\n')
    file.flush()


def append_preflight_report(path: Path) -> None:
    """采集测试开始时的系统和显示状态。"""

    with path.open('w', encoding='utf-8') as file:
        file.write(f'timestamp={now_iso()}\n')
        append_command_report(file, 'uname', ['uname', '-a'])
        append_command_report(file, 'nvidia-smi', ['nvidia-smi'])
        append_command_report(file, 'nvidia-smi -q summary', ['nvidia-smi', '-q'])
        append_command_report(file, 'prime-select', ['prime-select', 'query'])
        append_command_report(file, 'xrandr', ['xrandr', '--current'])
        append_command_report(file, 'nvidia driver version', ['cat', '/proc/driver/nvidia/version'])


def poll_kernel_logs(since: dt.datetime, kernel_file, nvidia_file) -> tuple[dt.datetime, list[str]]:
    """轮询内核日志，并把 NVIDIA/Xid 相关行额外写入单独文件。

    压力测试如果导致黑屏，脚本可能没有机会在结尾导出完整 journal。因此这里在
    运行过程中周期性轮询并 flush，尽量保留出事前最后一小段内核日志。
    """

    until = now()
    code, stdout, stderr = run_command(
        [
            'journalctl',
            '-k',
            '--since',
            journal_time(since),
            '--until',
            journal_time(until),
            '--no-pager',
            '-o',
            'short-iso',
        ],
        timeout=4.0,
    )
    if code != 0:
        line = f'{now_iso()} journalctl failed: {stderr or stdout}\n'
        kernel_file.write(line)
        nvidia_file.write(line)
        kernel_file.flush()
        nvidia_file.flush()
        return until, []

    xid_lines: list[str] = []
    if stdout:
        kernel_file.write(stdout)
        if not stdout.endswith('\n'):
            kernel_file.write('\n')
        for line in stdout.splitlines():
            lowered = line.lower()
            if any(keyword in lowered for keyword in NVIDIA_LOG_KEYWORDS):
                nvidia_file.write(line + '\n')
                xid_lines.append(line)

    kernel_file.flush()
    nvidia_file.flush()
    return until, xid_lines


def build_log_dir(args: argparse.Namespace) -> Path:
    """确定本次压力测试日志目录。"""

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = DEFAULT_LOG_ROOT / timestamp_for_path()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description='独立 PyTorch CUDA 压力测试并采集诊断日志。')
    parser.add_argument('--log-dir', default=None, help='日志输出目录，默认 runs/cuda_stress/<时间戳>/。')
    parser.add_argument('--duration', type=float, default=300.0, help='压力测试持续秒数。')
    parser.add_argument('--matrix-size', type=int, default=4096, help='矩阵乘法尺寸 N，计算为 [N,N] @ [N,N]。')
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32', help='矩阵计算 dtype。')
    parser.add_argument('--device', default='cuda:0', help='CUDA 设备，例如 cuda 或 cuda:0。')
    parser.add_argument('--sample-interval', type=float, default=3.0, help='指标采样间隔秒数。')
    parser.add_argument('--journal-poll-interval', type=float, default=6.0, help='内核日志轮询间隔秒数。')
    parser.add_argument('--nvidia-timeout', type=float, default=2.0, help='每次 nvidia-smi 查询超时时间。')
    parser.add_argument('--reserve-gpu-memory-mb', type=int, default=0, help='额外预留显存 MiB，用于模拟训练显存占用。')
    parser.add_argument('--max-temperature-c', type=float, default=86.0, help='GPU 温度达到该值时主动停止。')
    parser.add_argument('--stop-on-xid', action='store_true', help='轮询到 Xid/NVIDIA 错误日志后主动停止。')
    parser.add_argument('--allow-tf32', action='store_true', help='允许 float32 matmul 使用 TF32。')
    parser.add_argument('--seed', type=int, default=0, help='随机种子。')
    parser.add_argument('--quiet', action='store_true', help='不打印每轮心跳，只写日志。')
    return parser.parse_args()


def torch_dtype(torch_module, dtype_name: str):
    """把命令行 dtype 映射到 PyTorch dtype。"""

    if dtype_name == 'float16':
        return torch_module.float16
    if dtype_name == 'bfloat16':
        return torch_module.bfloat16
    return torch_module.float32


def reserve_gpu_memory(torch_module, device, reserve_mb: int):
    """按 MiB 预留一块 GPU 显存。

    这里使用 `uint8` 张量，避免额外计算。返回的张量必须由调用方持有引用，否则
    Python 垃圾回收后显存会释放。
    """

    if reserve_mb <= 0:
        return None
    return torch_module.empty(reserve_mb * 1024 * 1024, dtype=torch_module.uint8, device=device)


def run_stress(args: argparse.Namespace) -> int:
    """运行 CUDA 压力测试主流程。"""

    if args.duration <= 0:
        raise ValueError('--duration must be positive')
    if args.matrix_size <= 0:
        raise ValueError('--matrix-size must be positive')
    if args.sample_interval <= 0:
        raise ValueError('--sample-interval must be positive')

    log_dir = build_log_dir(args)
    run_started_at = time.monotonic()
    should_stop = False
    stop_reason = 'duration_reached'

    def request_stop(signum, _frame) -> None:
        nonlocal should_stop, stop_reason
        should_stop = True
        stop_reason = f'signal_{signum}'
        print(f'[cuda-stress] received signal {signum}, stopping...', flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    metrics_fields = [
        'timestamp',
        'elapsed_sec',
        'iteration',
        'matrix_size',
        'dtype',
        'device',
        'interval_iterations',
        'total_tflops_estimate',
        'process_rss_mb',
        'process_vms_mb',
        'mem_available_mb',
        'swap_used_mb',
        'gpu_name',
        'gpu_temp_c',
        'gpu_util_percent',
        'gpu_memory_util_percent',
        'gpu_memory_used_mb',
        'gpu_memory_total_mb',
        'gpu_power_draw_w',
        'gpu_power_limit_w',
        'gpu_clock_sm_mhz',
        'gpu_clock_mem_mhz',
        'gpu_pstate',
        'gpu_query_error',
        'last_checksum',
    ]
    metrics_file, metrics_writer = open_csv(log_dir / 'metrics.csv', metrics_fields)
    events_file = (log_dir / 'events.jsonl').open('w', encoding='utf-8')
    kernel_file = (log_dir / 'journal_kernel_poll.log').open('w', encoding='utf-8')
    nvidia_log_file = (log_dir / 'journal_nvidia_poll.log').open('w', encoding='utf-8')

    metadata = {
        'timestamp': now_iso(),
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
        'cwd': str(Path.cwd()),
        'argv': sys.argv,
        'args': vars(args),
    }
    (log_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    write_jsonl(events_file, {'timestamp': now_iso(), 'event': 'stress_started', 'details': metadata})

    print(f'[cuda-stress] logging to {log_dir}', flush=True)
    append_preflight_report(log_dir / 'preflight.txt')

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError('torch.cuda.is_available() is False')

        device = torch.device(args.device)
        dtype = torch_dtype(torch, args.dtype)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

        device_name = torch.cuda.get_device_name(device)
        write_jsonl(
            events_file,
            {
                'timestamp': now_iso(),
                'event': 'cuda_ready',
                'details': {
                    'torch_version': torch.__version__,
                    'cuda_version': torch.version.cuda,
                    'device': str(device),
                    'device_name': device_name,
                    'allow_tf32': args.allow_tf32,
                },
            },
        )

        reserved = reserve_gpu_memory(torch, device, args.reserve_gpu_memory_mb)
        if reserved is not None:
            write_jsonl(
                events_file,
                {
                    'timestamp': now_iso(),
                    'event': 'reserved_gpu_memory',
                    'details': {'reserve_gpu_memory_mb': args.reserve_gpu_memory_mb},
                },
            )

        # 固定两块矩阵循环相乘，避免每轮重复分配导致压力测试混入大量 allocator 噪声。
        a = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=dtype)
        b = torch.randn((args.matrix_size, args.matrix_size), device=device, dtype=dtype)
        c = torch.empty((args.matrix_size, args.matrix_size), device=device, dtype=dtype)
        torch.cuda.synchronize(device)

        stress_started_at = time.monotonic()
        last_sample_at = stress_started_at - args.sample_interval
        last_journal_poll_at = stress_started_at - args.journal_poll_interval
        journal_since = now()
        last_iteration = 0
        iteration = 0
        last_checksum = 0.0
        peak_gpu_memory = 0.0
        peak_temperature = 0.0
        normalization = 1.0 / args.matrix_size

        while not should_stop:
            elapsed = time.monotonic() - stress_started_at
            if elapsed >= args.duration:
                break

            # out= 参数让结果张量复用同一块显存；每轮 swap a/c，让输入持续变化一点，
            # 同时避免把大量历史结果留在显存里。
            torch.mm(a, b, out=c)
            c.mul_(normalization)
            a, c = c, a
            iteration += 1

            current_time = time.monotonic()
            should_sample = current_time - last_sample_at >= args.sample_interval
            should_poll_journal = current_time - last_journal_poll_at >= args.journal_poll_interval
            if not should_sample and not should_poll_journal:
                continue

            torch.cuda.synchronize(device)

            if should_poll_journal:
                journal_since, xid_lines = poll_kernel_logs(journal_since, kernel_file, nvidia_log_file)
                last_journal_poll_at = current_time
                if xid_lines:
                    write_jsonl(
                        events_file,
                        {
                            'timestamp': now_iso(),
                            'event': 'nvidia_kernel_log_detected',
                            'details': {'lines': xid_lines[-10:]},
                        },
                    )
                    if args.stop_on_xid:
                        should_stop = True
                        stop_reason = 'xid_detected'

            if not should_sample:
                continue

            sample_elapsed = current_time - last_sample_at
            interval_iterations = iteration - last_iteration
            # 一个 N x N 矩阵乘法大约是 2*N^3 次浮点操作。
            total_tflops = 0.0
            if sample_elapsed > 0:
                total_tflops = interval_iterations * 2 * (args.matrix_size ** 3) / sample_elapsed / 1e12

            last_checksum = float(a[0, 0].float().item())
            gpu, gpu_error = query_gpu(args.nvidia_timeout)
            meminfo = read_meminfo()
            self_mem = read_self_memory_mb()
            gpu_memory_used = gpu.get('memory_used_total_mb')
            gpu_temp = gpu.get('temperature_max_c')
            if isinstance(gpu_memory_used, (int, float)):
                peak_gpu_memory = max(peak_gpu_memory, float(gpu_memory_used))
            if isinstance(gpu_temp, (int, float)):
                peak_temperature = max(peak_temperature, float(gpu_temp))

            if isinstance(gpu_temp, (int, float)) and float(gpu_temp) >= args.max_temperature_c:
                should_stop = True
                stop_reason = 'max_temperature_reached'
                write_jsonl(
                    events_file,
                    {
                        'timestamp': now_iso(),
                        'event': 'max_temperature_reached',
                        'details': {'gpu_temp_c': gpu_temp, 'max_temperature_c': args.max_temperature_c},
                    },
                )

            metrics_writer.writerow(
                {
                    'timestamp': now_iso(),
                    'elapsed_sec': round(elapsed, 3),
                    'iteration': iteration,
                    'matrix_size': args.matrix_size,
                    'dtype': args.dtype,
                    'device': str(device),
                    'interval_iterations': interval_iterations,
                    'total_tflops_estimate': round(total_tflops, 4),
                    'process_rss_mb': round(self_mem['process_rss_mb'], 1),
                    'process_vms_mb': round(self_mem['process_vms_mb'], 1),
                    'mem_available_mb': round(meminfo['mem_available_mb'], 1),
                    'swap_used_mb': round(meminfo['swap_used_mb'], 1),
                    'gpu_name': gpu.get('name', ''),
                    'gpu_temp_c': '' if gpu_temp is None else round(float(gpu_temp), 1),
                    'gpu_util_percent': gpu.get('gpu_util_max_percent', ''),
                    'gpu_memory_util_percent': gpu.get('memory_util_percent', ''),
                    'gpu_memory_used_mb': '' if gpu_memory_used is None else round(float(gpu_memory_used), 1),
                    'gpu_memory_total_mb': gpu.get('memory_total_all_mb', ''),
                    'gpu_power_draw_w': gpu.get('power_draw_total_w', ''),
                    'gpu_power_limit_w': gpu.get('power_limit_w', ''),
                    'gpu_clock_sm_mhz': gpu.get('clock_sm_mhz', ''),
                    'gpu_clock_mem_mhz': gpu.get('clock_mem_mhz', ''),
                    'gpu_pstate': gpu.get('pstate', ''),
                    'gpu_query_error': gpu_error or '',
                    'last_checksum': round(last_checksum, 6),
                }
            )
            metrics_file.flush()

            if not args.quiet:
                print(
                    f'[cuda-stress] elapsed={elapsed:6.1f}s '
                    f'iter={iteration:<6d} '
                    f'tflops={total_tflops:6.2f} '
                    f'gpu_mem={gpu_memory_used or 0:.0f}MiB '
                    f'temp={gpu_temp or 0:.0f}C '
                    f'gpu_util={gpu.get("gpu_util_max_percent") or 0:.0f}% '
                    f'checksum={last_checksum:+.4f}',
                    flush=True,
                )

            last_sample_at = current_time
            last_iteration = iteration

        torch.cuda.synchronize(device)
        del a
        del b
        del c
        del reserved
        torch.cuda.empty_cache()

        # 测试结束时再补一轮日志轮询，尽量把最后几秒的 Xid/modeset 信息写进文件。
        journal_since, xid_lines = poll_kernel_logs(journal_since, kernel_file, nvidia_log_file)
        if xid_lines:
            write_jsonl(
                events_file,
                {
                    'timestamp': now_iso(),
                    'event': 'nvidia_kernel_log_detected_at_shutdown',
                    'details': {'lines': xid_lines[-10:]},
                },
            )

        total_elapsed = time.monotonic() - stress_started_at
        summary = {
            'timestamp': now_iso(),
            'elapsed_sec': round(total_elapsed, 3),
            'stop_reason': stop_reason,
            'iterations': iteration,
            'matrix_size': args.matrix_size,
            'dtype': args.dtype,
            'device': str(device),
            'device_name': device_name,
            'reserve_gpu_memory_mb': args.reserve_gpu_memory_mb,
            'peak_gpu_memory_used_mb': round(peak_gpu_memory, 1),
            'peak_temperature_c': round(peak_temperature, 1),
            'last_checksum': last_checksum,
            'log_dir': str(log_dir),
        }
        (log_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        write_jsonl(events_file, {'timestamp': now_iso(), 'event': 'stress_stopped', 'details': summary})
        append_preflight_report(log_dir / 'postflight.txt')

        print(
            f'[cuda-stress] stopped reason={stop_reason} '
            f'elapsed={total_elapsed:.1f}s iterations={iteration} '
            f'peak_gpu_mem={peak_gpu_memory:.0f}MiB peak_temp={peak_temperature:.0f}C '
            f'log_dir={log_dir}',
            flush=True,
        )
        return 0
    except Exception as exc:
        write_jsonl(
            events_file,
            {
                'timestamp': now_iso(),
                'event': 'stress_failed',
                'details': {'error': repr(exc)},
            },
        )
        (log_dir / 'summary.json').write_text(
            json.dumps(
                {
                    'timestamp': now_iso(),
                    'elapsed_sec': round(time.monotonic() - run_started_at, 3),
                    'stop_reason': 'exception',
                    'error': repr(exc),
                    'log_dir': str(log_dir),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        print(f'[cuda-stress] fatal: {exc}', file=sys.stderr, flush=True)
        return 1
    finally:
        for file in (metrics_file, events_file, kernel_file, nvidia_log_file):
            file.close()


def main() -> int:
    """命令行入口。"""

    return run_stress(parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
