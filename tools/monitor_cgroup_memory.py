#!/usr/bin/env python3
"""Monitor cgroup-v2 memory pressure during one training process lifetime.

``memory.current`` includes filesystem page cache.  Treating
``memory.max - memory.current`` as the only available-memory signal therefore
creates false alarms while replay segments and checkpoints are being written.
This monitor records both the raw headroom and the reclaim-aware working-set
headroom:

``working_set = max(memory.current - inactive_file, 0)``

The health decision uses working-set headroom together with ``memory.events``.
The raw value is still preserved in every row for diagnosing write bursts.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_CGROUP_ROOT = Path('/sys/fs/cgroup')
DEFAULT_TARGET_MARKER = 'daxigua_rl.scripts.train_dqn'
EVENT_NAMES = ('low', 'high', 'max', 'oom', 'oom_kill')
stop_requested = False


@dataclass(frozen=True)
class MemorySnapshot:
    """One cgroup-v2 memory sample, expressed in bytes."""

    current: int
    maximum: int
    inactive_file: int
    working_set: int
    raw_available: int
    effective_available: int
    events: dict[str, int]


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def handle_signal(_signum, _frame):
    global stop_requested
    stop_requested = True


def read_bounded_int(path):
    text = path.read_text(encoding='utf-8').strip()
    if text == 'max':
        return None
    return int(text)


def read_keyed_ints(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        name, value = line.split()
        result[name] = int(value)
    return result


def read_memory_snapshot(cgroup_root=DEFAULT_CGROUP_ROOT):
    """Read one cgroup-v2 sample and separate page cache from working set."""

    root = Path(cgroup_root)
    maximum = read_bounded_int(root / 'memory.max')
    if maximum is None:
        raise RuntimeError('cgroup memory.max is unlimited')
    current = int(read_bounded_int(root / 'memory.current') or 0)
    memory_stat = read_keyed_ints(root / 'memory.stat')
    inactive_file = max(0, memory_stat.get('inactive_file', 0))
    working_set = max(0, current - inactive_file)
    return MemorySnapshot(
        current=current,
        maximum=maximum,
        inactive_file=inactive_file,
        working_set=working_set,
        raw_available=max(0, maximum - current),
        effective_available=max(0, maximum - working_set),
        events=read_keyed_ints(root / 'memory.events'),
    )


def target_pids(marker=DEFAULT_TARGET_MARKER):
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / 'cmdline')
                .read_bytes()
                .replace(b'\0', b' ')
                .decode('utf-8', errors='replace')
            )
        except OSError:
            continue
        if marker in command:
            result.append(int(entry.name))
    return sorted(result)


def process_rss_bytes(pid):
    try:
        lines = Path(f'/proc/{pid}/status').read_text(
            encoding='utf-8',
        ).splitlines()
    except OSError:
        return 0
    for line in lines:
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    return 0


def event_deltas(starting, ending):
    return {
        name: ending.get(name, 0) - starting.get(name, 0)
        for name in EVENT_NAMES
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            '记录训练期间 cgroup-v2 原始内存余量、可回收页缓存、工作集余量'
            '和 OOM 事件。'
        ),
    )
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--interval', type=float, default=3.0)
    parser.add_argument(
        '--target-marker',
        default=DEFAULT_TARGET_MARKER,
    )
    parser.add_argument(
        '--min-effective-available-gb',
        type=float,
        default=4.0,
    )
    parser.add_argument(
        '--cgroup-root',
        default=str(DEFAULT_CGROUP_ROOT),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def run_monitor(args):
    if args.interval <= 0:
        raise ValueError('--interval must be positive')
    if args.min_effective_available_gb < 0:
        raise ValueError('--min-effective-available-gb must be non-negative')

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / 'cgroup_metrics.csv'
    summary_path = log_dir / 'cgroup_summary.json'
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError('cgroup monitor outputs already exist')

    cgroup_root = Path(args.cgroup_root).expanduser().resolve()
    first = read_memory_snapshot(cgroup_root)
    starting_events = first.events
    peak_current = 0
    peak_working_set = 0
    peak_inactive_file = 0
    min_raw_available = first.maximum
    min_effective_available = first.maximum
    max_target_rss = 0
    samples = 0
    target_seen = False
    target_exit_observed = False

    fields = (
        'timestamp',
        'elapsed_sec',
        'sample',
        'memory_current_bytes',
        'memory_max_bytes',
        'memory_raw_available_bytes',
        'memory_inactive_file_bytes',
        'memory_working_set_bytes',
        'memory_effective_available_bytes',
        'target_process_count',
        'target_rss_bytes',
        'event_low',
        'event_high',
        'event_max',
        'event_oom',
        'event_oom_kill',
    )
    started = time.monotonic()
    with csv_path.open(
            'w', encoding='utf-8', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        while not stop_requested:
            loop_started = time.monotonic()
            pids = target_pids(args.target_marker)
            if pids:
                target_seen = True
            elif target_seen:
                target_exit_observed = True

            snapshot = read_memory_snapshot(cgroup_root)
            rss = sum(process_rss_bytes(pid) for pid in pids)
            peak_current = max(peak_current, snapshot.current)
            peak_working_set = max(
                peak_working_set,
                snapshot.working_set,
            )
            peak_inactive_file = max(
                peak_inactive_file,
                snapshot.inactive_file,
            )
            min_raw_available = min(
                min_raw_available,
                snapshot.raw_available,
            )
            min_effective_available = min(
                min_effective_available,
                snapshot.effective_available,
            )
            max_target_rss = max(max_target_rss, rss)
            writer.writerow({
                'timestamp': now_iso(),
                'elapsed_sec': round(loop_started - started, 3),
                'sample': samples,
                'memory_current_bytes': snapshot.current,
                'memory_max_bytes': snapshot.maximum,
                'memory_raw_available_bytes': snapshot.raw_available,
                'memory_inactive_file_bytes': snapshot.inactive_file,
                'memory_working_set_bytes': snapshot.working_set,
                'memory_effective_available_bytes': (
                    snapshot.effective_available
                ),
                'target_process_count': len(pids),
                'target_rss_bytes': rss,
                **{
                    f'event_{name}': snapshot.events.get(name, 0)
                    for name in EVENT_NAMES
                },
            })
            file_obj.flush()
            samples += 1
            if target_exit_observed:
                break
            elapsed = time.monotonic() - loop_started
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)

    ending = read_memory_snapshot(cgroup_root)
    deltas = event_deltas(starting_events, ending.events)
    threshold_bytes = int(
        args.min_effective_available_gb * 1024 ** 3
    )
    healthy = (
        target_seen
        and target_exit_observed
        and min_effective_available >= threshold_bytes
        and all(value == 0 for value in deltas.values())
    )
    summary = {
        'timestamp': now_iso(),
        'samples': samples,
        'target_seen': target_seen,
        'target_exit_observed': target_exit_observed,
        'memory_max_bytes': first.maximum,
        'peak_memory_current_bytes': peak_current,
        'peak_memory_working_set_bytes': peak_working_set,
        'peak_memory_inactive_file_bytes': peak_inactive_file,
        'min_memory_raw_available_bytes': min_raw_available,
        'min_memory_effective_available_bytes': (
            min_effective_available
        ),
        'min_effective_available_required_bytes': threshold_bytes,
        'max_target_rss_bytes': max_target_rss,
        'working_set_formula': (
            'max(memory.current - memory.stat[inactive_file], 0)'
        ),
        'starting_events': starting_events,
        'ending_events': ending.events,
        'event_deltas': deltas,
        'healthy': healthy,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return 0 if healthy else 1


def main(argv=None):
    try:
        return run_monitor(parse_args(argv))
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    'healthy': False,
                    'error': f'{type(exc).__name__}: {exc}',
                },
                ensure_ascii=False,
            ),
        )
        return 2


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    raise SystemExit(main())
