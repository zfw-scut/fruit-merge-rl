"""轻量标定当前几何堵塞检测器的端到端训练batch吞吐。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from daxigua.rl.pair_blockage import (  # noqa: E402
    PairBlockageModel,
    SPLIT_TRAIN,
)
from tools.train_pair_blockage import (  # noqa: E402
    _autocast,
    _dataset_paths,
    _iter_split_batches,
    _model_config_from_dataset,
    _resolve_device,
    balanced_sample_weights,
    build_training_balance,
    training_balance_to,
)


def _positive_int_list(value):
    result = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError('batch sizes must be positive integers')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='短跑标定当前几何堵塞检测器训练吞吐。'
    )
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--batch-sizes', type=_positive_int_list,
        default=(4096, 8192, 16384, 32768, 65536),
    )
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--measure-steps', type=int, default=30)
    parser.add_argument('--throughput-tolerance', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=20260824)
    parser.add_argument('--level-embedding-dim', type=int, default=8)
    parser.add_argument('--context-hidden-dim', type=int, default=40)
    parser.add_argument('--head-hidden-dim', type=int, default=48)
    parser.add_argument(
        '--autocast-bfloat16',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args(argv)


def _next_batch(iterator, factory):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = factory()
        return next(iterator), iterator


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _run_steps(
        model, optimizer, iterator, iterator_factory, balance,
        steps, device, autocast_bfloat16):
    samples = 0
    last_loss = math.nan
    for _ in range(int(steps)):
        batch, iterator = _next_batch(iterator, iterator_factory)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, autocast_bfloat16):
            logits = model(batch)
            losses = F.binary_cross_entropy_with_logits(
                logits, batch['label'].float(), reduction='none'
            )
            weights = balanced_sample_weights(batch, balance)
            loss = (losses * weights).sum() / float(logits.numel())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        samples += int(logits.numel())
        last_loss = float(loss.detach().item())
    return samples, last_loss, iterator


def benchmark_candidate(args, paths, balance, device, batch_size):
    torch.manual_seed(int(args.seed))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(int(args.seed))
    model = PairBlockageModel(
        _model_config_from_dataset(args, args.dataset_dir)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4,
        fused=device.type == 'cuda',
    )
    generator = torch.Generator().manual_seed(int(args.seed) + batch_size)

    def iterator_factory():
        return iter(_iter_split_batches(
            paths, SPLIT_TRAIN, batch_size, device,
            shuffle=True, generator=generator,
        ))

    iterator = iterator_factory()
    model.train()
    _, _, iterator = _run_steps(
        model, optimizer, iterator, iterator_factory, balance,
        args.warmup_steps, device, args.autocast_bfloat16,
    )
    _synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    samples, last_loss, _ = _run_steps(
        model, optimizer, iterator, iterator_factory, balance,
        args.measure_steps, device, args.autocast_bfloat16,
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        'batch_size': int(batch_size),
        'status': 'complete',
        'measured_steps': int(args.measure_steps),
        'samples': samples,
        'elapsed_seconds': elapsed,
        'samples_per_second': samples / max(elapsed, 1e-9),
        'mean_step_seconds': elapsed / max(1, int(args.measure_steps)),
        'last_loss': last_loss,
        'peak_cuda_allocated_bytes': (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == 'cuda' else None
        ),
        'peak_cuda_reserved_bytes': (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == 'cuda' else None
        ),
    }
    del optimizer, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def benchmark(args):
    if args.warmup_steps < 0 or args.measure_steps <= 0:
        raise ValueError('benchmark steps are invalid')
    if not 0.0 <= args.throughput_tolerance < 1.0:
        raise ValueError('throughput tolerance must be in [0, 1)')
    _manifest, paths = _dataset_paths(args.dataset_dir)
    device = _resolve_device(args.device)
    balance = training_balance_to(build_training_balance(paths), device)
    rows = []
    for batch_size in args.batch_sizes:
        try:
            row = benchmark_candidate(
                args, paths, balance, device, int(batch_size)
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if 'out of memory' not in str(error).lower():
                raise
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            row = {
                'batch_size': int(batch_size),
                'status': 'out_of_memory',
                'error': str(error),
            }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    valid = [row for row in rows if row['status'] == 'complete']
    if not valid:
        raise RuntimeError('all current-blockage benchmark candidates failed')
    fastest = max(float(row['samples_per_second']) for row in valid)
    threshold = fastest * (1.0 - float(args.throughput_tolerance))
    eligible = [
        row for row in valid
        if float(row['samples_per_second']) >= threshold
    ]
    selected = min(eligible, key=lambda row: int(row['batch_size']))
    result = {
        'format_version': 1,
        'purpose': 'pair_current_blockage_training_throughput_calibration',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_dir': str(Path(args.dataset_dir).resolve()),
        'device': str(device),
        'device_name': (
            torch.cuda.get_device_name(device)
            if device.type == 'cuda' else None
        ),
        'model_parameters': sum(
            parameter.numel()
            for parameter in PairBlockageModel(
                _model_config_from_dataset(args, args.dataset_dir)
            ).parameters()
        ),
        'warmup_steps': int(args.warmup_steps),
        'measure_steps': int(args.measure_steps),
        'throughput_tolerance': float(args.throughput_tolerance),
        'results': rows,
        'fastest_samples_per_second': fastest,
        'selected_batch_size': int(selected['batch_size']),
        'training_balance': balance['summary'],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.output is not None:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    return result


def main(argv=None):
    benchmark(parse_args(argv))


if __name__ == '__main__':
    main()
