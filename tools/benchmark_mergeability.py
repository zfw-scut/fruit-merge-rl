"""测量固定批量单水果可合成性计算器的纯计算吞吐。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from daxigua.core import (  # noqa: E402
    MAX_FRUIT_LEVEL,
    merged_fruit_physics_radius,
)
from daxigua.rl.mergeability import (  # noqa: E402
    MergeabilityCalculator,
    MergeabilityConfig,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='测量可合成性计算器本身的 CPU/CUDA 批量吞吐。'
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--fruit-slots', type=int, default=64)
    parser.add_argument('--active-fruits', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260825)
    return parser.parse_args(argv)


def _synthetic_batch(args, device):
    if not 1 <= args.active_fruits <= args.fruit_slots:
        raise ValueError('active-fruits must be within fruit slots')
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    levels = torch.randint(
        1,
        MAX_FRUIT_LEVEL + 1,
        (args.batch_size, args.fruit_slots),
        generator=generator,
        device=device,
    )
    radius_table = torch.tensor(
        [0.0] + [
            float(merged_fruit_physics_radius(level))
            for level in range(1, MAX_FRUIT_LEVEL + 1)
        ],
        device=device,
    )
    radii = radius_table[levels]
    positions = torch.empty(
        (args.batch_size, args.fruit_slots, 2), device=device
    )
    positions[..., 0].uniform_(40.0, 520.0, generator=generator)
    positions[..., 1].uniform_(320.0, 1080.0, generator=generator)
    active = torch.arange(
        args.fruit_slots, device=device
    ).unsqueeze(0) < args.active_fruits
    active = active.expand(args.batch_size, -1)
    positions = positions * active.unsqueeze(-1)
    radii = radii * active
    levels = levels * active
    return positions, radii, levels, active


@torch.inference_mode()
def run(args):
    if args.batch_size <= 0 or args.fruit_slots <= 0:
        raise ValueError('batch-size and fruit-slots must be positive')
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError('warmup must be non-negative and iterations positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    inputs = _synthetic_batch(args, device)
    calculator = MergeabilityCalculator(MergeabilityConfig()).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup):
        calculator.compute(*inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    checksum = None
    for _ in range(args.iterations):
        checksum = calculator.compute(*inputs).score.sum()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    milliseconds = elapsed * 1000.0 / args.iterations
    report = {
        'device': str(device),
        'batch_size': args.batch_size,
        'fruit_slots': args.fruit_slots,
        'active_fruits': args.active_fruits,
        'warmup': args.warmup,
        'iterations': args.iterations,
        'milliseconds_per_batch': milliseconds,
        'environments_per_second': args.batch_size / (milliseconds / 1000.0),
        'active_fruits_per_second': (
            args.batch_size * args.active_fruits / (milliseconds / 1000.0)
        ),
        'checksum': float(checksum.item()),
    }
    if device.type == 'cuda':
        report['peak_allocated_mb'] = (
            torch.cuda.max_memory_allocated(device) / 1024.0 ** 2
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
