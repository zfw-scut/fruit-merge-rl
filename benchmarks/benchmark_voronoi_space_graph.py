"""Measure the CUDA cost of the weighted Voronoi / free-space prototype.

The benchmark deliberately keeps visualization serialization out of the timed
path.  Each component case runs in a child process so that an OOM or timeout is
recorded without invalidating the rest of the matrix.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from daxigua.core import merged_fruit_physics_radius
from daxigua.simulator import (
    PHYSICS_IDENTITY,
    SimulatorConfig,
    TensorVectorSimulator,
    WeightedVoronoiGraphBuilder,
)


MIB = 1024 ** 2
PRESETS = {
    'smoke': {
        'component_batches': (1, 8),
        'active_counts': (12, 48),
        'component_repeats': 1,
        'shadow_batches': (8,),
        'shadow_steps': 1,
        'pre_roll_steps': 8,
    },
    'standard': {
        'component_batches': (1, 16, 64, 256),
        'active_counts': (12, 56),
        'component_repeats': 2,
        'shadow_batches': (32, 128),
        'shadow_steps': 2,
        'pre_roll_steps': 12,
    },
    'stress': {
        'component_batches': (512, 1024),
        'active_counts': (56,),
        'component_repeats': 1,
        'shadow_batches': (256,),
        'shadow_steps': 2,
        'pre_roll_steps': 16,
    },
    'historical': {
        'component_batches': (1536, 1792, 4096),
        'active_counts': (12, 56),
        'component_repeats': 1,
        'shadow_batches': (1536, 1792, 4096),
        'shadow_steps': 1,
        'pre_roll_steps': 12,
    },
}


def _integer_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError('expected comma-separated positive integers')
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--preset', choices=tuple(PRESETS), default='standard')
    parser.add_argument('--sample-spacing', type=float, default=4.0)
    parser.add_argument('--max-fruits', type=int, default=64)
    parser.add_argument('--component-batches', type=_integer_list)
    parser.add_argument('--active-counts', type=_integer_list)
    parser.add_argument('--component-repeats', type=int)
    parser.add_argument('--warmup-repeats', type=int, default=1)
    parser.add_argument('--shadow-batches', type=_integer_list)
    parser.add_argument('--shadow-steps', type=int)
    parser.add_argument('--pre-roll-steps', type=int)
    parser.add_argument('--skip-shadow', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, default=300.0)
    parser.add_argument('--seed', type=int, default=20260815)
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'runs' / 'voronoi_benchmark' / 'report.json',
    )
    parser.add_argument(
        '--single-kind', choices=('component', 'shadow'), help=argparse.SUPPRESS
    )
    parser.add_argument('--batch-size', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--active-count', type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def _sync(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _memory(device: torch.device):
    if device.type != 'cuda':
        return {'peak_allocated_mib': 0.0, 'peak_reserved_mib': 0.0}
    return {
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / MIB,
        'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / MIB,
    }


def _summary(samples):
    ordered = sorted(float(value) for value in samples)
    p90_index = max(0, math.ceil(len(ordered) * 0.9) - 1)
    return {
        'samples_ms': ordered,
        'min_ms': ordered[0],
        'median_ms': statistics.median(ordered),
        'p90_ms': ordered[p90_index],
        'max_ms': ordered[-1],
    }


def _measure(operation, *, device, warmups, repeats):
    last_result = None
    for _ in range(warmups):
        last_result = operation()
        _sync(device)
        del last_result
        last_result = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        result = operation()
        _sync(device)
        samples.append((time.perf_counter() - started) * 1000.0)
        if last_result is not None:
            del last_result
        last_result = result
    return _summary(samples), _memory(device), last_result


def _synthetic_pile(config, *, batch_size, active_count, device):
    """Create a deterministic fixed-capacity pile used only for scaling tests."""

    if active_count > config.max_fruits:
        raise ValueError('active_count exceeds max_fruits')
    slot = torch.arange(config.max_fruits, device=device).view(1, -1)
    environment = torch.arange(batch_size, device=device).view(-1, 1)
    radius_pattern = torch.tensor(tuple(
        float(merged_fruit_physics_radius(level))
        for level in (1, 2, 3, 4, 5, 1, 2, 6)
    ), dtype=torch.float32, device=device)
    inner_width = config.board_width - config.wall_width * 2
    column_spacing = inner_width / 8.0
    floor = config.board_height - config.wall_width
    column = slot % 8
    row = torch.div(slot, 8, rounding_mode='floor')
    jitter_x = ((environment * 17 + slot * 13) % 7 - 3) * 0.55
    jitter_y = ((environment * 11 + slot * 5) % 5 - 2) * 0.45
    x = config.wall_width + (column + 0.5) * column_spacing + jitter_x
    y = floor - 32.0 - row * 76.0 + jitter_y
    positions = torch.stack((x, y), dim=-1).to(torch.float32)
    active = (slot < active_count).expand(batch_size, -1)
    active_radii = radius_pattern[slot % radius_pattern.numel()].expand(
        batch_size, -1
    )
    radii = torch.where(active, active_radii, torch.ones_like(active_radii))
    return positions, radii, active


def _run_component_case(args):
    if args.active_count is None:
        raise SystemExit('--active-count is required for a component case')
    device = torch.device(args.device)
    config = SimulatorConfig(max_fruits=args.max_fruits)
    builder = WeightedVoronoiGraphBuilder(
        config, device=device, sample_spacing=args.sample_spacing
    )
    positions, radii, active = _synthetic_pile(
        config,
        batch_size=args.batch_size,
        active_count=args.active_count,
        device=device,
    )

    def distance_operation():
        grid = builder._nearest_sites(  # benchmark-only phase isolation
            builder._points, positions, radii, active
        )
        centers = builder._nearest_sites(
            builder._cell_centers.reshape(-1, 2), positions, radii, active
        )
        return grid, centers

    with torch.inference_mode():
        distance, distance_memory, distance_result = _measure(
            distance_operation,
            device=device,
            warmups=args.warmup_repeats,
            repeats=args.component_repeats,
        )
        del distance_result
        full, full_memory, graphs = _measure(
            lambda: builder.build(positions, radii, active),
            device=device,
            warmups=args.warmup_repeats,
            repeats=args.component_repeats,
        )

    edge_counts = [int(graph.edge_start.shape[0]) for graph in graphs]
    vertex_counts = [int(graph.vertex_position.shape[0]) for graph in graphs]
    full_median = full['median_ms']
    distance_median = distance['median_ms']
    report = {
        'kind': 'component',
        'status': 'ok',
        'input_family': 'deterministic_pile_stress',
        'batch_size': args.batch_size,
        'max_fruits': args.max_fruits,
        'active_count': args.active_count,
        'sample_spacing': args.sample_spacing,
        'raster_shape': list(builder.raster_shape),
        'distance_field': {**distance, **distance_memory},
        'full_graph': {**full, **full_memory},
        'topology_and_compaction_estimate_ms': max(
            0.0, full_median - distance_median
        ),
        'full_graph_envs_per_second': (
            args.batch_size * 1000.0 / max(full_median, 1e-9)
        ),
        'mean_edge_count': statistics.fmean(edge_counts),
        'max_edge_count': max(edge_counts, default=0),
        'mean_vertex_count': statistics.fmean(vertex_counts),
        'max_vertex_count': max(vertex_counts, default=0),
    }
    return report


def _random_actions(simulator, generator):
    return torch.randint(
        0,
        simulator.config.action_count,
        (simulator.num_envs,),
        device=simulator.device,
        generator=generator,
    )


def _step_and_reset(simulator, generator):
    result = simulator.step(_random_actions(simulator, generator))
    reset_mask = result.physics.done | result.physics.truncated
    if bool(reset_mask.any().item()):
        simulator.reset(reset_mask)
    return result


def _run_shadow_case(args):
    device = torch.device(args.device)
    config = SimulatorConfig.training_fast(max_fruits=args.max_fruits)
    simulator = TensorVectorSimulator(
        args.batch_size, config=config, device=device
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    builder = WeightedVoronoiGraphBuilder(
        config, device=device, sample_spacing=args.sample_spacing
    )
    with torch.inference_mode():
        for _ in range(args.pre_roll_steps):
            _step_and_reset(simulator, generator)
        warmup_graphs = builder.build(
            simulator.positions, simulator.physics_radii, simulator.active
        )
        _sync(device)
        del warmup_graphs
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        physics_samples = []
        graph_samples = []
        edge_counts = []
        active_counts = []
        for _ in range(args.shadow_steps):
            _sync(device)
            started = time.perf_counter()
            _step_and_reset(simulator, generator)
            _sync(device)
            physics_samples.append((time.perf_counter() - started) * 1000.0)

            started = time.perf_counter()
            graphs = builder.build(
                simulator.positions, simulator.physics_radii, simulator.active
            )
            _sync(device)
            graph_samples.append((time.perf_counter() - started) * 1000.0)
            edge_counts.extend(int(graph.edge_start.shape[0]) for graph in graphs)
            active_counts.append(
                float(simulator.active.sum(dim=1).float().mean().item())
            )
            del graphs

    physics = _summary(physics_samples)
    graph = _summary(graph_samples)
    physics_total = sum(physics_samples)
    graph_total = sum(graph_samples)
    transitions = args.batch_size * args.shadow_steps
    baseline_rate = transitions * 1000.0 / max(physics_total, 1e-9)
    shadow_rate = transitions * 1000.0 / max(
        physics_total + graph_total, 1e-9
    )
    return {
        'kind': 'shadow',
        'status': 'ok',
        'input_family': 'real_random_simulator_rollout_state',
        'batch_size': args.batch_size,
        'max_fruits': args.max_fruits,
        'pre_roll_steps': args.pre_roll_steps,
        'measured_steps': args.shadow_steps,
        'sample_spacing': args.sample_spacing,
        'mean_active_fruits': statistics.fmean(active_counts),
        'physics_step': physics,
        'full_graph': graph,
        'baseline_env_steps_per_second': baseline_rate,
        'shadow_env_steps_per_second': shadow_rate,
        'throughput_retained_ratio': shadow_rate / max(baseline_rate, 1e-9),
        'graph_share_of_synchronized_time': graph_total / max(
            physics_total + graph_total, 1e-9
        ),
        'mean_edge_count': statistics.fmean(edge_counts),
        **_memory(device),
        'finite_state': bool(
            torch.isfinite(simulator.positions).all().item()
            and torch.isfinite(simulator.velocities).all().item()
        ),
    }


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _system_info(device_name):
    device = torch.device(device_name)
    info = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python': sys.version,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'device': str(device),
    }
    if device.type == 'cuda' and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        info.update({
            'gpu_name': properties.name,
            'gpu_total_memory_mib': properties.total_memory / MIB,
            'gpu_compute_capability': (
                f'{properties.major}.{properties.minor}'
            ),
            'cudnn': torch.backends.cudnn.version(),
        })
    return info


def _resolved_settings(args):
    preset = PRESETS[args.preset]
    return {
        'component_batches': (
            args.component_batches or preset['component_batches']
        ),
        'active_counts': args.active_counts or preset['active_counts'],
        'component_repeats': (
            args.component_repeats
            if args.component_repeats is not None
            else preset['component_repeats']
        ),
        'shadow_batches': args.shadow_batches or preset['shadow_batches'],
        'shadow_steps': (
            args.shadow_steps
            if args.shadow_steps is not None else preset['shadow_steps']
        ),
        'pre_roll_steps': (
            args.pre_roll_steps
            if args.pre_roll_steps is not None else preset['pre_roll_steps']
        ),
    }


def _child_command(args, settings, *, kind, batch_size, active_count=None):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--device', args.device,
        '--preset', args.preset,
        '--sample-spacing', str(args.sample_spacing),
        '--max-fruits', str(args.max_fruits),
        '--component-repeats', str(settings['component_repeats']),
        '--warmup-repeats', str(args.warmup_repeats),
        '--shadow-steps', str(settings['shadow_steps']),
        '--pre-roll-steps', str(settings['pre_roll_steps']),
        '--seed', str(args.seed),
        '--single-kind', kind,
        '--batch-size', str(batch_size),
    ]
    if active_count is not None:
        command += ['--active-count', str(active_count)]
    return command


def _run_child(command, *, timeout_seconds, metadata):
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return {
            **metadata,
            'status': 'timeout',
            'elapsed_seconds': time.perf_counter() - started,
            'error': f'case exceeded {timeout_seconds:g} seconds',
            'stdout_tail': (error.stdout or '')[-2000:],
            'stderr_tail': (error.stderr or '')[-4000:],
        }
    if completed.returncode != 0:
        return {
            **metadata,
            'status': 'failed',
            'returncode': completed.returncode,
            'elapsed_seconds': time.perf_counter() - started,
            'stdout_tail': completed.stdout[-2000:],
            'stderr_tail': completed.stderr[-4000:],
        }
    result = None
    parse_error = None
    for line in reversed(completed.stdout.splitlines()):
        if not line.strip():
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError as error:
            parse_error = error
            continue
        if isinstance(candidate, dict) and candidate.get('kind') in {
                'component', 'shadow'}:
            result = candidate
            break
    if result is None:
        return {
            **metadata,
            'status': 'invalid_output',
            'elapsed_seconds': time.perf_counter() - started,
            'error': str(parse_error or 'no JSON object found'),
            'stdout_tail': completed.stdout[-4000:],
            'stderr_tail': completed.stderr[-4000:],
        }
    result['process_elapsed_seconds'] = time.perf_counter() - started
    return result


def _csv_rows(report):
    rows = []
    for case in report['cases']:
        row = {
            'kind': case.get('kind'),
            'status': case.get('status'),
            'batch_size': case.get('batch_size'),
            'active_count': case.get('active_count'),
            'mean_active_fruits': case.get('mean_active_fruits'),
            'distance_median_ms': case.get('distance_field', {}).get('median_ms'),
            'full_graph_median_ms': case.get('full_graph', {}).get('median_ms'),
            'topology_estimate_ms': case.get(
                'topology_and_compaction_estimate_ms'
            ),
            'full_graph_envs_per_second': case.get(
                'full_graph_envs_per_second'
            ),
            'baseline_env_steps_per_second': case.get(
                'baseline_env_steps_per_second'
            ),
            'shadow_env_steps_per_second': case.get(
                'shadow_env_steps_per_second'
            ),
            'throughput_retained_ratio': case.get('throughput_retained_ratio'),
            'peak_allocated_mib': case.get(
                'peak_allocated_mib',
                case.get('full_graph', {}).get('peak_allocated_mib'),
            ),
            'mean_edge_count': case.get('mean_edge_count'),
            'error': case.get('error'),
        }
        rows.append(row)
    return rows


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    report['updated_at'] = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    rows = _csv_rows(report)
    csv_path = path.with_suffix('.csv')
    with csv_path.open('w', encoding='utf-8-sig', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=tuple(rows[0]) if rows else ())
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _run_matrix(args):
    settings = _resolved_settings(args)
    voronoi_source = SRC_ROOT / 'daxigua' / 'simulator' / 'voronoi.py'
    report = {
        'format_version': 1,
        'benchmark': 'weighted_voronoi_space_graph_prototype',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'preset': args.preset,
        'physics_identity': PHYSICS_IDENTITY,
        'system': _system_info(args.device),
        'source': {
            'benchmark_sha256': _sha256(Path(__file__).resolve()),
            'voronoi_sha256': _sha256(voronoi_source),
        },
        'settings': {
            **settings,
            'sample_spacing': args.sample_spacing,
            'max_fruits': args.max_fruits,
            'warmup_repeats': args.warmup_repeats,
            'timeout_seconds_per_case': args.timeout_seconds,
            'visualization_serialization_timed': False,
            'component_input_is_real_rollout': False,
            'shadow_input_is_real_rollout': True,
        },
        'cases': [],
    }
    _write_report(args.output, report)

    for active_count in settings['active_counts']:
        blocked = False
        for batch_size in settings['component_batches']:
            metadata = {
                'kind': 'component',
                'batch_size': batch_size,
                'active_count': active_count,
            }
            if blocked:
                result = {
                    **metadata,
                    'status': 'skipped_after_smaller_case_failure',
                }
            else:
                result = _run_child(
                    _child_command(
                        args,
                        settings,
                        kind='component',
                        batch_size=batch_size,
                        active_count=active_count,
                    ),
                    timeout_seconds=args.timeout_seconds,
                    metadata=metadata,
                )
                blocked = result['status'] != 'ok'
            report['cases'].append(result)
            _write_report(args.output, report)

    if not args.skip_shadow:
        blocked = False
        for batch_size in settings['shadow_batches']:
            metadata = {'kind': 'shadow', 'batch_size': batch_size}
            if blocked:
                result = {
                    **metadata,
                    'status': 'skipped_after_smaller_case_failure',
                }
            else:
                result = _run_child(
                    _child_command(
                        args,
                        settings,
                        kind='shadow',
                        batch_size=batch_size,
                    ),
                    timeout_seconds=args.timeout_seconds,
                    metadata=metadata,
                )
                blocked = result['status'] != 'ok'
            report['cases'].append(result)
            _write_report(args.output, report)

    statuses = [case['status'] for case in report['cases']]
    report['status'] = (
        'complete' if statuses and all(status == 'ok' for status in statuses)
        else 'complete_with_failures'
    )
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    settings = _resolved_settings(args)
    args.component_repeats = settings['component_repeats']
    args.shadow_steps = settings['shadow_steps']
    args.pre_roll_steps = settings['pre_roll_steps']
    if args.component_repeats <= 0 or args.warmup_repeats < 0:
        raise ValueError('repeat counts are invalid')
    if args.shadow_steps <= 0 or args.pre_roll_steps < 0:
        raise ValueError('shadow step counts are invalid')
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if args.single_kind is None:
        _run_matrix(args)
        return
    if args.batch_size is None:
        raise SystemExit('--batch-size is required for a single case')
    result = (
        _run_component_case(args)
        if args.single_kind == 'component'
        else _run_shadow_case(args)
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
