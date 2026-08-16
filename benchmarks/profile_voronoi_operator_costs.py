"""Profile operator-level costs of the current weighted Voronoi builder.

This diagnostic complements the large-batch wall-clock benchmark.  It profiles
a representative small batch because tracing every operation at 1792 or 4096
environments would itself create a very large Kineto trace.  The current
builder executes the same per-environment topology program, so call counts per
environment can be projected to the historical training batch sizes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.profiler import ProfilerActivity, profile

from benchmark_voronoi_space_graph import _synthetic_pile
from daxigua.simulator import SimulatorConfig, WeightedVoronoiGraphBuilder


IMPORTANT_EVENTS = (
    'aten::index',
    'aten::nonzero',
    'cudaStreamSynchronize',
    'cudaMemcpyAsync',
    'cudaLaunchKernel',
    'Memcpy DtoH (Device -> Pinned)',
    'Memcpy DtoD (Device -> Device)',
    'aten::cat',
    'aten::sort',
    'aten::stack',
    'aten::bincount',
    'aten::sum',
    'aten::minimum',
    'aten::masked_fill',
    'aten::min',
    'aten::sqrt',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--active-count', type=int, default=56)
    parser.add_argument('--max-fruits', type=int, default=64)
    parser.add_argument('--sample-spacing', type=float, default=4.0)
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument(
        '--project-env-counts',
        default='1536,1792,4096',
        help='comma-separated environment counts for structural call projection',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=(
            PROJECT_ROOT / 'runs' / 'voronoi_benchmark'
            / 'operator_profile.json'
        ),
    )
    return parser.parse_args()


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _measure(operation, *, device, repeats):
    operation()
    _sync(device)
    samples = []
    result = None
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        result = operation()
        _sync(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        'samples_ms': samples,
        'median_ms': statistics.median(samples),
        'min_ms': min(samples),
        'max_ms': max(samples),
    }, result


def _event_value(event, *names):
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _event_payload(event):
    return {
        'name': event.key,
        'calls': int(event.count),
        'self_cpu_ms': float(event.self_cpu_time_total) / 1000.0,
        'cpu_total_ms': float(event.cpu_time_total) / 1000.0,
        'self_device_ms': _event_value(
            event, 'self_device_time_total', 'self_cuda_time_total'
        ) / 1000.0,
        'device_total_ms': _event_value(
            event, 'device_time_total', 'cuda_time_total'
        ) / 1000.0,
    }


def _profile_once(operation, *, device):
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    _sync(device)
    profiler_started = time.perf_counter()
    with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False) as profiler:
        _sync(device)
        operation_started = time.perf_counter()
        result = operation()
        _sync(device)
        operation_wall_ms = (
            time.perf_counter() - operation_started
        ) * 1000.0
    profiler_total_wall_ms = (
        time.perf_counter() - profiler_started
    ) * 1000.0
    events = [_event_payload(event) for event in profiler.key_averages()]
    by_name = {event['name']: event for event in events}
    important = {
        name: by_name.get(name, {
            'name': name,
            'calls': 0,
            'self_cpu_ms': 0.0,
            'cpu_total_ms': 0.0,
            'self_device_ms': 0.0,
            'device_total_ms': 0.0,
        })
        for name in IMPORTANT_EVENTS
    }
    high_level_events = [
        event for event in events
        if event['name'].startswith(('aten::', 'cuda', 'Memcpy', 'Activity'))
    ]
    top_cpu = sorted(
        high_level_events,
        key=lambda item: item['self_cpu_ms'], reverse=True
    )[:25]
    top_device = sorted(
        high_level_events,
        key=lambda item: item['self_device_ms'], reverse=True
    )[:25]
    return {
        'profiled_operation_wall_ms': operation_wall_ms,
        'profiler_total_wall_ms_including_trace_finalization': (
            profiler_total_wall_ms
        ),
        'important_events': important,
        'top_self_cpu': top_cpu,
        'top_self_device': top_device,
    }, result


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.active_count <= 0:
        raise ValueError('batch_size and active_count must be positive')
    if args.timing_repeats <= 0:
        raise ValueError('timing_repeats must be positive')
    projected_counts = tuple(
        int(item.strip())
        for item in args.project_env_counts.split(',') if item.strip()
    )
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

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
        grid = builder._nearest_sites(
            builder._points, positions, radii, active
        )
        centers = builder._nearest_sites(
            builder._cell_centers.reshape(-1, 2), positions, radii, active
        )
        return grid, centers

    def full_operation():
        return builder.build(positions, radii, active)

    with torch.inference_mode():
        distance_timing, distance_result = _measure(
            distance_operation, device=device, repeats=args.timing_repeats
        )
        del distance_result
        full_timing, graphs = _measure(
            full_operation, device=device, repeats=args.timing_repeats
        )
        edge_count = sum(int(graph.edge_start.shape[0]) for graph in graphs)
        vertex_count = sum(
            int(graph.vertex_position.shape[0]) for graph in graphs
        )
        del graphs
        operator_profile, profiled_graphs = _profile_once(
            full_operation, device=device
        )
        del profiled_graphs

    important = operator_profile['important_events']
    per_environment = {
        name: event['calls'] / args.batch_size
        for name, event in important.items()
    }
    # build() performs one batch-global active-radius validation index.  The
    # remaining index/nonzero calls come from the fixed per-environment
    # topology extraction program and therefore scale exactly with B.
    batch_global_validation_calls = 1
    dynamic_selection_calls_per_environment = (
        important['aten::index']['calls'] - batch_global_validation_calls
    ) / args.batch_size
    projections = {
        str(count): {
            'dynamic_boolean_index_calls': round(
                dynamic_selection_calls_per_environment * count
                + batch_global_validation_calls
            ),
            'dynamic_nonzero_calls': round(
                dynamic_selection_calls_per_environment * count
                + batch_global_validation_calls
            ),
        }
        for count in projected_counts
    }
    full_ms = full_timing['median_ms']
    distance_ms = distance_timing['median_ms']
    report = {
        'format_version': 1,
        'profile': 'weighted_voronoi_operator_costs',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'system': {
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python': sys.version,
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'device': str(device),
            'gpu_name': (
                torch.cuda.get_device_name(device)
                if device.type == 'cuda' else None
            ),
        },
        'input': {
            'family': 'deterministic_pile_stress',
            'batch_size': args.batch_size,
            'active_count': args.active_count,
            'max_fruits': args.max_fruits,
            'sample_spacing': args.sample_spacing,
            'raster_shape': list(builder.raster_shape),
            'timing_repeats': args.timing_repeats,
        },
        'unprofiled_timing': {
            'distance_field': distance_timing,
            'full_graph': full_timing,
            'topology_and_compaction_estimate_ms': max(
                0.0, full_ms - distance_ms
            ),
            'distance_share_of_full': distance_ms / max(full_ms, 1e-9),
            'topology_share_of_full': max(
                0.0, full_ms - distance_ms
            ) / max(full_ms, 1e-9),
            'edges_total': edge_count,
            'vertices_total': vertex_count,
        },
        'operator_profile': operator_profile,
        'calls_per_environment': per_environment,
        'derived_structure': {
            'batch_global_validation_index_calls': (
                batch_global_validation_calls
            ),
            'dynamic_boolean_selections_per_environment': (
                dynamic_selection_calls_per_environment
            ),
        },
        'projected_structural_calls': projections,
        'scope': {
            'includes_json_serialization': False,
            'includes_visual_rendering': False,
            'profiled_batch_is_training_batch': False,
            'projection_is_structural_call_count_not_time': True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
