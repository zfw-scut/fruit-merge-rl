#!/usr/bin/env python3
"""用已有前后快照复算空间结构分数，并分析合成得分的影响。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from daxigua.rl.mergeability import (  # noqa: E402
    MergeabilityCalculator,
    MergeabilityConfig,
)
from daxigua.rl.mergeability_rollout import (  # noqa: E402
    fruit_material_mass,
    lineage_aligned_spatial_change,
)


DEFAULT_DATASET = (
    PROJECT_ROOT / 'runs' / 'diagnostics'
    / 'mergeability_negative_transitions_sab128_env2000_steps1000_20260825'
    / 'negative_transitions.pt'
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='复算已有成对场景的空间结构变化和合成得分关系。'
    )
    parser.add_argument('dataset', type=Path, nargs='?', default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dpi', type=int, default=150)
    return parser.parse_args(argv)


def _stack(samples, side, name, device):
    return torch.stack([
        sample[side][name] for sample in samples
    ]).to(device=device)


def _stack_events(samples, name, device):
    return torch.stack([
        sample['merge_events'][name] for sample in samples
    ]).to(device=device)


def _current_spatial(levels, active, spatial_score):
    mass = fruit_material_mass(levels, active, dtype=spatial_score.dtype)
    total = mass.sum(dim=1)
    score = torch.where(
        total > 0.0,
        (spatial_score * mass).sum(dim=1) / total.clamp_min(1e-12),
        torch.zeros_like(total),
    )
    return score, total


def _quantiles(values, fractions=(0.05, 0.25, 0.5, 0.75, 0.95)):
    values = values[torch.isfinite(values)].to(torch.float32)
    if values.numel() == 0:
        return {str(value): math.nan for value in fractions}
    q = torch.quantile(values, torch.tensor(fractions, dtype=torch.float32))
    return {
        str(fraction): float(value)
        for fraction, value in zip(fractions, q.tolist(), strict=True)
    }


def _group_code(merge_count):
    return torch.where(
        merge_count == 0,
        torch.zeros_like(merge_count),
        torch.where(
            merge_count == 1,
            torch.ones_like(merge_count),
            torch.where(
                merge_count <= 3,
                torch.full_like(merge_count, 2),
                torch.full_like(merge_count, 3),
            ),
        ),
    )


GROUP_LABELS = ('no merge', 'single merge', '2-3 merges', '4+ merges')
GROUP_COLORS = ('#64748b', '#2563eb', '#f59e0b', '#dc2626')


def _distribution(values):
    values = values[torch.isfinite(values)].to(torch.float32)
    if values.numel() == 0:
        return {'count': 0}
    return {
        'count': int(values.numel()),
        'mean': float(values.mean().item()),
        'std': float(values.std(unbiased=False).item()),
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'negative_fraction': float((values < 0.0).float().mean().item()),
        'near_zero_fraction_abs_le_0_05': float(
            (values.abs() <= 0.05).float().mean().item()
        ),
        'quantiles': _quantiles(values),
    }


def _correlation(x, y):
    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid].to(torch.float64)
    y = y[valid].to(torch.float64)
    if x.numel() < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return math.nan
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1].item())


def _write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_correction(table, output, dpi):
    valid = (
        table['spatial_delta_valid'] & torch.isfinite(table['spatial_delta'])
    ).numpy()
    old_delta = table['old_delta'].numpy()[valid]
    structural = table['spatial_delta'].numpy()[valid]
    groups = table['merge_group'].numpy()[valid]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for code, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        mask = groups == code
        axes[0, 0].scatter(
            old_delta[mask], structural[mask], s=22, alpha=0.72,
            label=label, color=color,
        )
    axes[0, 0].axhline(0.0, color='#111827', linewidth=1)
    axes[0, 0].set_xlabel('old area-weighted delta')
    axes[0, 0].set_ylabel('lineage-aligned spatial delta')
    axes[0, 0].set_title('Old negative selection vs new structural change')
    axes[0, 0].legend()

    group_values = [
        structural[groups == code] for code in range(len(GROUP_LABELS))
    ]
    axes[0, 1].boxplot(group_values, tick_labels=GROUP_LABELS, showfliers=False)
    axes[0, 1].axhline(0.0, color='#111827', linewidth=1)
    axes[0, 1].tick_params(axis='x', rotation=20)
    axes[0, 1].set_ylabel('lineage-aligned spatial delta')
    axes[0, 1].set_title('Structural change by merge class')

    bins = np.linspace(-1.0, 1.0, 101)
    for code, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        values = structural[groups == code]
        if values.size:
            axes[1, 0].hist(
                values, bins=bins, density=True, histtype='step', linewidth=2,
                label=label, color=color,
            )
    axes[1, 0].axvline(0.0, color='#111827', linewidth=1)
    axes[1, 0].set_xlabel('lineage-aligned spatial delta')
    axes[1, 0].set_ylabel('density')
    axes[1, 0].set_title('Spatial delta distribution')
    axes[1, 0].legend()

    current_delta = table['current_scene_spatial_delta'].numpy()[valid]
    axes[1, 1].scatter(current_delta, structural, s=24, alpha=0.7, color='#7c3aed')
    axes[1, 1].axhline(0.0, color='#111827', linewidth=1)
    axes[1, 1].axvline(0.0, color='#111827', linewidth=1)
    axes[1, 1].set_xlabel('naive current-scene spatial delta')
    axes[1, 1].set_ylabel('old-material aligned spatial delta')
    axes[1, 1].set_title('Effect of excluding newly dropped material')
    figure.suptitle(
        'Spatial mergeability correction on the existing 256-pair gallery',
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


def _plot_merge_score(table, output, dpi):
    valid = (
        table['spatial_delta_valid'] & torch.isfinite(table['spatial_delta'])
    ).numpy()
    structural = table['spatial_delta'].numpy()[valid]
    merge_score = table['merge_score'].numpy()[valid]
    merge_count = table['merge_count'].numpy()[valid]
    groups = table['merge_group'].numpy()[valid]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for code, (label, color) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        mask = groups == code
        axes[0, 0].scatter(
            merge_score[mask], structural[mask], s=24, alpha=0.72,
            label=label, color=color,
        )
    axes[0, 0].axhline(0.0, color='#111827', linewidth=1)
    axes[0, 0].set_xlabel('merge score gained by the drop')
    axes[0, 0].set_ylabel('lineage-aligned spatial delta')
    axes[0, 0].set_title('Merge score vs structural change')
    axes[0, 0].legend()

    axes[0, 1].scatter(merge_count, structural, s=24, alpha=0.65, color='#ea580c')
    axes[0, 1].axhline(0.0, color='#111827', linewidth=1)
    axes[0, 1].set_xlabel('merge count')
    axes[0, 1].set_ylabel('lineage-aligned spatial delta')
    axes[0, 1].set_title('Merge chain length vs structural change')

    medians = []
    negatives = []
    for code in range(len(GROUP_LABELS)):
        values = structural[groups == code]
        medians.append(float(np.median(values)) if values.size else math.nan)
        negatives.append(float(np.mean(values < 0.0)) if values.size else math.nan)
    x = np.arange(len(GROUP_LABELS))
    axes[1, 0].bar(x, medians, color=GROUP_COLORS)
    axes[1, 0].axhline(0.0, color='#111827', linewidth=1)
    axes[1, 0].set_xticks(x, GROUP_LABELS, rotation=20, ha='right')
    axes[1, 0].set_ylabel('median spatial delta')
    axes[1, 0].set_title('Median structural effect')

    axes[1, 1].bar(x, negatives, color=GROUP_COLORS)
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_xticks(x, GROUP_LABELS, rotation=20, ha='right')
    axes[1, 1].set_ylabel('fraction below zero')
    axes[1, 1].set_title('Negative structural-change frequency')
    figure.suptitle('Merge outcome and spatial structure', fontsize=16)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(figure)


@torch.inference_mode()
def analyze(dataset_path, *, output_dir=None, device='cuda', dpi=150):
    dataset_path = Path(dataset_path).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None else dataset_path.parent / 'structural_analysis'
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(dataset_path, map_location='cpu', weights_only=False)
    samples = payload.get('samples')
    if not isinstance(samples, list) or not samples:
        raise ValueError('dataset contains no transition samples')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    simulator_config = payload['manifest']['simulator_config']
    config = MergeabilityConfig(
        board_width=float(simulator_config['board_width']),
        spawn_y=float(simulator_config['spawn_y']),
        wall_width=float(simulator_config['wall_width']),
    )
    calculator = MergeabilityCalculator(config).to(device).eval()
    before = {
        name: _stack(samples, 'before', name, device)
        for name in ('positions', 'physics_radii', 'levels', 'active', 'fruit_ids')
    }
    after = {
        name: _stack(samples, 'after', name, device)
        for name in ('positions', 'physics_radii', 'levels', 'active', 'fruit_ids')
    }
    before_result = calculator.compute(
        before['positions'], before['physics_radii'],
        before['levels'], before['active'],
    )
    after_result = calculator.compute(
        after['positions'], after['physics_radii'],
        after['levels'], after['active'],
    )
    before_scene, _ = _current_spatial(
        before['levels'], before['active'], before_result.spatial_score
    )
    after_scene, _ = _current_spatial(
        after['levels'], after['active'], after_result.spatial_score
    )
    merge_count = torch.tensor(
        [int(sample['merge_count']) for sample in samples],
        dtype=torch.int64, device=device,
    )
    merge_score = torch.tensor(
        [int(sample['score_delta']) for sample in samples],
        dtype=torch.float32, device=device,
    )
    change = lineage_aligned_spatial_change(
        before['fruit_ids'], before['levels'], before['active'],
        before_result.spatial_score,
        after['fruit_ids'], after['active'], after_result.spatial_score,
        merge_count,
        _stack_events(samples, 'source_ids', device),
        _stack_events(samples, 'new_fruit_ids', device),
    )
    group = _group_code(merge_count)
    table = {
        'old_delta': torch.tensor(
            [float(sample['delta']) for sample in samples]
        ),
        'before_spatial_scene_score': before_scene.cpu(),
        'after_spatial_scene_score': after_scene.cpu(),
        'current_scene_spatial_delta': (after_scene - before_scene).cpu(),
        'before_aligned_spatial_score': change.before_score.cpu(),
        'after_aligned_spatial_score': change.after_score.cpu(),
        'spatial_delta': change.delta.cpu(),
        'spatial_delta_valid': change.valid.cpu(),
        'lineage_coverage': change.coverage.cpu(),
        'merge_count': merge_count.cpu(),
        'merge_score': merge_score.cpu(),
        'merge_group': group.cpu(),
        'episode_drop': torch.tensor([
            int(sample['episode_drop']) for sample in samples
        ]),
        'terminal': torch.tensor([
            bool(sample['terminal']) for sample in samples
        ]),
    }
    valid = table['spatial_delta_valid'] & torch.isfinite(table['spatial_delta'])
    group_rows = []
    group_summary = {}
    for code, label in enumerate(GROUP_LABELS):
        mask = valid & (table['merge_group'] == code)
        values = table['spatial_delta'][mask]
        stats = _distribution(values)
        stats['merge_score'] = _distribution(table['merge_score'][mask])
        group_summary[label] = stats
        group_rows.append({
            'group': label,
            'count': stats.get('count', 0),
            'delta_mean': stats.get('mean', math.nan),
            'delta_median': stats.get('quantiles', {}).get('0.5', math.nan),
            'negative_fraction': stats.get('negative_fraction', math.nan),
            'near_zero_fraction_abs_le_0_05': stats.get(
                'near_zero_fraction_abs_le_0_05', math.nan
            ),
            'merge_score_mean': stats['merge_score'].get('mean', math.nan),
        })
    summary = {
        'purpose': 'mergeability_spatial_semantics_reanalysis',
        'source_dataset': str(dataset_path),
        'selection_warning': (
            'all 256 samples were selected from the negative tail of the old '
            'area-weighted metric; this is a correction audit, not an unbiased '
            'natural rollout distribution'
        ),
        'device': str(device),
        'sample_count': len(samples),
        'valid_lineage_count': int(valid.sum().item()),
        'full_lineage_fraction': float(valid.float().mean().item()),
        'spatial_delta': _distribution(table['spatial_delta'][valid]),
        'groups': group_summary,
        'correlation_old_delta_vs_spatial_delta': _correlation(
            table['old_delta'][valid], table['spatial_delta'][valid]
        ),
        'correlation_merge_score_vs_spatial_delta': _correlation(
            table['merge_score'][valid], table['spatial_delta'][valid]
        ),
    }
    serializable_rows = []
    for index, sample in enumerate(samples):
        serializable_rows.append({
            'sample_index': index,
            'severity_key': sample['severity_key'],
            'episode_id': int(sample['episode_id']),
            'episode_drop': int(sample['episode_drop']),
            'old_delta': float(table['old_delta'][index]),
            'before_spatial_scene_score': float(
                table['before_spatial_scene_score'][index]
            ),
            'after_spatial_scene_score': float(
                table['after_spatial_scene_score'][index]
            ),
            'current_scene_spatial_delta': float(
                table['current_scene_spatial_delta'][index]
            ),
            'spatial_delta': float(table['spatial_delta'][index]),
            'spatial_delta_valid': bool(table['spatial_delta_valid'][index]),
            'lineage_coverage': float(table['lineage_coverage'][index]),
            'merge_count': int(table['merge_count'][index]),
            'merge_score': int(table['merge_score'][index]),
            'merge_group': GROUP_LABELS[int(table['merge_group'][index])],
            'terminal': bool(table['terminal'][index]),
        })
    torch.save({
        'format_version': 1,
        'summary': summary,
        'columns': table,
    }, output_dir / 'structural_transition_table.pt')
    _write_csv(output_dir / 'transitions.csv', serializable_rows)
    _write_csv(output_dir / 'merge_group_summary.csv', group_rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    _plot_correction(table, output_dir / '01_metric_correction.png', dpi)
    _plot_merge_score(table, output_dir / '02_merge_score_relationship.png', dpi)
    result = {
        'output_dir': str(output_dir),
        'summary': str(output_dir / 'summary.json'),
        'images': [str(path) for path in sorted(output_dir.glob('*.png'))],
        'sample_count': len(samples),
        'valid_lineage_count': int(valid.sum().item()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    args = parse_args(argv)
    analyze(
        args.dataset,
        output_dir=args.output_dir,
        device=args.device,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
