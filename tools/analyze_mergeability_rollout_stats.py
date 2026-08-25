"""离线分析场景可合成性变化量并生成阈值参考图片。"""

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

from daxigua.rl.merge_potential_stats import (  # noqa: E402
    load_table_columns,
)


SIGNED_QUANTILES = (
    0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25,
    0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0,
)
MAGNITUDE_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999)
LOAD_COLUMNS = (
    'episode_id',
    'episode_drop',
    'fruit_count',
    'scene_mergeability',
    'occupied_area',
    'area_weighted_mean',
    'delta',
    'delta_valid',
    'done',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='分析可合成性长期rollout分片并输出统计图片。'
    )
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sample-rows', type=int, default=250_000)
    return parser.parse_args(argv)


def _finite(values):
    return values[torch.isfinite(values)]


def _quantiles(values, fractions):
    values = _finite(values.to(torch.float32))
    if values.numel() == 0:
        return {str(fraction): math.nan for fraction in fractions}
    fractions_tensor = torch.tensor(fractions, dtype=torch.float32)
    result = torch.quantile(values, fractions_tensor)
    return {
        str(fraction): float(value)
        for fraction, value in zip(fractions, result.tolist(), strict=True)
    }


def _distribution(values):
    values = _finite(values.to(torch.float32))
    if values.numel() == 0:
        return {'count': 0}
    return {
        'count': int(values.numel()),
        'mean': float(values.mean().item()),
        'std': float(values.std(unbiased=False).item()),
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'quantiles': _quantiles(values, SIGNED_QUANTILES),
    }


def _stage_edges(max_drop):
    candidates = [1, 10, 25, 50, 100, 200, 400, 800, 1200, 1600]
    edges = sorted({value for value in candidates if value <= max_drop})
    if not edges or edges[0] != 1:
        edges.insert(0, 1)
    edges.append(max_drop + 1)
    return edges


def _stage_rows(columns, valid):
    drops = columns['episode_drop'][valid].to(torch.int64)
    delta = columns['delta'][valid].to(torch.float32)
    scene_value = columns['scene_mergeability'][valid].to(torch.float32)
    max_drop = int(drops.max().item()) if drops.numel() else 1
    edges = _stage_edges(max_drop)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (drops >= lower) & (drops < upper)
        if not bool(mask.any().item()):
            continue
        stage_delta = delta[mask]
        stage_value = scene_value[mask]
        delta_q = _quantiles(stage_delta, (0.05, 0.25, 0.5, 0.75, 0.95))
        value_q = _quantiles(stage_value, (0.10, 0.50, 0.90))
        rows.append({
            'drop_start': lower,
            'drop_end_inclusive': upper - 1,
            'drop_midpoint': 0.5 * (lower + upper - 1),
            'count': int(mask.sum().item()),
            'delta_p05': delta_q['0.05'],
            'delta_p25': delta_q['0.25'],
            'delta_median': delta_q['0.5'],
            'delta_p75': delta_q['0.75'],
            'delta_p95': delta_q['0.95'],
            'abs_delta_p95': _quantiles(
                stage_delta.abs(), (0.95,)
            )['0.95'],
            'scene_value_p10': value_q['0.1'],
            'scene_value_median': value_q['0.5'],
            'scene_value_p90': value_q['0.9'],
        })
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sample_indices(count, limit):
    limit = min(int(limit), int(count))
    if limit <= 0:
        raise ValueError('sample-rows must be positive')
    if limit == count:
        return torch.arange(count)
    return torch.linspace(0, count - 1, limit).round().to(torch.int64)


def _plot_distribution(delta, positive, negative_magnitude, summary, output):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    clipped = _quantiles(delta, (0.005, 0.995))
    lower, upper = clipped['0.005'], clipped['0.995']
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        lower, upper = float(delta.min()), float(delta.max() + 1.0)
    axes[0, 0].hist(
        delta.numpy(), bins=180, range=(lower, upper), color='#2563eb', alpha=0.85
    )
    axes[0, 0].axvline(0.0, color='#111827', linewidth=1.0)
    axes[0, 0].set_title('Signed delta histogram (0.5%–99.5% range)')
    axes[0, 0].set_xlabel('Δ scene mergeability')
    axes[0, 0].set_ylabel('samples')

    sorted_delta = delta.sort().values
    ecdf_rows = _sample_indices(sorted_delta.numel(), min(20_000, sorted_delta.numel()))
    ecdf_x = sorted_delta[ecdf_rows].numpy()
    ecdf_y = (
        (ecdf_rows.to(torch.float64) + 1.0) / float(sorted_delta.numel())
    ).numpy()
    axes[0, 1].plot(ecdf_x, ecdf_y, color='#0f766e')
    axes[0, 1].axvline(0.0, color='#111827', linewidth=1.0)
    axes[0, 1].set_xlim(lower, upper)
    axes[0, 1].set_title('Empirical CDF (clipped x-axis)')
    axes[0, 1].set_xlabel('Δ scene mergeability')
    axes[0, 1].set_ylabel('cumulative probability')

    for values, label, color in (
            (positive, 'positive Δ', '#16a34a'),
            (negative_magnitude, '|negative Δ|', '#dc2626')):
        if values.numel() == 0:
            continue
        sorted_values = values.sort().values
        rows = _sample_indices(sorted_values.numel(), min(20_000, sorted_values.numel()))
        x = sorted_values[rows].clamp_min(1e-6).numpy()
        survival = (
            1.0 - rows.to(torch.float64) / float(sorted_values.numel())
        ).clamp_min(1e-6).numpy()
        axes[1, 0].plot(x, survival, label=label, color=color)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Positive / negative magnitude survival')
    axes[1, 0].set_xlabel('magnitude')
    axes[1, 0].set_ylabel('P(magnitude ≥ x)')
    axes[1, 0].legend()

    signed = summary['delta']['quantiles']
    table_rows = []
    for label, fraction in (
            ('0.1%', '0.001'), ('0.5%', '0.005'), ('1%', '0.01'),
            ('2.5%', '0.025'), ('5%', '0.05'), ('50%', '0.5'),
            ('95%', '0.95'), ('97.5%', '0.975'), ('99%', '0.99'),
            ('99.5%', '0.995'), ('99.9%', '0.999')):
        table_rows.append((label, f"{signed[fraction]:,.2f}"))
    axes[1, 1].axis('off')
    table = axes[1, 1].table(
        cellText=table_rows,
        colLabels=('signed quantile', 'Δ threshold'),
        loc='center',
        cellLoc='right',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    axes[1, 1].set_title(
        f"valid={summary['valid_delta_rows']:,} · zero={summary['zero_fraction']:.2%}"
    )
    figure.suptitle('Scene mergeability change distribution', fontsize=16)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(figure)


def _plot_stages(stage_rows, output):
    x = np.asarray([row['drop_midpoint'] for row in stage_rows])
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    p05 = np.asarray([row['delta_p05'] for row in stage_rows])
    p25 = np.asarray([row['delta_p25'] for row in stage_rows])
    p50 = np.asarray([row['delta_median'] for row in stage_rows])
    p75 = np.asarray([row['delta_p75'] for row in stage_rows])
    p95 = np.asarray([row['delta_p95'] for row in stage_rows])
    axes[0, 0].fill_between(x, p05, p95, color='#93c5fd', alpha=0.45, label='p05–p95')
    axes[0, 0].fill_between(x, p25, p75, color='#3b82f6', alpha=0.35, label='p25–p75')
    axes[0, 0].plot(x, p50, color='#1d4ed8', marker='o', label='median')
    axes[0, 0].axhline(0.0, color='#111827', linewidth=1.0)
    axes[0, 0].set_title('Delta distribution by episode stage')
    axes[0, 0].set_xlabel('episode drop (bin midpoint)')
    axes[0, 0].legend()

    value_p10 = np.asarray([row['scene_value_p10'] for row in stage_rows])
    value_p50 = np.asarray([row['scene_value_median'] for row in stage_rows])
    value_p90 = np.asarray([row['scene_value_p90'] for row in stage_rows])
    axes[0, 1].fill_between(x, value_p10, value_p90, color='#86efac', alpha=0.4)
    axes[0, 1].plot(x, value_p50, color='#15803d', marker='o')
    axes[0, 1].set_title('Scene mergeability by episode stage')
    axes[0, 1].set_xlabel('episode drop (bin midpoint)')

    axes[1, 0].plot(
        x,
        [row['abs_delta_p95'] for row in stage_rows],
        color='#c2410c', marker='o',
    )
    axes[1, 0].set_title('95th percentile of |Δ| by stage')
    axes[1, 0].set_xlabel('episode drop (bin midpoint)')

    axes[1, 1].bar(
        np.arange(len(stage_rows)),
        [row['count'] for row in stage_rows],
        color='#64748b',
    )
    axes[1, 1].set_xticks(
        np.arange(len(stage_rows)),
        [f"{row['drop_start']}–{row['drop_end_inclusive']}" for row in stage_rows],
        rotation=35,
        ha='right',
    )
    axes[1, 1].set_title('Valid samples per episode stage')
    axes[1, 1].set_ylabel('samples')
    figure.suptitle('Episode-stage diagnostics', fontsize=16)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(figure)


def _plot_relationships(columns, valid, sample_rows, output):
    valid_rows = torch.nonzero(valid, as_tuple=False).flatten()
    selected = valid_rows.index_select(
        0, _sample_indices(valid_rows.numel(), sample_rows)
    )
    delta = columns['delta'][selected].to(torch.float32)
    clip = _quantiles(delta, (0.01, 0.99))
    y_min, y_max = clip['0.01'], clip['0.99']
    y = delta.clamp(y_min, y_max).numpy()
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    relationships = (
        ('scene_mergeability', 'current scene mergeability'),
        ('fruit_count', 'fruit count'),
        ('occupied_area', 'occupied circle area'),
        ('area_weighted_mean', 'area-weighted mean score'),
    )
    for axis, (name, label) in zip(axes.flat, relationships, strict=True):
        x = columns[name][selected].to(torch.float32).numpy()
        plot = axis.hexbin(x, y, gridsize=70, bins='log', mincnt=1, cmap='viridis')
        axis.axhline(0.0, color='#e2e8f0', linewidth=1.0)
        axis.set_xlabel(label)
        axis.set_ylabel('Δ scene mergeability (p01–p99 clipped)')
        figure.colorbar(plot, ax=axis, label='log sample density')
    figure.suptitle('Delta relationships with current state', fontsize=16)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(figure)


def _plot_threshold_reference(delta, positive, negative_magnitude, output):
    rows = []
    positive_q = _quantiles(positive, MAGNITUDE_QUANTILES)
    negative_q = _quantiles(negative_magnitude, MAGNITUDE_QUANTILES)
    absolute_q = _quantiles(delta.abs(), MAGNITUDE_QUANTILES)
    for fraction in MAGNITUDE_QUANTILES:
        key = str(fraction)
        rows.append((
            f'{fraction:.1%}',
            f"{negative_q[key]:,.2f}",
            f"{positive_q[key]:,.2f}",
            f"{absolute_q[key]:,.2f}",
        ))
    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.axis('off')
    table = axis.table(
        cellText=rows,
        colLabels=(
            'within-tail quantile', '|negative Δ|', 'positive Δ', '|all Δ|'
        ),
        loc='center',
        cellLoc='right',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    axis.set_title(
        'Tail magnitude reference\n'
        'Use as threshold evidence; no detector threshold is selected yet.',
        fontsize=15,
        pad=24,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(figure)


def analyze(dataset_dir, *, output_dir=None, sample_rows=250_000):
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None else dataset_dir / 'analysis'
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = load_table_columns(
        dataset_dir, 'scene_values', LOAD_COLUMNS, area='raw'
    )
    if columns['delta'].numel() == 0:
        raise ValueError('dataset has no scene value rows')
    valid = columns['delta_valid'] & torch.isfinite(columns['delta'])
    delta = columns['delta'][valid].to(torch.float32)
    if delta.numel() == 0:
        raise ValueError('dataset has no valid within-episode deltas')
    positive = delta[delta > 0.0]
    negative_magnitude = -delta[delta < 0.0]
    stage_rows = _stage_rows(columns, valid)
    summary = {
        'purpose': 'mergeability_scene_value_delta_analysis',
        'dataset_dir': str(dataset_dir),
        'scene_value_definition': 'sum(mergeability * pi * physics_radius^2)',
        'total_rows': int(columns['delta'].numel()),
        'valid_delta_rows': int(delta.numel()),
        'invalid_boundary_rows': int((~valid).sum().item()),
        'episodes': int(torch.unique(columns['episode_id']).numel()),
        'terminal_rows': int(columns['done'].sum().item()),
        'zero_fraction': float((delta == 0.0).to(torch.float32).mean().item()),
        'positive_fraction': float((delta > 0.0).to(torch.float32).mean().item()),
        'negative_fraction': float((delta < 0.0).to(torch.float32).mean().item()),
        'delta': _distribution(delta),
        'positive_delta': {
            **_distribution(positive),
            'magnitude_quantiles': _quantiles(positive, MAGNITUDE_QUANTILES),
        },
        'negative_delta_magnitude': {
            **_distribution(negative_magnitude),
            'magnitude_quantiles': _quantiles(
                negative_magnitude, MAGNITUDE_QUANTILES
            ),
        },
        'absolute_delta': {
            **_distribution(delta.abs()),
            'magnitude_quantiles': _quantiles(
                delta.abs(), MAGNITUDE_QUANTILES
            ),
        },
        'stage_rows': stage_rows,
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    quantile_rows = [
        {'quantile': fraction, 'signed_delta': value}
        for fraction, value in summary['delta']['quantiles'].items()
    ]
    _write_csv(output_dir / 'delta_quantiles.csv', quantile_rows)
    _write_csv(output_dir / 'stage_summary.csv', stage_rows)
    _plot_distribution(
        delta, positive, negative_magnitude, summary,
        output_dir / '01_delta_distribution.png',
    )
    _plot_stages(stage_rows, output_dir / '02_episode_stage.png')
    _plot_relationships(
        columns, valid, sample_rows,
        output_dir / '03_state_relationships.png',
    )
    _plot_threshold_reference(
        delta, positive, negative_magnitude,
        output_dir / '04_threshold_reference.png',
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'summary': str(output_dir / 'summary.json'),
        'images': [str(path) for path in sorted(output_dir.glob('*.png'))],
        'valid_delta_rows': int(delta.numel()),
    }, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    args = parse_args(argv)
    analyze(
        args.dataset_dir,
        output_dir=args.output_dir,
        sample_rows=args.sample_rows,
    )


if __name__ == '__main__':
    main()
