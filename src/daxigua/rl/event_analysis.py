"""关键合成事件和评估分数分布的标准归档图。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch

from .curves import _configure_font


EVENT_ANALYSIS_FILENAME = 'evaluation_event_analysis.png'
EVENT_ANALYSIS_METADATA_FILENAME = 'evaluation_event_analysis.json'


def _histogram(scores, width, maximum=None):
    scores = scores.to(torch.int64)
    maximum = int(scores.max().item()) if maximum is None else int(maximum)
    bin_count = max(1, maximum // width + 1)
    indices = scores.div(width, rounding_mode='floor').clamp(0, bin_count - 1)
    counts = torch.bincount(indices, minlength=bin_count)
    return counts


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)


def render_evaluation_event_analysis(
        run_dir, index_30fps_path, index_120fps_path, *, score_bin_width=500):
    """生成 30/120 FPS 分布、L11 条件分布与关键事件比例图。"""

    run_dir = Path(run_dir)
    index30 = torch.load(
        index_30fps_path, map_location='cpu', weights_only=False
    )
    index120 = torch.load(
        index_120fps_path, map_location='cpu', weights_only=False
    )
    scores30 = index30['scores'].to(torch.int64)
    scores120 = index120['scores'].to(torch.int64)
    maximum = max(int(scores30.max()), int(scores120.max()))
    counts30 = _histogram(scores30, score_bin_width, maximum)
    counts120 = _histogram(scores120, score_bin_width, maximum)
    created30 = index30['created_level_counts'][:, 11] > 0
    removed30 = index30['source_level_merge_counts'][:, 11] > 0
    categories = (
        ('未生成 L11', 'No L11 created', ~created30),
        ('生成但未消除 L11', 'L11 created, not removed',
         created30 & ~removed30),
        ('已消除 L11', 'L11 removed', removed30),
    )
    bin_starts = torch.arange(counts30.numel()) * int(score_bin_width)
    event_levels = tuple(range(8, 12))
    event_episode_rates = []
    for level in event_levels:
        event_episode_rates.append(float(
            (index30['created_level_counts'][:, level] > 0)
            .to(torch.float32).mean().item()
        ))

    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(plot_dir / '.mplconfig'))
    import matplotlib

    matplotlib.use('Agg', force=True)
    has_chinese_font = _configure_font(matplotlib)
    import matplotlib.pyplot as plt

    output = plot_dir / EVENT_ANALYSIS_FILENAME
    temporary = output.with_suffix('.png.tmp')
    figure, axes = plt.subplots(
        2, 2, figsize=(14, 8.8), constrained_layout=True
    )
    figure.patch.set_facecolor('#f3f6fa')
    title = (
        '评估分数密度与高等级关键事件'
        if has_chinese_font else
        'Evaluation Score Density and High-level Events'
    )
    figure.suptitle(title, fontsize=18, fontweight='bold')
    colors = ('#0067c0', '#8764b8', '#107c10', '#d13438')
    x = bin_starts.numpy()

    def label(chinese, english):
        return f'{chinese} / {english}' if has_chinese_font else english

    axis = axes[0, 0]
    axis.step(x, counts30.numpy() / max(1, scores30.numel()), where='post',
              label='30 FPS', color=colors[0], linewidth=2)
    axis.step(x, counts120.numpy() / max(1, scores120.numel()), where='post',
              label='120 FPS', color=colors[1], linewidth=2)
    axis.set_title(label('分数密度', 'Score density'),
                   loc='left', fontweight='bold')
    axis.set_xlabel(label('分数', 'Score'))
    axis.set_ylabel(label('对局比例', 'Episode ratio'))
    axis.legend(frameon=False)

    axis = axes[0, 1]
    for (chinese, english, mask), color in zip(categories, colors[1:]):
        subset = scores30[mask]
        if subset.numel():
            values = _histogram(subset, score_bin_width, maximum)
            axis.step(
                x, values.numpy() / subset.numel(), where='post',
                label=chinese if has_chinese_font else english,
                color=color, linewidth=1.8,
            )
    axis.set_title(label('30 FPS 的 L11 条件分布',
                         '30 FPS L11-conditioned distribution'),
                   loc='left', fontweight='bold')
    axis.set_xlabel(label('分数', 'Score'))
    axis.set_ylabel(label('条件比例', 'Conditional ratio'))
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1, 0]
    labels = [f'L{level}' for level in event_levels]
    axis.bar(labels, event_episode_rates, color=colors)
    axis.set_ylim(0.0, max(0.05, max(event_episode_rates, default=0.0) * 1.18))
    axis.set_title(label('生成高等级水果的对局比例',
                         'Episodes creating high-level fruits'),
                   loc='left', fontweight='bold')
    axis.set_ylabel(label('对局比例', 'Episode ratio'))
    for index, value in enumerate(event_episode_rates):
        axis.text(index, value, f'{value:.1%}', ha='center', va='bottom')

    axis = axes[1, 1]
    source_counts = index30['source_level_merge_counts'].to(torch.float64)
    mean_counts = source_counts[:, 8:12].mean(dim=0)
    axis.bar(labels, mean_counts.numpy(), color=colors)
    axis.set_title(label('每局高等级合成次数',
                         'High-level merges per episode'),
                   loc='left', fontweight='bold')
    axis.set_ylabel(label('平均次数', 'Mean count'))
    for index, value in enumerate(mean_counts.tolist()):
        axis.text(index, value, f'{value:.2f}', ha='center', va='bottom')

    for axis in axes.flat:
        axis.set_facecolor('#ffffff')
        axis.grid(True, axis='y', color='#dfe6ee', linewidth=0.8, alpha=0.82)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    figure.savefig(temporary, format='png', dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    os.replace(temporary, output)

    generated_at = time.time()
    metadata = {
        'generated_at': generated_at,
        'url': f'/plots/{EVENT_ANALYSIS_FILENAME}',
        'filename': EVENT_ANALYSIS_FILENAME,
        'size_bytes': output.stat().st_size,
        'score_bin_width': int(score_bin_width),
        'episodes_30fps': int(scores30.numel()),
        'episodes_120fps': int(scores120.numel()),
        'created_l11_episodes_30fps': int(created30.sum().item()),
        'removed_l11_episodes_30fps': int(removed30.sum().item()),
        'high_level_creation_episode_rates_30fps': {
            f'L{level}': rate
            for level, rate in zip(event_levels, event_episode_rates)
        },
        'histogram': {
            'bin_starts': bin_starts.tolist(),
            'counts_30fps': counts30.tolist(),
            'counts_120fps': counts120.tolist(),
        },
    }
    _atomic_json(plot_dir / EVENT_ANALYSIS_METADATA_FILENAME, metadata)
    return metadata
