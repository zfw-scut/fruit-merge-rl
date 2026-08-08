"""生成辅助动作学习面板的确定性接口预览图（不是训练结果）。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile

from daxigua.rl.curves import render_training_curve_snapshot


def _metric_rows():
    rows = []
    for index in range(49):
        transition = index * 500_000
        progress = index / 48.0
        decay = math.exp(-3.1 * progress)
        rows.append({
            'transitions': transition,
            'training_rolling_mean_score': 1650 + 2550 * progress,
            'training_window_mean_score': 1500 + 2750 * progress,
            'training_window_max_score': 4300 + 8200 * progress,
            'loss': 0.44 * decay + 0.055,
            'dqn_loss': 0.34 * decay + 0.045,
            'mean_abs_td_error': 0.29 * decay + 0.035,
            'aux_loss_merge': 0.66 * decay + 0.09,
            'aux_loss_q0_lineage': 0.72 * decay + 0.10,
            'aux_loss_first_contact': 0.58 * decay + 0.075,
            'aux_loss_generation': 0.81 * decay + 0.12,
            'aux_loss_outcome': 0.49 * decay + 0.065,
            'env_steps_per_second': 57_000 + 2800 * math.sin(index / 5),
            'learner_samples_per_second': 41_000 + 1900 * math.cos(index / 6),
        })
    return rows


def _evaluation_rows():
    rows = []
    for index, transition in enumerate(range(2_000_000, 24_000_001, 2_000_000)):
        density = [0.0] * 12
        density[7] = 8.5 + index * 0.55
        density[8] = 3.9 + index * 0.47
        density[9] = 1.35 + index * 0.31
        density[10] = 0.31 + index * 0.14
        density[11] = 0.04 + index * 0.055
        rows.append({
            'transition': transition,
            'physics_fps': 30,
            'mean_score': 1900 + index * 245,
            'created_level_density_per_1000_drops': density,
        })
    return rows


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
        encoding='utf-8',
    )


def _render_dashboard_panels(path, metrics, evaluations):
    import matplotlib

    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, 2, figsize=(14, 4.8), constrained_layout=True
    )
    figure.patch.set_facecolor('#f3f6fa')
    figure.suptitle(
        'Cloud dashboard interface preview · synthetic data, not a result',
        fontsize=15,
        fontweight='bold',
    )
    x = [row['transitions'] / 1_000_000 for row in metrics]
    for key, label, color in (
            ('aux_loss_merge', 'Merge', '#0067c0'),
            ('aux_loss_q0_lineage', 'q0 lineage', '#107c10'),
            ('aux_loss_first_contact', 'First contact', '#ca5010'),
            ('aux_loss_generation', 'Generated fruit', '#8764b8'),
            ('aux_loss_outcome', 'Outcome', '#c239b3')):
        axes[0].plot(x, [row[key] for row in metrics], label=label, color=color)
    axes[0].set_title('Auxiliary action-effect losses', loc='left', fontweight='bold')
    axes[0].set_xlabel('Million transitions')
    axes[0].set_ylabel('Loss')
    axes[0].legend(frameon=False, fontsize=8)

    eval_x = [row['transition'] / 1_000_000 for row in evaluations]
    for level, color in zip(
            range(7, 12), ('#69797e', '#0099bc', '#107c10', '#ca5010', '#d13438')):
        axes[1].plot(
            eval_x,
            [row['created_level_density_per_1000_drops'][level]
             for row in evaluations],
            marker='o',
            markersize=3,
            label=f'L{level}',
            color=color,
        )
    axes[1].set_title(
        'High-level fruit creation density', loc='left', fontweight='bold'
    )
    axes[1].set_xlabel('Million transitions')
    axes[1].set_ylabel('Created fruits / 1k drops')
    axes[1].legend(frameon=False, fontsize=8, ncol=3)
    for axis in axes:
        axis.set_facecolor('#ffffff')
        axis.grid(True, color='#dfe6ee', linewidth=0.8)
        axis.spines[['top', 'right']].set_visible(False)
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / 'docs' / 'assets' / 'auxiliary_action_learning'
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metric_rows()
    evaluations = _evaluation_rows()
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / 'preview_run'
        _write_jsonl(run_dir / 'metrics.jsonl', metrics)
        _write_jsonl(
            run_dir / 'evaluations' / 'metrics.jsonl', evaluations
        )
        render_training_curve_snapshot(run_dir)
        shutil.copy2(
            run_dir / 'plots' / 'training_curves.png',
            output_dir / 'training_curves_preview.png',
        )
    _render_dashboard_panels(
        output_dir / 'dashboard_panels_preview.png', metrics, evaluations
    )
    print(output_dir)


if __name__ == '__main__':
    main()
