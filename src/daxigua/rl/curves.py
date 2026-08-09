"""从持久指标生成可归档的训练曲线快照。"""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import threading
import time


CURVE_FILENAME = 'training_curves.png'
CURVE_METADATA_FILENAME = 'training_curves.json'
_FONT_CONFIGURED = False
_CHINESE_FONT_AVAILABLE = False


def _read_jsonl(path):
    rows = []
    try:
        with Path(path).open('r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return []
    return rows


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _series(rows, x_name, y_name, *, x_scale=1.0):
    xs = []
    ys = []
    for row in rows:
        x = _finite(row.get(x_name))
        y = _finite(row.get(y_name))
        if x is None or y is None:
            continue
        xs.append(x / x_scale)
        ys.append(y)
    return xs, ys


def _downsample(rows, limit=1800):
    if len(rows) <= limit:
        return rows
    indices = {
        round(index * (len(rows) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [rows[index] for index in sorted(indices)]


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(
        f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp'
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _configure_font(matplotlib):
    global _FONT_CONFIGURED, _CHINESE_FONT_AVAILABLE
    if _FONT_CONFIGURED:
        return _CHINESE_FONT_AVAILABLE

    from matplotlib import font_manager

    preferred = (
        'Microsoft YaHei',
        'Microsoft YaHei UI',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'WenQuanYi Micro Hei',
        'SimHei',
    )
    available = set()
    for path in font_manager.findSystemFonts():
        try:
            available.add(font_manager.FontProperties(fname=path).get_name())
        except (OSError, RuntimeError, ValueError):
            continue
    selected = next((name for name in preferred if name in available), None)
    if selected is not None:
        matplotlib.rcParams['font.sans-serif'] = [selected, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    _CHINESE_FONT_AVAILABLE = selected is not None
    _FONT_CONFIGURED = True
    return _CHINESE_FONT_AVAILABLE


def _style_axis(axis, has_chinese_font):
    axis.set_facecolor('#ffffff')
    axis.grid(True, color='#dfe6ee', linewidth=0.8, alpha=0.82)
    axis.tick_params(colors='#5f6b7a', labelsize=9)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_color('#c6d0dc')
    axis.spines['bottom'].set_color('#c6d0dc')
    axis.set_xlabel(
        '累计投放 / Million transitions'
        if has_chinese_font else 'Million transitions',
        color='#5f6b7a',
    )


def _plot_or_note(axis, xs, ys, *, label, color, **kwargs):
    if xs:
        axis.plot(xs, ys, label=label, color=color, **kwargs)
        return True
    axis.text(
        0.5, 0.5, 'Waiting for data',
        transform=axis.transAxes,
        ha='center', va='center', color='#7a8696',
    )
    return False


def render_training_curve_snapshot(run_dir):
    """读取 JSONL，原子更新标准 PNG，并返回可供面板展示的元数据。"""

    run_dir = Path(run_dir)
    metric_rows = _read_jsonl(run_dir / 'metrics.jsonl')
    metrics = _downsample(metric_rows)
    evaluations = _read_jsonl(run_dir / 'evaluations' / 'metrics.jsonl')
    if not metrics:
        raise RuntimeError('metrics.jsonl 尚无可绘制训练指标')

    plot_dir = run_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        'MPLCONFIGDIR', str((plot_dir / '.mplconfig').resolve())
    )

    import matplotlib

    matplotlib.use('Agg', force=True)
    has_chinese_font = _configure_font(matplotlib)
    import matplotlib.pyplot as plt

    labels = {
        'title': (
            '合成大西瓜 GNN-DQN 训练曲线快照'
            if has_chinese_font else
            'Fruit Merge GNN-DQN Training Curves'
        ),
        'average': (
            '平均得分 / Average score'
            if has_chinese_font else 'Average score'
        ),
        'maximum': (
            '训练窗口最高分 / Window maximum'
            if has_chinese_font else 'Training window maximum'
        ),
        'optimization': (
            '优化信号 / Optimization signals'
            if has_chinese_font else 'Optimization signals'
        ),
        'throughput': (
            '训练吞吐 / Throughput'
            if has_chinese_font else 'Training throughput'
        ),
        'auxiliary': (
            '辅助动作效果损失 / Auxiliary action-effect losses'
            if has_chinese_font else 'Auxiliary action-effect losses'
        ),
        'merge_density': (
            '高等级新水果生成密度 / High-level fruit creation density'
            if has_chinese_font else 'High-level fruit creation density'
        ),
        'bonus_shadow': (
            'Bonus影子动作改变率 / Shadow action change rate'
            if has_chinese_font else 'Bonus shadow action change rate'
        ),
        'decision_scale': (
            '决策量纲 / Decision scale'
            if has_chinese_font else 'Decision scale'
        ),
    }
    colors = {
        'blue': '#0067c0',
        'cyan': '#0099bc',
        'green': '#107c10',
        'red': '#d13438',
        'violet': '#8764b8',
        'orange': '#ca5010',
        'pink': '#c239b3',
        'slate': '#69797e',
    }

    def bilingual(chinese, english):
        return (
            f'{chinese} / {english}'
            if has_chinese_font else english
        )

    figure = None
    output_path = plot_dir / CURVE_FILENAME
    temporary = output_path.with_name(
        f'.{output_path.name}.{os.getpid()}.{threading.get_ident()}.tmp'
    )
    try:
        figure, axes = plt.subplots(
            4, 2, figsize=(14, 16.0), constrained_layout=True
        )
        figure.patch.set_facecolor('#f3f6fa')
        figure.suptitle(labels['title'], fontsize=18, fontweight='bold')

        average = axes[0, 0]
        _style_axis(average, has_chinese_font)
        average.set_title(labels['average'], loc='left', fontweight='bold')
        average.set_ylabel('分数 / Score' if has_chinese_font else 'Score')
        for key, series_label, color in (
                ('training_rolling_mean_score', bilingual('滚动局均分', 'Rolling'), colors['green']),
                ('training_window_mean_score', bilingual('窗口局均分', 'Window'), colors['blue'])):
            xs, ys = _series(metrics, 'transitions', key, x_scale=1_000_000)
            _plot_or_note(
                average, xs, ys,
                label=series_label, color=color, linewidth=2.0,
            )
        for fps, series_label, color, marker in (
                (30, bilingual('30 FPS 评估', '30 FPS Eval'), colors['cyan'], 'o'),
                (120, bilingual('120 FPS 评估', '120 FPS Eval'), colors['violet'], 'D')):
            rows = [
                row for row in evaluations
                if int(_finite(row.get('physics_fps')) or -1) == fps
            ]
            xs, ys = _series(
                rows, 'transition', 'mean_score', x_scale=1_000_000
            )
            if xs:
                average.plot(
                    xs, ys, linestyle='--', marker=marker, markersize=5,
                    label=series_label, color=color, linewidth=1.6,
                )
        average.legend(frameon=False, fontsize=8, loc='best')

        maximum = axes[0, 1]
        _style_axis(maximum, has_chinese_font)
        maximum.set_title(labels['maximum'], loc='left', fontweight='bold')
        maximum.set_ylabel('分数 / Score' if has_chinese_font else 'Score')
        xs, ys = _series(
            metrics, 'transitions', 'training_window_max_score',
            x_scale=1_000_000,
        )
        _plot_or_note(
            maximum, xs, ys,
            label=bilingual('窗口最高分', 'Window max'), color=colors['red'],
            linewidth=1.8,
        )
        if ys:
            peak = max(range(len(ys)), key=ys.__getitem__)
            maximum.scatter(
                [xs[peak]], [ys[peak]], s=36, color=colors['red'], zorder=4
            )
            maximum.annotate(
                f'{ys[peak]:,.0f}', (xs[peak], ys[peak]),
                xytext=(7, 7), textcoords='offset points',
                fontsize=9, color=colors['red'], fontweight='bold',
            )
            maximum.legend(frameon=False, fontsize=8, loc='best')

        optimization = axes[1, 0]
        _style_axis(optimization, has_chinese_font)
        optimization.set_title(
            labels['optimization'], loc='left', fontweight='bold'
        )
        loss_x, loss_y = _series(
            metrics, 'transitions', 'loss', x_scale=1_000_000
        )
        _plot_or_note(
            optimization, loss_x, loss_y,
            label='Total Loss', color=colors['blue'], linewidth=1.8,
        )
        dqn_x, dqn_y = _series(
            metrics, 'transitions', 'dqn_loss', x_scale=1_000_000
        )
        if dqn_x:
            optimization.plot(
                dqn_x, dqn_y, label='DQN Huber Loss',
                color=colors['green'], linewidth=1.5,
            )
        optimization.set_ylabel('Loss', color=colors['blue'])
        td_axis = optimization.twinx()
        td_axis.spines['top'].set_visible(False)
        td_axis.spines['right'].set_color('#c6d0dc')
        td_axis.tick_params(colors=colors['red'], labelsize=9)
        td_x, td_y = _series(
            metrics, 'transitions', 'mean_abs_td_error', x_scale=1_000_000
        )
        if td_x:
            td_axis.plot(
                td_x, td_y, label='Mean |TD error|',
                color=colors['red'], linewidth=1.6,
            )
        td_axis.set_ylabel('Mean |TD error|', color=colors['red'])
        handles, legend_labels = optimization.get_legend_handles_labels()
        td_handles, td_labels = td_axis.get_legend_handles_labels()
        optimization.legend(
            handles + td_handles, legend_labels + td_labels,
            frameon=False, fontsize=8, loc='best',
        )

        throughput = axes[1, 1]
        _style_axis(throughput, has_chinese_font)
        throughput.set_title(
            labels['throughput'], loc='left', fontweight='bold'
        )
        throughput.set_ylabel(
            '次数 / 秒' if has_chinese_font else 'Items / second'
        )
        for key, series_label, color in (
                ('env_steps_per_second', bilingual('投放', 'Env'), colors['blue']),
                ('learner_samples_per_second', bilingual('学习样本', 'Learner'), colors['green'])):
            xs, ys = _series(metrics, 'transitions', key, x_scale=1_000_000)
            _plot_or_note(
                throughput, xs, ys, label=series_label,
                color=color, linewidth=1.8,
            )
        throughput.legend(frameon=False, fontsize=8, loc='best')

        auxiliary = axes[2, 0]
        _style_axis(auxiliary, has_chinese_font)
        auxiliary.set_title(
            labels['auxiliary'], loc='left', fontweight='bold'
        )
        auxiliary.set_ylabel('Loss')
        for key, series_label, color in (
                ('aux_loss_merge', bilingual('合成', 'Merge'), colors['blue']),
                ('aux_loss_q0_lineage', bilingual('q0 谱系', 'q0 lineage'), colors['green']),
                ('aux_loss_first_contact', bilingual('首次接触', 'First contact'), colors['orange']),
                ('aux_loss_generation', bilingual('新水果', 'Generated fruit'), colors['violet']),
                ('aux_loss_outcome', bilingual('最终结果', 'Outcome'), colors['pink'])):
            xs, ys = _series(metrics, 'transitions', key, x_scale=1_000_000)
            if xs:
                auxiliary.plot(
                    xs, ys, label=series_label, color=color, linewidth=1.6
                )
        if not auxiliary.lines:
            auxiliary.text(
                0.5, 0.5, 'Waiting for auxiliary losses',
                transform=auxiliary.transAxes, ha='center', va='center',
                color='#7a8696',
            )
        if auxiliary.lines:
            auxiliary.legend(frameon=False, fontsize=8, loc='best')

        merge_density = axes[2, 1]
        _style_axis(merge_density, has_chinese_font)
        merge_density.set_title(
            labels['merge_density'], loc='left', fontweight='bold'
        )
        merge_density.set_ylabel(
            bilingual('每千次投放生成数', 'Created / 1k drops')
        )
        density_colors = (
            colors['slate'], colors['cyan'], colors['green'],
            colors['orange'], colors['red'],
        )
        for level, color in zip(range(7, 12), density_colors, strict=True):
            xs = []
            ys = []
            for row in evaluations:
                transition = _finite(row.get('transition'))
                density = row.get('created_level_density_per_1000_drops')
                if transition is None or not isinstance(density, list):
                    continue
                if level >= len(density):
                    continue
                value = _finite(density[level])
                if value is None:
                    continue
                xs.append(transition / 1_000_000)
                ys.append(value)
            if xs:
                merge_density.plot(
                    xs, ys, marker='o', markersize=3,
                    label=f'L{level}', color=color, linewidth=1.5,
                )
        if not merge_density.lines:
            merge_density.text(
                0.5, 0.5, 'Waiting for evaluation events',
                transform=merge_density.transAxes, ha='center', va='center',
                color='#7a8696',
            )
        if merge_density.lines:
            merge_density.legend(
                frameon=False, fontsize=8, loc='best', ncol=3
            )

        bonus_shadow = axes[3, 0]
        _style_axis(bonus_shadow, has_chinese_font)
        bonus_shadow.set_title(
            labels['bonus_shadow'], loc='left', fontweight='bold'
        )
        bonus_shadow.set_ylabel(
            bilingual('改变率', 'Changed action rate')
        )
        shadow_colors = (
            colors['slate'], colors['blue'], colors['green'],
            colors['orange'], colors['red'],
        )
        for suffix, label, color in zip(
                ('0p5', '1', '2', '4', '8'),
                ('0.5', '1', '2', '4', '8'),
                shadow_colors,
                strict=True):
            xs, ys = _series(
                metrics,
                'transitions',
                f'shadow_bonus_{suffix}_changed_action_rate',
                x_scale=1_000_000,
            )
            if xs:
                bonus_shadow.plot(
                    xs, ys, label=f'β={label}', color=color, linewidth=1.5
                )
        bonus_shadow.set_ylim(-0.02, 1.02)
        if not bonus_shadow.lines:
            bonus_shadow.text(
                0.5, 0.5, 'Waiting for shadow bonus metrics',
                transform=bonus_shadow.transAxes, ha='center', va='center',
                color='#7a8696',
            )
        else:
            bonus_shadow.legend(frameon=False, fontsize=8, loc='best')

        decision_scale = axes[3, 1]
        _style_axis(decision_scale, has_chinese_font)
        decision_scale.set_title(
            labels['decision_scale'], loc='left', fontweight='bold'
        )
        decision_scale.set_ylabel('Q / Uncertainty')
        for key, series_label, color in (
                ('actor_q_action_range', bilingual('Q动作范围', 'Q range'), colors['blue']),
                ('actor_q_top_margin', bilingual('Q前两名间隔', 'Q top margin'), colors['green']),
                ('actor_policy_disagreement', bilingual('平均不确定性', 'Mean uncertainty'), colors['violet']),
                ('actor_uncertainty_max', bilingual('最大不确定性', 'Max uncertainty'), colors['red'])):
            xs, ys = _series(metrics, 'transitions', key, x_scale=1_000_000)
            if xs:
                decision_scale.plot(
                    xs, ys, label=series_label, color=color, linewidth=1.5
                )
        if not decision_scale.lines:
            decision_scale.text(
                0.5, 0.5, 'Waiting for decision scale metrics',
                transform=decision_scale.transAxes,
                ha='center', va='center', color='#7a8696',
            )
        else:
            decision_scale.set_yscale('log')
            decision_scale.legend(frameon=False, fontsize=8, loc='best')

        generated_at = time.time()
        latest_transition = max(
            (_finite(row.get('transitions')) or 0.0 for row in metrics),
            default=0.0,
        )
        footer = (
            f'{"生成时间 / " if has_chinese_font else ""}Generated: '
            f'{datetime.fromtimestamp(generated_at).astimezone():%Y-%m-%d %H:%M:%S}  '
            f'|  Transition: {latest_transition:,.0f}'
        )
        figure.text(0.5, 0.005, footer, ha='center', color='#64748b', fontsize=8)
        figure.savefig(
            temporary, format='png', dpi=140,
            facecolor=figure.get_facecolor(), metadata={
                'Title': labels['title'],
                'Software': 'fruit-merge-rl-accelerated-v1',
            },
        )
        os.replace(temporary, output_path)
        stat = output_path.stat()
        metadata = {
            'name': CURVE_FILENAME,
            'url': f'/plots/{CURVE_FILENAME}',
            'generated_at': generated_at,
            'modified_at': stat.st_mtime,
            'size_bytes': stat.st_size,
            'source_metric_rows': len(metric_rows),
            'plotted_metric_rows': len(metrics),
            'source_evaluation_rows': len(evaluations),
            'source_last_transition': int(latest_transition),
            'chinese_font_available': has_chinese_font,
        }
        _atomic_write_json(plot_dir / CURVE_METADATA_FILENAME, metadata)
        return metadata
    finally:
        if figure is not None:
            plt.close(figure)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def existing_curve_metadata(run_dir):
    """读取已生成快照，供面板重启后立即恢复展示。"""

    run_dir = Path(run_dir)
    metadata_path = run_dir / 'plots' / CURVE_METADATA_FILENAME
    rows = []
    try:
        value = json.loads(metadata_path.read_text(encoding='utf-8'))
        if isinstance(value, dict):
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    output_path = run_dir / 'plots' / CURVE_FILENAME
    try:
        stat = output_path.stat()
    except OSError:
        return rows[0] if rows else None
    metadata = rows[0] if rows else {}
    return {
        **metadata,
        'name': CURVE_FILENAME,
        'url': f'/plots/{CURVE_FILENAME}',
        'modified_at': stat.st_mtime,
        'size_bytes': stat.st_size,
    }
