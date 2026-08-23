"""把轻量堵塞风险模型的测试集识别结果渲染为场景画廊。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import torch  # noqa: E402

from daxigua.rl.pair_risk import (  # noqa: E402
    PairRiskModel,
    PairRiskModelConfig,
    SPLIT_TEST,
    SPLIT_VALIDATION,
)


COLORS = (
    '#7d3c98', '#d35400', '#e67e22', '#f4d03f', '#73c66b', '#d94b45',
    '#f1948a', '#d4a017', '#8d6e63', '#239b56', '#117864',
)
CONFUSION_ORDER = ('TP', 'FP', 'FN', 'TN')
CONFUSION_TITLES = {
    'TP': 'Correct risk',
    'FP': 'False alarm',
    'FN': 'Missed risk',
    'TN': 'Correct safe',
}
CORRECT_KINDS = frozenset(('TP', 'TN'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='按等级和混淆类型渲染堵塞风险测试集场景。'
    )
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument(
        '--split', choices=('validation', 'test'), default='test'
    )
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--per-bucket', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--pool-multiplier', type=int, default=32)
    parser.add_argument(
        '--autocast-bfloat16',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    temporary.replace(path)


def _resolve_device(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA device is unavailable')
    return device


def confusion_kind(predicted, label):
    if bool(predicted):
        return 'TP' if bool(label) else 'FP'
    return 'FN' if bool(label) else 'TN'


def _priority(kind, probability):
    probability = float(probability)
    return probability if kind in ('TP', 'FP') else -probability


def _identity(candidate):
    if int(candidate['event_id']) >= 0:
        return ('event', int(candidate['event_id']))
    return (
        'negative', int(candidate['episode_id']),
        int(candidate['fruit_id_i']), int(candidate['fruit_id_j']),
    )


def select_unique_candidates(candidates, count):
    selected = []
    seen = set()
    for candidate in sorted(
            candidates, key=lambda item: item['priority'], reverse=True):
        identity = _identity(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(candidate)
        if len(selected) >= int(count):
            break
    return selected


def _candidate(columns, row, probability, kind):
    scalar_names = (
        'episode_id', 'step', 'pair_slot_i', 'pair_slot_j',
        'fruit_id_i', 'fruit_id_j', 'level', 'label', 'event_id',
        'lead_to_onset', 'danger_progress', 'over_danger_line',
    )
    result = {
        name: columns[name][row].item() for name in scalar_names
    }
    for name in (
            'positions', 'levels', 'fruit_ids', 'physics_radii', 'active',
            'fruit_queue'):
        result[name] = columns[name][row].detach().cpu().clone()
    result.update({
        'probability': float(probability),
        'kind': str(kind),
        'priority': _priority(kind, probability),
    })
    return result


def _model_columns(columns, indices, device):
    return {
        name: columns[name].index_select(0, indices).to(
            device, non_blocking=True
        )
        for name in PairRiskModel.REQUIRED_COLUMNS
    }


@torch.inference_mode()
def collect_candidates(args):
    if not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError('threshold must be between 0 and 1')
    if args.per_bucket <= 0 or args.batch_size <= 0:
        raise ValueError('per-bucket and batch-size must be positive')
    if args.pool_multiplier <= 0:
        raise ValueError('pool-multiplier must be positive')
    dataset_dir = Path(args.dataset_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    device = _resolve_device(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get('model_type') != 'pair_conditioned_deep_sets_risk_v1':
        raise ValueError('unsupported pair-risk checkpoint')
    model = PairRiskModel(
        PairRiskModelConfig(**checkpoint['model_config'])
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    split_value = (
        SPLIT_TEST if args.split == 'test' else SPLIT_VALIDATION
    )
    paths = tuple(sorted(
        (dataset_dir / 'labeled').glob('pair_risk_samples-*.pt')
    ))
    if not paths:
        raise FileNotFoundError('pair-risk dataset has no labeled shards')
    pool_limit = int(args.per_bucket) * int(args.pool_multiplier)
    pools = {
        (level, kind): []
        for level in range(7, 12)
        for kind in CONFUSION_ORDER
    }
    counts = {
        str(level): {kind: 0 for kind in CONFUSION_ORDER}
        for level in range(7, 12)
    }
    for path in paths:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if (
                payload.get('format_version') != 1
                or payload.get('table') != 'pair_risk_samples'):
            raise ValueError(f'invalid pair-risk shard: {path}')
        columns = payload['columns']
        split_rows = torch.nonzero(
            columns['split'].eq(split_value), as_tuple=False
        ).flatten()
        for rows in split_rows.split(int(args.batch_size)):
            autocast = (
                torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                if device.type == 'cuda' and args.autocast_bfloat16
                else nullcontext()
            )
            with autocast:
                logits = model(_model_columns(columns, rows, device))
            probabilities = torch.sigmoid(logits.float()).cpu()
            labels = columns['label'].index_select(0, rows).bool()
            levels = columns['level'].index_select(0, rows).long()
            predicted = probabilities.ge(float(args.threshold))
            for level in range(7, 12):
                level_mask = levels.eq(level)
                for kind in CONFUSION_ORDER:
                    if kind == 'TP':
                        mask = level_mask & predicted & labels
                    elif kind == 'FP':
                        mask = level_mask & predicted & ~labels
                    elif kind == 'FN':
                        mask = level_mask & ~predicted & labels
                    else:
                        mask = level_mask & ~predicted & ~labels
                    matches = torch.nonzero(mask, as_tuple=False).flatten()
                    counts[str(level)][kind] += int(matches.numel())
                    if matches.numel() == 0:
                        continue
                    priorities = (
                        probabilities[matches]
                        if kind in ('TP', 'FP')
                        else -probabilities[matches]
                    )
                    keep = min(pool_limit, int(matches.numel()))
                    chosen = matches[torch.topk(priorities, keep).indices]
                    bucket = pools[(level, kind)]
                    for local_row in chosen.tolist():
                        source_row = int(rows[local_row].item())
                        bucket.append(_candidate(
                            columns, source_row,
                            float(probabilities[local_row].item()), kind,
                        ))
                    bucket.sort(
                        key=lambda item: item['priority'], reverse=True
                    )
                    del bucket[pool_limit:]
    selected = {
        (level, kind): select_unique_candidates(
            pools[(level, kind)], args.per_bucket
        )
        for level in range(7, 12)
        for kind in CONFUSION_ORDER
    }
    return model.config, selected, counts


def _draw_scene(axis, candidate, board):
    width = float(board['board_width'])
    height = float(board['board_height'])
    wall = float(board['wall_width'])
    spawn_y = float(board.get('spawn_y', 252.0))
    correct = candidate['kind'] in CORRECT_KINDS
    accent = '#21a179' if correct else '#e63946'
    axis.set_facecolor('#f7f1e5')
    axis.add_patch(Rectangle(
        (wall, 0), width - 2.0 * wall, height - wall,
        fill=False, edgecolor='#4b3b2a', linewidth=2.2,
    ))
    axis.axhline(
        spawn_y, color='#c0392b', linewidth=1.0,
        linestyle='--', alpha=0.75,
    )
    pair_slots = {
        int(candidate['pair_slot_i']), int(candidate['pair_slot_j'])
    }
    pair_positions = []
    active_slots = torch.nonzero(
        candidate['active'], as_tuple=False
    ).flatten().tolist()
    for slot in active_slots:
        level = int(candidate['levels'][slot].item())
        x, y = (
            float(value) for value in candidate['positions'][slot].tolist()
        )
        radius = float(candidate['physics_radii'][slot].item())
        selected = slot in pair_slots
        fruit_id = int(candidate['fruit_ids'][slot].item())
        axis.add_patch(Circle(
            (x, y), radius,
            facecolor=COLORS[max(0, min(10, level - 1))],
            edgecolor=accent if selected else 'white',
            linewidth=4.0 if selected else 0.9,
            alpha=0.93,
            zorder=3 if selected else 2,
        ))
        axis.text(
            x, y, f'L{level}\n#{fruit_id}',
            ha='center', va='center', fontsize=7,
            color='white', weight='bold', zorder=4,
        )
        if selected:
            pair_positions.append((x, y))
    if len(pair_positions) == 2:
        axis.plot(
            [pair_positions[0][0], pair_positions[1][0]],
            [pair_positions[0][1], pair_positions[1][1]],
            color=accent, linewidth=2.0, linestyle=':', zorder=5,
        )
    lead = int(candidate['lead_to_onset'])
    lead_text = (
        f'onset in {lead} drops' if lead > 0
        else ('at onset' if lead == 0 else f'{-lead} drops after onset')
    )
    if not bool(candidate['label']):
        lead_text = 'no event in label horizon'
    axis.set_title(
        f"L{int(candidate['level'])} {candidate['kind']} · "
        f"p={float(candidate['probability']):.3f}\n"
        f"episode={int(candidate['episode_id'])} · "
        f"drop={int(candidate['step'])} · {lead_text}",
        fontsize=9, color=accent, weight='bold',
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])


def render_candidate(candidate, board, output_path):
    figure, axis = plt.subplots(figsize=(5.4, 9.2))
    _draw_scene(axis, candidate, board)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)


def render_overview(selected, board, output_path):
    figure, axes = plt.subplots(
        5, 4, figsize=(15.5, 28.0), squeeze=False
    )
    for row, level in enumerate(range(7, 12)):
        for column, kind in enumerate(CONFUSION_ORDER):
            axis = axes[row][column]
            candidates = selected[(level, kind)]
            if candidates:
                _draw_scene(axis, candidates[0], board)
            else:
                axis.axis('off')
                axis.text(
                    0.5, 0.5, f'L{level} {kind}\nno sample',
                    ha='center', va='center', transform=axis.transAxes,
                )
            if row == 0:
                axis.text(
                    0.5, 1.08, CONFUSION_TITLES[kind],
                    ha='center', va='bottom', transform=axis.transAxes,
                    fontsize=13, weight='bold',
                )
    figure.suptitle(
        'Pair-risk model validation gallery · extreme test examples',
        fontsize=17, weight='bold', y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.987))
    figure.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(figure)


def _report_candidate(candidate, filename):
    return {
        'file': filename,
        'kind': candidate['kind'],
        'level': int(candidate['level']),
        'probability': float(candidate['probability']),
        'label': bool(candidate['label']),
        'episode_id': int(candidate['episode_id']),
        'step': int(candidate['step']),
        'event_id': int(candidate['event_id']),
        'lead_to_onset': int(candidate['lead_to_onset']),
        'fruit_id_i': int(candidate['fruit_id_i']),
        'fruit_id_j': int(candidate['fruit_id_j']),
    }


def run(args):
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'gallery output is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_manifest = _load_json(
        Path(args.dataset_dir).resolve() / 'manifest.json'
    )
    model_config, selected, counts = collect_candidates(args)
    simulator = collection_manifest.get('simulator_config') or {}
    board = {
        'board_width': float(simulator.get(
            'board_width', model_config.board_width
        )),
        'board_height': float(simulator.get(
            'board_height', model_config.board_height
        )),
        'wall_width': float(simulator.get(
            'wall_width', model_config.wall_width
        )),
        'spawn_y': float(simulator.get('spawn_y', 252.0)),
    }
    report_rows = []
    for level in range(7, 12):
        for kind in CONFUSION_ORDER:
            for index, candidate in enumerate(selected[(level, kind)], 1):
                filename = f'L{level}_{kind}_{index:02d}.png'
                render_candidate(candidate, board, output_dir / filename)
                report_rows.append(_report_candidate(candidate, filename))
    overview = output_dir / 'gallery_overview.png'
    render_overview(selected, board, overview)
    report = {
        'format_version': 1,
        'purpose': 'pair_risk_prediction_scene_gallery',
        'dataset_dir': str(Path(args.dataset_dir).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'split': args.split,
        'threshold': float(args.threshold),
        'per_bucket': int(args.per_bucket),
        'confusion_counts': counts,
        'overview': overview.name,
        'images': report_rows,
    }
    _atomic_json(output_dir / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
