"""训练和评估轻量单帧水果对堵塞风险模型。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from daxigua.rl.pair_risk import (  # noqa: E402
    PairRiskModel,
    PairRiskModelConfig,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    checkpoint_payload,
    risk_metrics,
)
from daxigua.rl.task_telemetry import (  # noqa: E402
    TaskTelemetryPublisher,
    make_task_id,
)


SPLIT_NAMES = {
    SPLIT_TRAIN: 'train',
    SPLIT_VALIDATION: 'validation',
    SPLIT_TEST: 'test',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='训练或评估Pair-conditioned Deep Sets堵塞风险模型。'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    train = subparsers.add_parser('train')
    train.add_argument('dataset_dir', type=Path)
    train.add_argument('--output-dir', type=Path, required=True)
    train.add_argument('--device', default='auto')
    train.add_argument('--epochs', type=int, default=20)
    train.add_argument('--batch-size', type=int, default=1024)
    train.add_argument('--learning-rate', type=float, default=1e-3)
    train.add_argument('--weight-decay', type=float, default=1e-4)
    train.add_argument('--grad-clip-norm', type=float, default=5.0)
    train.add_argument('--seed', type=int, default=20260824)
    train.add_argument('--level-embedding-dim', type=int, default=8)
    train.add_argument('--context-hidden-dim', type=int, default=48)
    train.add_argument('--head-hidden-dim', type=int, default=64)
    train.add_argument(
        '--autocast-bfloat16',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument('--early-stop-patience', type=int, default=5)
    train.add_argument(
        '--balanced-training',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            '按等级×标签平衡总loss贡献，并让同一确认事件的连续'
            '正样本合计只贡献一次事件权重。'
        ),
    )

    evaluate = subparsers.add_parser('evaluate')
    evaluate.add_argument('dataset_dir', type=Path)
    evaluate.add_argument('checkpoint', type=Path)
    evaluate.add_argument(
        '--split', choices=('validation', 'test'), default='test'
    )
    evaluate.add_argument('--device', default='auto')
    evaluate.add_argument('--batch-size', type=int, default=2048)
    evaluate.add_argument(
        '--autocast-bfloat16',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _resolve_device(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA device is unavailable')
    return device


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    os.replace(temporary, path)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _dataset_paths(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / 'labeled' / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'labeled manifest not found: {manifest_path}')
    paths = tuple(sorted(
        (dataset_dir / 'labeled').glob('pair_risk_samples-*.pt')
    ))
    if not paths:
        raise FileNotFoundError('pair-risk dataset has no labeled shards')
    return _load_json(manifest_path), paths


def _load_shard(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (
            payload.get('format_version') != 1
            or payload.get('table') != 'pair_risk_samples'):
        raise ValueError(f'invalid pair-risk shard: {path}')
    return payload['columns']


def _batch_columns(columns, indices, device):
    names = PairRiskModel.REQUIRED_COLUMNS + (
        'label', 'level', 'event_id', 'lead_to_onset'
    )
    return {
        name: columns[name].index_select(0, indices).to(
            device, non_blocking=True
        )
        for name in names
    }


def _autocast(device, enabled):
    if not enabled or device.type != 'cuda':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def _shuffle_paths(paths, generator):
    order = torch.randperm(len(paths), generator=generator).tolist()
    return tuple(paths[index] for index in order)


def _iter_split_batches(
        paths,
        split,
        batch_size,
        device,
        *,
        shuffle,
        generator):
    selected_paths = (
        _shuffle_paths(paths, generator) if shuffle else tuple(paths)
    )
    for path in selected_paths:
        columns = _load_shard(path)
        indices = torch.nonzero(
            columns['split'].eq(int(split)), as_tuple=False
        ).flatten()
        if indices.numel() == 0:
            continue
        if shuffle:
            indices = indices[torch.randperm(
                indices.numel(), generator=generator
            )]
        for start in range(0, indices.numel(), int(batch_size)):
            yield _batch_columns(
                columns,
                indices[start:start + int(batch_size)],
                device,
            )


def build_training_balance(paths):
    """统计训练集的等级/标签单元和正例事件重复数。"""

    cell_samples = torch.zeros(12, 2, dtype=torch.int64)
    positive_event_parts = []
    positive_level_parts = []
    for path in paths:
        columns = _load_shard(path)
        selected = columns['split'].eq(int(SPLIT_TRAIN))
        if not selected.any():
            continue
        labels = columns['label'][selected].long()
        levels = columns['level'][selected].long()
        flat = levels * 2 + labels
        cell_samples.view(-1).add_(torch.bincount(
            flat, minlength=cell_samples.numel()
        ))
        positive = labels.bool()
        if positive.any():
            positive_event_parts.append(
                columns['event_id'][selected][positive].long()
            )
            positive_level_parts.append(levels[positive])

    if not positive_event_parts:
        raise ValueError('balanced training requires positive events')
    positive_events = torch.cat(positive_event_parts)
    positive_levels = torch.cat(positive_level_parts)
    if int(positive_events.min().item()) < 0:
        raise ValueError('positive pair-risk samples require event ids')
    event_counts = torch.bincount(positive_events)
    event_levels = torch.full(
        (event_counts.numel(),), -1, dtype=torch.int64
    )
    event_levels[positive_events] = positive_levels
    unique_events = torch.zeros(12, dtype=torch.int64)
    present_events = torch.nonzero(event_counts > 0, as_tuple=False).flatten()
    unique_events.add_(torch.bincount(
        event_levels[present_events], minlength=12
    ))

    cell_mass = cell_samples.double()
    cell_mass[:, 1] = unique_events.double()
    target_cells = torch.zeros_like(cell_mass, dtype=torch.bool)
    target_cells[7:12] = cell_mass[7:12] > 0
    cell_count = int(target_cells.sum().item())
    if cell_count <= 1:
        raise ValueError('balanced training requires multiple label cells')
    total_samples = int(cell_samples[7:12].sum().item())
    target_mass = total_samples / cell_count
    group_scale = torch.zeros_like(cell_mass, dtype=torch.float32)
    group_scale[target_cells] = (
        target_mass / cell_mass[target_cells]
    ).float()
    summary = {
        'strategy': 'level_label_and_positive_event_balanced',
        'training_samples': total_samples,
        'balanced_cells': cell_count,
        'target_weight_mass_per_cell': target_mass,
        'by_level': {
            str(level): {
                'negative_samples': int(cell_samples[level, 0].item()),
                'positive_samples': int(cell_samples[level, 1].item()),
                'positive_events': int(unique_events[level].item()),
            }
            for level in range(7, 12)
        },
    }
    return {
        'event_counts': event_counts,
        'group_scale': group_scale,
        'summary': summary,
    }


def training_balance_to(balance, device):
    if balance is None:
        return None
    return {
        'event_counts': balance['event_counts'].to(device),
        'group_scale': balance['group_scale'].to(device),
        'summary': balance['summary'],
    }


def balanced_sample_weights(batch, balance):
    """返回均值约为1的逐样本权重，不复制或丢弃训练行。"""

    labels = batch['label'].bool()
    levels = batch['level'].long()
    weights = balance['group_scale'][levels, labels.long()]
    positive = labels & batch['event_id'].ge(0)
    if positive.any():
        event_ids = batch['event_id'][positive].long()
        if int(event_ids.max().item()) >= balance['event_counts'].numel():
            raise ValueError('pair-risk event id exceeds balance table')
        repeats = balance['event_counts'][event_ids].clamp_min(1)
        weights = weights.clone()
        weights[positive] /= repeats.to(weights.dtype)
    return weights


def _event_warning_metrics(probabilities, event_ids, leads, threshold=0.5):
    detections = {}
    all_events = set()
    for probability, event_id, lead in zip(
            probabilities.tolist(), event_ids.tolist(), leads.tolist()):
        if int(event_id) < 0:
            continue
        event_id = int(event_id)
        all_events.add(event_id)
        if probability >= threshold:
            detections[event_id] = max(
                int(lead), detections.get(event_id, -32768)
            )
    positive_leads = [value for value in detections.values() if value > 0]
    positive_leads.sort()
    median = (
        positive_leads[len(positive_leads) // 2]
        if positive_leads else None
    )
    return {
        'events': len(all_events),
        'detected_events': len(detections),
        'event_recall': len(detections) / max(1, len(all_events)),
        'events_warned_before_onset': len(positive_leads),
        'median_early_warning_drops': median,
    }


@torch.inference_mode()
def evaluate_model(
        model,
        paths,
        split,
        batch_size,
        device,
        *,
        autocast_bfloat16,
        pos_weight=None):
    model.eval()
    logits_parts = []
    label_parts = []
    level_parts = []
    event_parts = []
    lead_parts = []
    loss_sum = 0.0
    sample_count = 0
    generator = torch.Generator().manual_seed(0)
    started = time.perf_counter()
    for batch in _iter_split_batches(
            paths,
            split,
            batch_size,
            device,
            shuffle=False,
            generator=generator):
        with _autocast(device, autocast_bfloat16):
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch['label'].float(),
                pos_weight=pos_weight,
                reduction='sum',
            )
        count = int(logits.numel())
        loss_sum += float(loss.item())
        sample_count += count
        logits_parts.append(logits.float().cpu())
        label_parts.append(batch['label'].bool().cpu())
        level_parts.append(batch['level'].long().cpu())
        event_parts.append(batch['event_id'].long().cpu())
        lead_parts.append(batch['lead_to_onset'].long().cpu())
    if sample_count == 0:
        return {'split': SPLIT_NAMES[int(split)], 'samples': 0}
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts)
    levels = torch.cat(level_parts)
    event_ids = torch.cat(event_parts)
    leads = torch.cat(lead_parts)
    probabilities = torch.sigmoid(logits)
    result = {
        'split': SPLIT_NAMES[int(split)],
        'loss': loss_sum / sample_count,
        **risk_metrics(logits, labels),
        'warning': _event_warning_metrics(
            probabilities, event_ids, leads
        ),
        'by_level': {},
        'elapsed_seconds': time.perf_counter() - started,
    }
    level_average_precisions = []
    for level in range(7, 12):
        mask = levels.eq(level)
        level_result = risk_metrics(
            logits[mask], labels[mask]
        )
        result['by_level'][str(level)] = level_result
        value = level_result.get('average_precision')
        if value is not None and math.isfinite(float(value)):
            level_average_precisions.append(float(value))
    result['macro_average_precision'] = (
        sum(level_average_precisions) / len(level_average_precisions)
        if level_average_precisions else math.nan
    )
    return result


def _model_config_from_dataset(args, dataset_dir):
    collection = _load_json(Path(dataset_dir) / 'manifest.json')
    simulator = collection.get('simulator_config') or {}
    return PairRiskModelConfig(
        max_fruits=int(simulator.get('max_fruits', 64)),
        queue_length=int(simulator.get('queue_length', 4)),
        level_embedding_dim=args.level_embedding_dim,
        context_hidden_dim=args.context_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        board_width=float(simulator.get('board_width', 560.0)),
        board_height=float(simulator.get('board_height', 1120.0)),
        wall_width=float(simulator.get('wall_width', 20.0)),
        physics_fps=float(simulator.get('physics_fps', 30.0)),
    )


def _training_counts(manifest):
    train = manifest['counts']['train']
    positives = sum(int(train[str(level)]['positive']) for level in range(7, 12))
    negatives = sum(int(train[str(level)]['negative']) for level in range(7, 12))
    if positives <= 0 or negatives <= 0:
        raise ValueError('training split requires positive and negative samples')
    return positives, negatives


def train(args):
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError('epochs and batch-size must be positive')
    if args.learning_rate <= 0.0 or args.grad_clip_norm <= 0.0:
        raise ValueError('optimizer parameters must be positive')
    dataset_manifest, paths = _dataset_paths(args.dataset_dir)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model = PairRiskModel(
        _model_config_from_dataset(args, args.dataset_dir)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == 'cuda',
    )
    balanced_training = bool(getattr(args, 'balanced_training', True))
    balance = (
        training_balance_to(build_training_balance(paths), device)
        if balanced_training else None
    )
    generator = torch.Generator().manual_seed(args.seed + 1)
    history = []
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    training_meta = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'device': str(device),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'grad_clip_norm': args.grad_clip_norm,
        'seed': args.seed,
        'balanced_training': balanced_training,
        'training_balance': (
            balance['summary'] if balance is not None else None
        ),
        'autocast_bfloat16': bool(args.autocast_bfloat16),
    }
    telemetry = TaskTelemetryPublisher(
        task_id=make_task_id('pair-risk-training', output_dir),
        task_type='supervised_training',
        name='堵塞风险预测器训练',
        output_dir=output_dir,
        identity={
            'model': 'Pair-conditioned Deep Sets',
            'device': str(device),
            'dataset': Path(args.dataset_dir).name,
        },
        metric_schema=[
            {'key': 'train_loss', 'label': '训练损失'},
            {'key': 'validation_map', 'label': '验证 macro AP', 'format': 'percent'},
            {'key': 'best_validation_map', 'label': '最佳 macro AP', 'format': 'percent'},
        ],
        series_schema=[
            {'key': 'train_loss', 'label': '训练损失'},
            {'key': 'validation_map', 'label': '验证 macro AP'},
        ],
    )
    telemetry.update(
        phase='训练', current=0, total=int(args.epochs), unit='epoch'
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        loss_weight_sum = 0.0
        samples = 0
        epoch_started = time.perf_counter()
        for batch in _iter_split_batches(
                paths,
                SPLIT_TRAIN,
                args.batch_size,
                device,
                shuffle=True,
                generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, args.autocast_bfloat16):
                logits = model(batch)
                losses = F.binary_cross_entropy_with_logits(
                    logits,
                    batch['label'].float(),
                    reduction='none',
                )
                sample_weights = (
                    balanced_sample_weights(batch, balance)
                    if balance is not None else torch.ones_like(losses)
                )
                weight_sum = sample_weights.sum().clamp_min(1e-6)
                # 全数据权重均值为1；固定以配置batch归一化，避免稀有单元
                # 出现的batch被再次按自身权重和归一化而抵消全局平衡。
                loss = (
                    (losses * sample_weights).sum()
                    / float(args.batch_size)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip_norm
            )
            optimizer.step()
            count = int(logits.numel())
            loss_sum += float((losses * sample_weights).sum().item())
            loss_weight_sum += float(weight_sum.item())
            samples += count
        validation = evaluate_model(
            model,
            paths,
            SPLIT_VALIDATION,
            args.batch_size * 2,
            device,
            autocast_bfloat16=args.autocast_bfloat16,
            pos_weight=None,
        )
        row = {
            'epoch': epoch,
            'train_loss': loss_sum / max(1e-6, loss_weight_sum),
            'train_samples': samples,
            'train_weight_mass': loss_weight_sum,
            'epoch_seconds': time.perf_counter() - epoch_started,
            'validation': validation,
        }
        history.append(row)
        score = float(validation.get(
            'macro_average_precision', math.nan
        ))
        if not math.isfinite(score):
            score = -float(validation.get('loss', math.inf))
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(
                output_dir / 'best.pt',
                checkpoint_payload(
                    model,
                    training={**training_meta, 'best_epoch': best_epoch},
                    dataset_manifest=dataset_manifest,
                    history=history,
                ),
            )
        else:
            stale_epochs += 1
        progress = {
            'status': 'training',
            'epoch': epoch,
            'target_epochs': args.epochs,
            'best_epoch': best_epoch,
            'best_validation_macro_average_precision': best_score,
            'validation_average_precision': validation.get(
                'average_precision'
            ),
            'elapsed_seconds': time.perf_counter() - started,
            'latest': row,
        }
        _atomic_json(output_dir / 'training_progress.json', progress)
        telemetry.update(
            phase='训练',
            current=epoch,
            total=int(args.epochs),
            unit='epoch',
            metrics={
                'train_loss': row['train_loss'],
                'validation_map': validation.get('macro_average_precision'),
                'best_validation_map': best_score,
                'epoch_seconds': row['epoch_seconds'],
            },
            history_step=epoch,
        )
        print(json.dumps(_json_safe(progress), ensure_ascii=False), flush=True)
        if stale_epochs >= int(args.early_stop_patience):
            break

    checkpoint = torch.load(
        output_dir / 'best.pt', map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    test_result = evaluate_model(
        model,
        paths,
        SPLIT_TEST,
        args.batch_size * 2,
        device,
        autocast_bfloat16=args.autocast_bfloat16,
        pos_weight=None,
    )
    final = {
        'status': 'complete',
        'best_epoch': best_epoch,
        'best_validation_macro_average_precision': best_score,
        'test': test_result,
        'elapsed_seconds': time.perf_counter() - started,
        'checkpoint': str((output_dir / 'best.pt').resolve()),
    }
    _atomic_json(output_dir / 'training_progress.json', final)
    _atomic_json(output_dir / 'evaluation.json', final)
    telemetry.complete(
        phase='完成',
        current=history[-1]['epoch'] if history else 0,
        total=int(args.epochs),
        unit='epoch',
        metrics={
            'best_validation_map': best_score,
            'test_average_precision': test_result.get('average_precision'),
            'test_macro_average_precision': test_result.get('macro_average_precision'),
        },
        history_step=best_epoch,
        record_history=False,
    )
    print(json.dumps(_json_safe(final), ensure_ascii=False, indent=2))
    return final


def evaluate_checkpoint(args):
    manifest, paths = _dataset_paths(args.dataset_dir)
    device = _resolve_device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    if checkpoint.get('model_type') != 'pair_conditioned_deep_sets_risk_v1':
        raise ValueError('unsupported pair-risk checkpoint')
    model = PairRiskModel(
        PairRiskModelConfig(**checkpoint['model_config'])
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    split = SPLIT_VALIDATION if args.split == 'validation' else SPLIT_TEST
    result = evaluate_model(
        model,
        paths,
        split,
        args.batch_size,
        device,
        autocast_bfloat16=args.autocast_bfloat16,
        pos_weight=None,
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'train':
        train(args)
    else:
        evaluate_checkpoint(args)


if __name__ == '__main__':
    main()
