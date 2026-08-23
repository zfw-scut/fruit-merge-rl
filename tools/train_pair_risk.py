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
    for level in range(7, 12):
        mask = levels.eq(level)
        result['by_level'][str(level)] = risk_metrics(
            logits[mask], labels[mask]
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
    positives, negatives = _training_counts(dataset_manifest)
    pos_weight_value = min(20.0, max(1.0, negatives / positives))
    pos_weight = torch.tensor(pos_weight_value, device=device)
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
        'pos_weight': pos_weight_value,
        'autocast_bfloat16': bool(args.autocast_bfloat16),
    }
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
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
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    batch['label'].float(),
                    pos_weight=pos_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip_norm
            )
            optimizer.step()
            count = int(logits.numel())
            loss_sum += float(loss.item()) * count
            samples += count
        validation = evaluate_model(
            model,
            paths,
            SPLIT_VALIDATION,
            args.batch_size * 2,
            device,
            autocast_bfloat16=args.autocast_bfloat16,
            pos_weight=pos_weight,
        )
        row = {
            'epoch': epoch,
            'train_loss': loss_sum / max(1, samples),
            'train_samples': samples,
            'epoch_seconds': time.perf_counter() - epoch_started,
            'validation': validation,
        }
        history.append(row)
        score = float(validation.get('average_precision', math.nan))
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
            'best_validation_average_precision': best_score,
            'elapsed_seconds': time.perf_counter() - started,
            'latest': row,
        }
        _atomic_json(output_dir / 'training_progress.json', progress)
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
        pos_weight=pos_weight,
    )
    final = {
        'status': 'complete',
        'best_epoch': best_epoch,
        'best_validation_average_precision': best_score,
        'test': test_result,
        'elapsed_seconds': time.perf_counter() - started,
        'checkpoint': str((output_dir / 'best.pt').resolve()),
    }
    _atomic_json(output_dir / 'training_progress.json', final)
    _atomic_json(output_dir / 'evaluation.json', final)
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
    positives, negatives = _training_counts(manifest)
    pos_weight = torch.tensor(
        min(20.0, max(1.0, negatives / positives)), device=device
    )
    split = SPLIT_VALIDATION if args.split == 'validation' else SPLIT_TEST
    result = evaluate_model(
        model,
        paths,
        split,
        args.batch_size,
        device,
        autocast_bfloat16=args.autocast_bfloat16,
        pos_weight=pos_weight,
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
