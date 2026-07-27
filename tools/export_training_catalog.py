#!/usr/bin/env python3
"""把本地 ``runs/`` 中的训练产物导出为可由 Git 跟踪的轻量实验目录。

训练产生的 checkpoint 和 ReplayBuffer 往往很大，因此项目通过 ``.gitignore``
忽略整个 ``runs/``。本脚本只读取这些本地产物，提取配置、进度、单局得分、
reward breakdown、StateAnalyzer 性能、吞吐和文件体积等摘要，写入
``docs/training_runs/``。Reward V2 和历史 Reward V1 使用不同指标列，导出时会
分别识别并保留原始语义。

生成目录不复制模型权重、ReplayBuffer 或完整 CSV。迁移后的开发者和 Agent
可以先阅读摘要，再按 ``artifacts.md`` 决定需要另外搬运哪些大文件。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / 'runs'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'training_runs'

# Reward V2 和历史 reward 使用不同的 CSV 列。这里同时登记两套语义名称，
# 让新实验能完整导出 potential，旧实验也不会因为字段退役而丢失摘要。
REWARD_METRIC_FIELDS = (
    ('total', ('collect_mean_reward_total',)),
    ('task_reward', ('collect_mean_task_reward',)),
    (
        'potential_shaping_reward',
        ('collect_mean_potential_shaping_reward',),
    ),
    ('terminal_penalty', ('collect_mean_terminal_penalty',)),
    ('previous_potential', ('collect_mean_previous_potential',)),
    ('next_potential', ('collect_mean_next_potential',)),
    ('potential_delta', ('collect_mean_potential_delta',)),
    (
        'previous_top_connected_capacity',
        ('collect_mean_previous_top_connected_capacity',),
    ),
    (
        'next_top_connected_capacity',
        ('collect_mean_next_top_connected_capacity',),
    ),
    (
        'previous_recoverability',
        ('collect_mean_previous_recoverability',),
    ),
    (
        'next_recoverability',
        ('collect_mean_next_recoverability',),
    ),
    (
        'previous_chain_readiness',
        ('collect_mean_previous_chain_readiness',),
    ),
    (
        'next_chain_readiness',
        ('collect_mean_next_chain_readiness',),
    ),
    ('merge_event_count', ('collect_mean_merge_event_count',)),
    # Reward V1 历史字段。只用于读取旧训练，不代表当前训练仍会生成这些奖励。
    ('score_reward', ('collect_mean_score_reward',)),
    ('survival_bonus', ('collect_mean_survival_bonus',)),
    ('height_delta_reward', ('collect_mean_height_delta_reward',)),
    ('danger_penalty', ('collect_mean_danger_penalty',)),
)

KEY_CONFIG_FIELDS = (
    'device',
    'total_updates',
    'warmup_steps',
    'collect_per_update',
    'batch_size',
    'replay_capacity',
    'hot_replay_capacity',
    'epsilon_schedule',
    'epsilon_start',
    'epsilon_end',
    'epsilon_decay_steps',
    'learning_rate',
    'gamma',
    'n_step',
    'target_update_interval',
    'grad_clip_norm',
    'causal_replay_capacity',
    'causal_batch_size',
    'causal_update_interval',
    'lambda_rule',
    'lambda_cf',
    'counterfactual_return_scale',
    'counterfactual_target_clip',
    'counterfactual_enabled',
    'counterfactual_workers',
    'counterfactual_horizon',
    'counterfactual_cost_ratio',
    'counterfactual_hard_limit',
    'counterfactual_min_real_steps',
    'counterfactual_snapshot_ring_size',
    'counterfactual_proposal_sample_rate',
    'counterfactual_max_alternatives',
    'shapley_enabled',
    'shapley_event_ratio_max',
    'shapley_candidate_limit',
    'shapley_paired_permutations',
    'shapley_minimum_candidates',
    'shapley_minimum_utility',
    'checkpoint_keep_last',
    'hidden_dim',
    'message_layers',
    'action_count',
    'physics_mode',
    'physics_fps',
    'max_physics_frames',
    'stable_frames',
    'space_iterations',
    'num_envs',
    'async_rollout',
    'lambda_phi',
    'capacity_weight',
    'recoverability_weight',
    'chain_readiness_weight',
    'terminal_penalty',
    # 以下字段只存在于 Reward V1 历史配置；保留用于解释旧训练。
    'score_scale',
    'survival_bonus',
    'height_delta_weight',
    'danger_height_weight',
)


@dataclass(frozen=True)
class FileGroup:
    """一种训练产物的数量和总体积。"""

    count: int
    size_bytes: int


@dataclass(frozen=True)
class RunSummary:
    """一个训练目录的可迁移摘要。"""

    run_id: str
    source_dir: Path
    status: str
    reward_version: str
    config: dict[str, Any]
    config_args: dict[str, Any]
    final_metrics: dict[str, Any]
    episode_stats: dict[str, Any]
    reward_stats: dict[str, Any]
    artifact_stats: dict[str, Any]
    total_size_bytes: int


def parse_args() -> argparse.Namespace:
    """解析输入和输出目录。"""

    parser = argparse.ArgumentParser(
        description='将 runs/ 训练产物导出为 docs/training_runs/ 轻量目录。',
    )
    parser.add_argument(
        '--runs-dir',
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help='训练产物根目录，默认是项目下的 runs/。',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='摘要输出目录，默认是 docs/training_runs/。',
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    """读取 JSON；文件不存在或格式损坏时返回空字典。"""

    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 行，并忽略完全空白的记录。"""

    if not path.is_file():
        return []
    try:
        with path.open('r', encoding='utf-8', newline='') as file:
            return [
                row
                for row in csv.DictReader(file)
                if row and any(str(value).strip() for value in row.values())
            ]
    except (OSError, csv.Error):
        return []


def number(value: Any) -> float | None:
    """把 CSV/JSON 标量安全转换成有限浮点数。"""

    if value is None or value == '':
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    """把整数形式的标量安全转换为 int。"""

    parsed = number(value)
    return None if parsed is None else int(parsed)


def truthy(value: Any) -> bool:
    """兼容 CSV 中的 0/1 和 true/false。"""

    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def json_value(value: Any) -> Any:
    """安全解析 CSV 中的结构化 JSON 指标。"""

    if value is None or value == '':
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def last_number(rows: Iterable[dict[str, str]], field: str) -> float | None:
    """从后向前找到某个指标最近一次非空数值。"""

    for row in reversed(list(rows)):
        parsed = number(row.get(field))
        if parsed is not None:
            return parsed
    return None


def percentile(values: list[float], ratio: float) -> float | None:
    """用线性插值计算小样本也可用的分位数。"""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def directory_size(path: Path) -> int:
    """递归统计目录体积，忽略迁移过程中可能消失的文件。"""

    total = 0
    if not path.exists():
        return total
    for child in path.rglob('*'):
        if not child.is_file():
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


def file_group(paths: Iterable[Path]) -> FileGroup:
    """汇总一组文件。"""

    count = 0
    size_bytes = 0
    for path in paths:
        if not path.is_file():
            continue
        try:
            size_bytes += path.stat().st_size
        except OSError:
            continue
        count += 1
    return FileGroup(count=count, size_bytes=size_bytes)


def sha256(path: Path) -> str | None:
    """只为关键 checkpoint 计算校验和，避免遍历大型 ReplayBuffer。"""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open('rb') as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def human_size(size_bytes: int | float | None) -> str:
    """把字节数格式化为便于阅读的二进制单位。"""

    if size_bytes is None:
        return '未记录'
    value = float(size_bytes)
    units = ('B', 'KiB', 'MiB', 'GiB', 'TiB')
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f'{value:.1f} {unit}'
        value /= 1024
    return f'{value:.1f} TiB'


def format_value(value: Any, digits: int = 3) -> str:
    """统一 Markdown 中的空值、整数和浮点数格式。"""

    if value is None or value == '':
        return '未记录'
    if isinstance(value, bool):
        return '是' if value else '否'
    if isinstance(value, int):
        return f'{value:,}'
    if isinstance(value, float):
        if value.is_integer():
            return f'{int(value):,}'
        if value != 0 and abs(value) < 10 ** -digits:
            return f'{value:.{digits}e}'
        return f'{value:,.{digits}f}'
    return str(value)


def markdown_cell(value: Any) -> str:
    """转义会破坏 Markdown 表格的字符。"""

    return format_value(value).replace('|', r'\|').replace('\n', ' ')


def weighted_metric_mean(rows: list[dict[str, str]], field: str) -> float | None:
    """按 collect_steps 加权统计 reward breakdown 的整次训练均值。"""

    weighted_total = 0.0
    weight_total = 0.0
    for row in rows:
        value = number(row.get(field))
        weight = number(row.get('collect_steps'))
        if value is None or weight is None or weight <= 0:
            continue
        weighted_total += value * weight
        weight_total += weight
    if weight_total <= 0:
        return None
    return weighted_total / weight_total


def first_metric_field(
        rows: list[dict[str, str]],
        candidates: Iterable[str]) -> str | None:
    """返回 CSV 中实际存在的第一个候选字段。

    当前候选 tuple 通常只有一个名字；保留别名能力可以兼容短期实验中曾经使用过、
    但没有进入最终接口的列名，而无需复制整套汇总逻辑。
    """

    available = {
        field
        for row in rows
        for field in row
    }
    return next(
        (field for field in candidates if field in available),
        None,
    )


def detect_reward_version(
        config_args: dict[str, Any],
        metric_rows: list[dict[str, str]]) -> str:
    """根据配置和 CSV 字段区分当前 Reward V2 与历史奖励。

    历史 ``config.json`` 没有显式版本号，因此优先识别 V2 的 task/potential 字段，
    再退回旧 score/survival/height 字段；两类证据都没有时保持未知。
    """

    available = {
        field
        for row in metric_rows
        for field in row
    }
    if (
            'collect_mean_task_reward' in available
            or 'collect_mean_potential_shaping_reward' in available
            or 'lambda_phi' in config_args):
        return 'Reward V2'
    if any(
            field in available
            for field in (
                'collect_mean_score_reward',
                'collect_mean_survival_bonus',
                'collect_mean_height_delta_reward',
                'collect_mean_danger_penalty',
            )):
        return 'Reward V1（历史）'
    return '未记录'


def collect_final_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    """提取训练结束或中断时的主要指标。"""

    if not rows:
        return {}

    final_row = rows[-1]
    fields = (
        'update_step',
        'env_steps',
        'epsilon',
        'buffer_size',
        'loss',
        'td_loss',
        'rule_rank_loss',
        'weighted_rule_rank_loss',
        'counterfactual_loss',
        'weighted_counterfactual_loss',
        'causal_batch_size',
        'rule_batch_size',
        'counterfactual_batch_size',
        'shapley_batch_size',
        'rule_pair_accuracy',
        'rule_margin_satisfaction_rate',
        'counterfactual_sign_accuracy',
        'counterfactual_mean_abs_error',
        'mean_q',
        'mean_target',
        'mean_reward',
        'mean_abs_td_error',
        'grad_norm',
        'updates_per_second',
        'env_steps_per_second',
        'best_eval_score',
        'best_eval_update',
        'causal_replay_size',
        'causal_replay_positive_count',
        'causal_replay_negative_count',
        'causal_replay_counterfactual_count',
        'causal_replay_rule_count',
        'causal_replay_cf_count',
        'causal_replay_shapley_count',
        'causal_rule_empirical_agreement_count',
        'causal_rule_empirical_disagreement_count',
        'causal_rule_empirical_agreement_rate',
        'counterfactual_proposals_received',
        'counterfactual_proposals_admitted',
        'collect_counterfactual_proposals_transfer_selected',
        'collect_counterfactual_proposals_transfer_throttled',
        'counterfactual_proposals_rejected',
        'counterfactual_pending_tasks',
        'counterfactual_admission_slots_used',
        'counterfactual_admission_slots_available',
        'counterfactual_candidate_pool_capacity',
        'counterfactual_candidate_pool_count',
        'counterfactual_candidate_offers',
        'counterfactual_candidate_pool_evictions',
        'counterfactual_candidate_dispatch_attempts',
        'counterfactual_candidate_dispatch_admitted',
        'counterfactual_candidate_close_dropped',
        'counterfactual_results_completed',
        'counterfactual_results_partial',
        'counterfactual_results_failed',
        'counterfactual_reproduction_passed',
        'counterfactual_reproduction_failed',
        'counterfactual_label_ready_results',
        'counterfactual_samples_inserted',
        'counterfactual_tokens_reserved',
        'counterfactual_tokens_consumed',
        'counterfactual_tokens_refunded',
        'counterfactual_actual_token_ratio',
        'counterfactual_projected_token_ratio',
        'counterfactual_hard_budget_respected',
        'counterfactual_circuit_open',
        'shapley_events_observed',
        'shapley_events_selected',
        'shapley_tasks_submitted',
        'shapley_tasks_completed',
        'shapley_tasks_failed',
        'shapley_reproduction_passed',
        'shapley_reproduction_failed',
        'shapley_samples_inserted',
        'shapley_tokens_consumed',
        'checkpoint_bytes',
        'checkpoint_pruned_count',
        'save_seconds',
        'collect_max_fruit_level',
        # Reward V2/StateAnalyzer 校准指标。旧 CSV 中不存在时自然得到 None。
        'collect_p95_abs_potential_shaping_reward',
        'collect_state_analysis_calls',
        'collect_state_analysis_seconds',
        'collect_mean_state_analysis_seconds',
        'collect_state_analysis_cache_hits',
        'collect_state_analysis_degraded_count',
        'collect_state_analysis_cache_hit_rate',
        'collect_state_analysis_degraded_rate',
    )
    result = {field: number(final_row.get(field)) for field in fields}
    for field in (
            'collect_attribution_event_status_counts',
            'collect_attribution_confidence_tier_counts',
            'collect_merge_level_counts',
            'causal_replay_cause_type_counts',
            'counterfactual_drop_reasons',
            'shapley_drop_reasons'):
        result[field] = json_value(final_row.get(field))

    # 评估列只有在 eval_interval 命中时才有值，因此不能只看最后一行。
    for field in ('eval_score_mean', 'eval_score_max', 'eval_score_min', 'eval_reward_mean', 'eval_length_mean'):
        result[field] = last_number(rows, field)
    return result


def collect_episode_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """汇总训练过程中完成的完整游戏。"""

    scores = [value for row in rows if (value := number(row.get('score'))) is not None]
    rewards = [value for row in rows if (value := number(row.get('episode_reward'))) is not None]
    lengths = [value for row in rows if (value := number(row.get('episode_length'))) is not None]

    stats: dict[str, Any] = {
        'count': len(scores),
        'terminated_count': sum(truthy(row.get('terminated')) for row in rows),
        'truncated_count': sum(truthy(row.get('truncated')) for row in rows),
    }
    if scores:
        stats.update(
            score_mean=statistics.fmean(scores),
            score_median=statistics.median(scores),
            score_std=statistics.pstdev(scores),
            score_min=min(scores),
            score_p10=percentile(scores, 0.10),
            score_p90=percentile(scores, 0.90),
            score_max=max(scores),
        )
    if rewards:
        stats['reward_mean'] = statistics.fmean(rewards)
    if lengths:
        stats['length_mean'] = statistics.fmean(lengths)
        stats['length_median'] = statistics.median(lengths)
    return stats


def collect_artifact_stats(run_dir: Path) -> dict[str, Any]:
    """统计 checkpoint、ReplayBuffer、图表、日志和关键文件。"""

    checkpoint_files = sorted((run_dir / 'checkpoints').glob('*.pt'))
    replay_files = sorted((run_dir / 'replay_cold').glob('*.pt'))
    replay_files += sorted((run_dir / 'replay_store').glob('*.pt'))
    plot_files = sorted((run_dir / 'plots').glob('*'))
    log_files = sorted(run_dir.rglob('*.log'))

    groups = {
        'checkpoints': file_group(checkpoint_files),
        'replay': file_group(replay_files),
        'plots': file_group(plot_files),
        'logs': file_group(log_files),
    }

    key_files: dict[str, Any] = {}
    for name in ('config.json', 'metrics.csv', 'episode_metrics.csv'):
        path = run_dir / name
        if path.is_file():
            key_files[name] = {
                'relative_path': name,
                'size_bytes': path.stat().st_size,
                'sha256': None,
            }

    for name in ('best.pt', 'latest.pt'):
        path = run_dir / 'checkpoints' / name
        if path.is_file():
            key_files[f'checkpoints/{name}'] = {
                'relative_path': f'checkpoints/{name}',
                'size_bytes': path.stat().st_size,
                'sha256': sha256(path),
            }

    return {
        'groups': {
            name: {'count': group.count, 'size_bytes': group.size_bytes}
            for name, group in groups.items()
        },
        'key_files': key_files,
    }


def detect_status(config_args: dict[str, Any], final_metrics: dict[str, Any]) -> str:
    """依据实际更新数判断训练是完成、未完成还是无法判断。"""

    target = integer(config_args.get('total_updates'))
    actual = integer(final_metrics.get('update_step'))
    if target is None or actual is None:
        return '无法判断'
    if actual >= target:
        return '已完成'
    return '未完成或中断'


def build_run_summary(run_dir: Path) -> RunSummary | None:
    """从一个包含 config.json 和 metrics.csv 的目录构建摘要。"""

    config_path = run_dir / 'config.json'
    metrics_path = run_dir / 'metrics.csv'
    if not config_path.is_file() or not metrics_path.is_file():
        return None

    config = read_json(config_path)
    args = config.get('args')
    config_args = args if isinstance(args, dict) else {}
    metric_rows = read_csv_rows(metrics_path)
    episode_rows = read_csv_rows(run_dir / 'episode_metrics.csv')
    final_metrics = collect_final_metrics(metric_rows)
    reward_stats = {}
    for name, field_candidates in REWARD_METRIC_FIELDS:
        field = first_metric_field(metric_rows, field_candidates)
        if field is not None:
            reward_stats[name] = weighted_metric_mean(metric_rows, field)

    return RunSummary(
        run_id=run_dir.name,
        source_dir=run_dir,
        status=detect_status(config_args, final_metrics),
        reward_version=detect_reward_version(config_args, metric_rows),
        config=config,
        config_args=config_args,
        final_metrics=final_metrics,
        episode_stats=collect_episode_stats(episode_rows),
        reward_stats=reward_stats,
        artifact_stats=collect_artifact_stats(run_dir),
        total_size_bytes=directory_size(run_dir),
    )


def key_config(config_args: dict[str, Any]) -> dict[str, Any]:
    """只保留 Agent 快速比较实验时最常用的配置字段。"""

    return {
        field: config_args[field]
        for field in KEY_CONFIG_FIELDS
        if field in config_args
    }


def write_json(path: Path, value: Any) -> None:
    """以稳定、可读的格式写入结构化摘要。"""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def write_run_summary(output_dir: Path, summary: RunSummary) -> None:
    """生成单个训练实验的配置快照、摘要和产物清单。"""

    run_output = output_dir / 'runs' / summary.run_id
    run_output.mkdir(parents=True, exist_ok=True)

    # 保存原始 config.json 的结构化副本，确保未来 TOML 改动不会覆盖历史参数。
    write_json(run_output / 'config.json', summary.config)

    metrics_summary = {
        'run_id': summary.run_id,
        'status': summary.status,
        'reward_version': summary.reward_version,
        'source_relative_path': f'runs/{summary.run_id}',
        'total_size_bytes': summary.total_size_bytes,
        'config': key_config(summary.config_args),
        'identity': {
            'run_manifest': summary.config.get('run_manifest'),
            'fingerprints': summary.config.get('fingerprints'),
            'runtime': summary.config.get('runtime'),
            'git': summary.config.get('git'),
        },
        'final_metrics': summary.final_metrics,
        'episode_stats': summary.episode_stats,
        'reward_breakdown_weighted_mean': summary.reward_stats,
        'artifacts': summary.artifact_stats,
    }
    write_json(run_output / 'metrics_summary.json', metrics_summary)

    created_at = summary.config.get('created_at')
    final = summary.final_metrics
    episode = summary.episode_stats
    config = summary.config_args
    summary_lines = [
        f'# {summary.run_id}',
        '',
        '## 实验概况',
        '',
        '| 字段 | 值 |',
        '| --- | ---: |',
        f'| 状态 | {markdown_cell(summary.status)} |',
        f'| 奖励版本 | {markdown_cell(summary.reward_version)} |',
        f'| 创建时间 | {markdown_cell(created_at)} |',
        f'| 目标更新数 | {markdown_cell(integer(config.get("total_updates")))} |',
        f'| 实际更新数 | {markdown_cell(integer(final.get("update_step")))} |',
        f'| 环境投放数 | {markdown_cell(integer(final.get("env_steps")))} |',
        f'| 完整游戏数 | {markdown_cell(episode.get("count"))} |',
        f'| 训练设备 | {markdown_cell(config.get("device"))} |',
        f'| 物理模式 | {markdown_cell(config.get("physics_mode"))} |',
        f'| 本地数据体积 | {human_size(summary.total_size_bytes)} |',
        '',
        (
            '> 代码身份：'
            f'`{summary.config.get("git", {}).get("commit", "历史产物未记录")}`；'
            '完整 manifest、指纹与运行时见 `metrics_summary.json`。'
        ),
        '',
        '## 模型与训练参数',
        '',
        '| 参数 | 值 |',
        '| --- | ---: |',
    ]
    for field, value in key_config(config).items():
        summary_lines.append(f'| `{field}` | {markdown_cell(value)} |')

    summary_lines.extend(
        [
            '',
            '## 最终训练指标',
            '',
            '| 指标 | 值 |',
            '| --- | ---: |',
        ]
    )
    for field in (
        'epsilon',
        'buffer_size',
        'loss',
        'td_loss',
        'weighted_rule_rank_loss',
        'weighted_counterfactual_loss',
        'mean_q',
        'mean_target',
        'mean_reward',
        'mean_abs_td_error',
        'grad_norm',
        'updates_per_second',
        'env_steps_per_second',
        'eval_score_mean',
        'eval_score_max',
        'eval_score_min',
        'best_eval_score',
        'best_eval_update',
        'causal_replay_size',
        'causal_replay_rule_count',
        'causal_replay_cf_count',
        'causal_replay_shapley_count',
        'causal_rule_empirical_agreement_rate',
        'counterfactual_proposals_received',
        'counterfactual_proposals_admitted',
        'collect_counterfactual_proposals_transfer_selected',
        'collect_counterfactual_proposals_transfer_throttled',
        'counterfactual_reproduction_passed',
        'counterfactual_reproduction_failed',
        'counterfactual_samples_inserted',
        'counterfactual_actual_token_ratio',
        'counterfactual_hard_budget_respected',
        'shapley_events_selected',
        'shapley_tasks_completed',
        'shapley_samples_inserted',
        'checkpoint_bytes',
        'save_seconds',
        'checkpoint_step_materialization',
        'checkpoint_extra_materialization',
    ):
        summary_lines.append(f'| `{field}` | {markdown_cell(final.get(field))} |')

    reward_v2_calibration_fields = (
        'collect_p95_abs_potential_shaping_reward',
        'collect_state_analysis_calls',
        'collect_state_analysis_seconds',
        'collect_mean_state_analysis_seconds',
        'collect_state_analysis_cache_hits',
        'collect_state_analysis_degraded_count',
        'collect_state_analysis_cache_hit_rate',
        'collect_state_analysis_degraded_rate',
    )
    if any(
            final.get(field) is not None
            for field in reward_v2_calibration_fields):
        for field in reward_v2_calibration_fields:
            summary_lines.append(
                f'| `{field}` | {markdown_cell(final.get(field))} |'
            )

    summary_lines.extend(
        [
            '',
            '## 单局得分分布',
            '',
            '| 指标 | 值 |',
            '| --- | ---: |',
        ]
    )
    for field in (
        'count',
        'score_mean',
        'score_median',
        'score_std',
        'score_min',
        'score_p10',
        'score_p90',
        'score_max',
        'reward_mean',
        'length_mean',
        'length_median',
        'terminated_count',
        'truncated_count',
    ):
        summary_lines.append(f'| `{field}` | {markdown_cell(episode.get(field))} |')

    summary_lines.extend(
        [
            '',
            '## Reward Breakdown',
            '',
            f'识别版本：**{summary.reward_version}**。',
            '',
            '以下数值按每行 `collect_steps` 加权，表示整个已记录训练区间内每次投放的平均贡献。 '
            'Reward V2 使用 task/potential 字段，历史 Reward V1 保留原 score/survival/height 字段。',
            '',
            '| 奖励项 | 每次投放平均贡献 |',
            '| --- | ---: |',
        ]
    )
    if summary.reward_stats:
        for field, value in summary.reward_stats.items():
            summary_lines.append(f'| `{field}` | {markdown_cell(value)} |')
    else:
        summary_lines.append('| 未记录 | 未记录 |')

    summary_lines.extend(
        [
            '',
            '## 迁移提示',
            '',
            '- 分析训练趋势时，优先查看本目录的 `metrics_summary.json`。',
            '- 复现参数时，使用本目录的 `config.json`，不要依赖后来可能修改的默认 TOML。',
            '- 观看模型至少需要另行迁移 `checkpoints/best.pt` 或 `checkpoints/latest.pt`。',
            '- 只有需要继续使用原经验池训练时，才迁移体积较大的 ReplayBuffer。',
            '',
        ]
    )
    (run_output / 'summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')

    group_lines = [
        f'# {summary.run_id} 产物清单',
        '',
        f'原始相对目录：`runs/{summary.run_id}`',
        '',
        '| 产物类型 | 文件数 | 总体积 | 是否进入 Git 摘要 |',
        '| --- | ---: | ---: | --- |',
    ]
    for name, values in summary.artifact_stats['groups'].items():
        group_lines.append(
            f'| `{name}` | {values["count"]:,} | {human_size(values["size_bytes"])} | 否，只记录元数据 |'
        )

    group_lines.extend(
        [
            '',
            '## 关键文件',
            '',
            '| 相对路径 | 体积 | SHA-256 |',
            '| --- | ---: | --- |',
        ]
    )
    key_files = summary.artifact_stats['key_files']
    if key_files:
        for values in key_files.values():
            checksum = values.get('sha256') or '未计算'
            group_lines.append(
                f'| `{values["relative_path"]}` | {human_size(values["size_bytes"])} | `{checksum}` |'
            )
    else:
        group_lines.append('| 未发现 | 0 B | 未计算 |')

    group_lines.extend(
        [
            '',
            '## 迁移等级',
            '',
            '- 最小分析包：`config.json`、`metrics.csv`、`episode_metrics.csv` 和 `plots/`。',
            '- 模型观看包：最小分析包加 `checkpoints/best.pt` 或 `checkpoints/latest.pt`。',
            '- 断点训练包：模型观看包加训练所需的 checkpoint 和 ReplayBuffer；体积可能很大。',
            '',
        ]
    )
    (run_output / 'artifacts.md').write_text('\n'.join(group_lines), encoding='utf-8')


def classify_other_outputs(runs_dir: Path, training_ids: set[str]) -> list[dict[str, Any]]:
    """列出没有标准训练指标的辅助输出，避免迁移时遗漏诊断材料。"""

    outputs = []
    for path in sorted(runs_dir.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or path.name in training_ids:
            continue

        if (path / 'summary.csv').is_file():
            category = '物理模式或评估对比'
        elif any(path.rglob('summary.json')) or any(path.rglob('system_metrics.csv')):
            category = '资源或 CUDA 诊断'
        elif path.name.endswith('logs') or 'log' in path.name:
            category = '日志集合'
        else:
            category = '其他运行输出'

        outputs.append(
            {
                'name': path.name,
                'category': category,
                'size_bytes': directory_size(path),
                'relative_path': f'runs/{path.name}',
            }
        )
    return outputs


def write_index(output_dir: Path, summaries: list[RunSummary], other_outputs: list[dict[str, Any]]) -> None:
    """写入供迁移后 Agent 首先阅读的总索引。"""

    generated_at = dt.datetime.now().astimezone().isoformat(timespec='seconds')
    lines = [
        '# 训练实验索引',
        '',
        f'生成时间：`{generated_at}`',
        '',
        '本索引由 `tools/export_training_catalog.py` 从本地 `runs/` 自动提取。',
        '数值只代表 CSV 中已经落盘的最后状态，不代表后台进程退出前尚未写入的数据。',
        '',
        '## 训练实验',
        '',
        '| Run | 状态 | Reward | Updates | Env Steps | Episodes | 平均分 | 中位数 | 最高分 | Eval 最佳 | 数据体积 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for summary in summaries:
        final = summary.final_metrics
        episode = summary.episode_stats
        lines.append(
            '| [{run}](runs/{run}/summary.md) | {status} | {reward_version} | {updates} | {env_steps} | '
            '{episodes} | {mean} | {median} | {maximum} | {best_eval} | {size} |'.format(
                run=summary.run_id,
                status=markdown_cell(summary.status),
                reward_version=markdown_cell(summary.reward_version),
                updates=markdown_cell(integer(final.get('update_step'))),
                env_steps=markdown_cell(integer(final.get('env_steps'))),
                episodes=markdown_cell(episode.get('count')),
                mean=markdown_cell(episode.get('score_mean')),
                median=markdown_cell(episode.get('score_median')),
                maximum=markdown_cell(episode.get('score_max')),
                best_eval=markdown_cell(final.get('best_eval_score')),
                size=human_size(summary.total_size_bytes),
            )
        )

    lines.extend(
        [
            '',
            '## 非训练输出',
            '',
            '这些目录没有标准 `config.json + metrics.csv` 训练组合，因此不参与模型得分比较。',
            '',
            '| 目录 | 类型 | 本地相对路径 | 数据体积 |',
            '| --- | --- | --- | ---: |',
        ]
    )
    if other_outputs:
        for output in other_outputs:
            lines.append(
                f'| `{output["name"]}` | {output["category"]} | '
                f'`{output["relative_path"]}` | {human_size(output["size_bytes"])} |'
            )
    else:
        lines.append('| 无 | 无 | 无 | 0 B |')

    lines.extend(
        [
            '',
            '## 阅读说明',
            '',
            '- 比较模型效果时以真实 `episode score` 为准，不以 shaped reward 代替游戏分数。',
            '- `未完成或中断` 仅表示最后一行 update 小于配置目标，不能判断进程是手动停止还是异常退出。',
            '- Smoke/debug 运行主要验证代码链路，不应与正式长训直接比较。',
            '- 本目录没有复制 checkpoint 和 ReplayBuffer；需要时按各 Run 的 `artifacts.md` 单独迁移。',
            '',
        ]
    )
    (output_dir / 'INDEX.md').write_text('\n'.join(lines), encoding='utf-8')


def write_readme(output_dir: Path) -> None:
    """写入目录用途、维护方式和迁移流程。"""

    text = """# 训练实验目录

本目录保存从本地 `runs/` 提取的轻量训练摘要，目的是让迁移后的开发者和
Agent 在没有原机器全部训练文件的情况下，仍能了解历史实验参数、训练进度、
单局得分、reward breakdown、性能和大型产物位置。

## 阅读顺序

1. `INDEX.md`：查看所有训练实验和辅助输出。
2. `runs/<run_id>/summary.md`：查看某次训练的人类可读总结。
3. `runs/<run_id>/metrics_summary.json`：供程序或 Agent 做结构化比较。
4. `runs/<run_id>/config.json`：查看该次训练真正生效的参数。
5. `runs/<run_id>/artifacts.md`：决定还需要单独迁移哪些大文件。

## 更新方式

在原始训练数据仍位于项目 `runs/` 时运行：

```bash
python tools/export_training_catalog.py
```

训练数据在其他目录时：

```bash
python tools/export_training_catalog.py \\
  --runs-dir /path/to/runs \\
  --output-dir docs/training_runs
```

脚本会重建 `runs/` 下的生成摘要。不要在生成的单 Run 目录中手工记录长期
结论；需要人工补充的分析应另建文档，避免下次导出时丢失。

## Git 与大型文件

本目录只包含小型 Markdown 和 JSON，不包含：

- PyTorch checkpoint；
- ReplayBuffer 冷段；
- 完整训练 CSV；
- 资源监控原始日志。

这些原始文件仍位于被 `.gitignore` 忽略的 `runs/`。迁移模型时至少另行复制
所需 checkpoint；只有继续使用原经验池训练时才需要复制 ReplayBuffer。

## 数据解释限制

- 早期训练版本没有 `episode_metrics.csv` 或 reward breakdown 时，对应字段会显示“未记录”。
- 当前导出器同时识别 Reward V2 的 task/potential 指标与 Reward V1 的
  score/survival/height 指标；旧字段只用于解释历史实验，不代表仍在当前训练中启用。
- Reward V2 的 `StateAnalyzer` 性能、降级率和 shaping p95 只会出现在采用新版
  `metrics.csv` 的实验中。
- 历史 `config.json` 没有 Git commit 字段，因此不能可靠推断训练对应的源码提交。
- 指标摘要来自已经落盘的 CSV；突然断电前尚未 flush 的最后几轮不会出现。
"""
    (output_dir / 'README.md').write_text(text, encoding='utf-8')


def export_catalog(runs_dir: Path, output_dir: Path) -> list[RunSummary]:
    """执行完整扫描并返回已导出的训练摘要。"""

    runs_dir = runs_dir.resolve()
    output_dir = output_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f'runs directory does not exist: {runs_dir}')

    summaries = []
    for run_dir in sorted(runs_dir.iterdir(), key=lambda item: item.name):
        if not run_dir.is_dir():
            continue
        summary = build_run_summary(run_dir)
        if summary is not None:
            summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_runs_dir = output_dir / 'runs'
    if generated_runs_dir.exists():
        shutil.rmtree(generated_runs_dir)
    generated_runs_dir.mkdir(parents=True)

    write_readme(output_dir)
    for summary in summaries:
        write_run_summary(output_dir, summary)

    other_outputs = classify_other_outputs(runs_dir, {summary.run_id for summary in summaries})
    write_index(output_dir, summaries, other_outputs)
    return summaries


def main() -> int:
    """命令行入口。"""

    args = parse_args()
    summaries = export_catalog(args.runs_dir, args.output_dir)
    print(
        f'exported {len(summaries)} training runs to '
        f'{args.output_dir.resolve()}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
