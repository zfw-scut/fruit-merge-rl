#!/usr/bin/env python3
"""只读审计一次完整因果训练 run 是否通过阶段结束门禁。

本工具只读取训练输出，不 import 训练循环，也不会修复、截断或补写 run。默认把
JSON 结果写到 stdout；显式 ``--output`` 时使用同目录临时文件和 ``os.replace``
原子写入，并拒绝把审计结果写回被检查的 run 目录。

``checkpoints/latest.pt`` 包含 pickle 数据。和训练恢复入口一样，本工具只应读取
项目自身生成、来源可信的 checkpoint；加载时统一映射到 CPU。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    # torch.load 反序列化 TensorTransition / GraphTensor 时需要能找到项目包。
    sys.path.insert(0, str(SRC_DIR))


READINESS_SCHEMA_VERSION = 3
DEFAULT_SHAPING_P95_LIMIT = (2.0 ** 1.5) * 0.25
DEFAULT_BASELINE_CONFIG = (
    PROJECT_ROOT / 'configs' / 'train_dqn_causal_500k.toml'
)
BASELINE_STAGE_FIELDS = frozenset(
    {
        'checkpoint_keep_last',
        'config',
        'eval_episodes',
        'eval_interval',
        'eval_max_steps',
        'log_interval',
        'overwrite_run_dir',
        'plot_interval',
        'progress_interval',
        'resume',
        'run_dir',
        'save_interval',
        'total_updates',
        'warmup_steps',
    }
)

# 这些 metrics 字段不是标量：一个是枚举，其余是 JSON object。除此之外的非空
# 单元格都必须能解析为有限数值，避免新增指标静默绕过 NaN/Inf 门禁。
METRIC_TEXT_FIELDS = frozenset(
    {
        'replay_mode',
        'collect_rule_causal_skip_reasons',
        'collect_attribution_event_status_counts',
        'collect_attribution_confidence_tier_counts',
        'collect_merge_level_counts',
        'collect_counterfactual_proposal_skip_reasons',
        'counterfactual_drop_reasons',
        'counterfactual_failure_reasons',
        'counterfactual_failure_diagnostic_codes',
        'counterfactual_failure_trigger_reasons',
        'shapley_drop_reasons',
        'causal_replay_cause_type_counts',
        'checkpoint_step_materialization',
        'checkpoint_extra_materialization',
    }
)
METRIC_JSON_FIELDS = METRIC_TEXT_FIELDS.difference(
    {
        'replay_mode',
        'checkpoint_step_materialization',
        'checkpoint_extra_materialization',
    }
)
EPISODE_TEXT_FIELDS = frozenset({'phase'})

REPRODUCTION_NUMERIC_ERROR_SUFFIXES = (
    'merge_event_position',
    'fruit_position',
    'linear_velocity',
    'orientation',
    'angular_velocity',
)
REPRODUCTION_OUTCOME_METRIC_FIELDS = frozenset(
    f'{prefix}_{field}'
    for prefix in ('counterfactual', 'shapley')
    for field in (
        'numeric_jitter_dropped',
        'semantic_divergence_dropped',
        *(
            f'numeric_jitter_max_{suffix}_error'
            for suffix in REPRODUCTION_NUMERIC_ERROR_SUFFIXES
        ),
    )
)

STRUCTURAL_SUPERVISION_METRIC_FIELDS = frozenset(
    {
        'structural_loss',
        'weighted_structural_loss',
        'structural_valid_count',
        'structural_sample_count',
        'structural_mean_abs_error',
    }
)
CENTRAL_ACTOR_METRIC_FIELDS = frozenset(
    {
        'actor_inference_requests',
        'actor_inference_batches',
        'actor_inference_mean_batch_size',
        'actor_inference_max_batch',
        'actor_inference_seconds',
    }
)
REQUIRED_METRIC_FIELDS = frozenset(
    {
        'update_step',
        'env_steps',
        'td_loss',
        'mean_q',
        'mean_target',
        'mean_abs_td_error',
        'causal_update_applied',
        'rule_batch_size',
        'counterfactual_batch_size',
        'shapley_batch_size',
        'causal_replay_positive_count',
        'causal_replay_negative_count',
        'causal_replay_rule_count',
        'causal_replay_cf_count',
        'collect_counterfactual_snapshot_failures',
        'collect_p95_abs_potential_shaping_reward',
        'collect_state_analysis_degraded_rate',
        'counterfactual_results_completed',
        'counterfactual_results_failed',
        'counterfactual_reproduction_passed',
        'counterfactual_reproduction_failed',
        'counterfactual_samples_inserted',
        'counterfactual_pending_tasks',
        'counterfactual_admission_slots_used',
        'counterfactual_admission_slots_available',
        'counterfactual_candidate_pool_capacity',
        'counterfactual_candidate_pool_count',
        'counterfactual_candidate_offers',
        'counterfactual_candidate_dispatch_attempts',
        'counterfactual_candidate_dispatch_admitted',
        'counterfactual_candidate_close_dropped',
        'counterfactual_actual_token_ratio',
        'counterfactual_projected_token_ratio',
        'counterfactual_hard_budget_respected',
        'counterfactual_drop_reasons',
        'counterfactual_failure_reasons',
        'counterfactual_failure_diagnostic_codes',
        'counterfactual_failure_trigger_reasons',
        'shapley_enabled',
        'shapley_events_observed',
        'shapley_events_selected',
        'shapley_tasks_completed',
        'shapley_tasks_failed',
        'shapley_terminal_dropped',
        'shapley_reproduction_passed',
        'shapley_reproduction_failed',
        'shapley_samples_inserted',
        'checkpoint_bytes',
        'checkpoint_step_materialization',
        'checkpoint_extra_materialization',
    }
) | (
    REPRODUCTION_OUTCOME_METRIC_FIELDS
    | STRUCTURAL_SUPERVISION_METRIC_FIELDS
    | CENTRAL_ACTOR_METRIC_FIELDS
)
REQUIRED_EPISODE_FIELDS = frozenset(
    {
        'episode_index',
        'phase',
        'update_step',
        'env_steps',
        'score',
        'episode_reward',
        'episode_length',
        'terminated',
        'truncated',
    }
)


class ReadinessInputError(RuntimeError):
    """命令或路径使审计无法开始，而不是某一训练门禁未通过。"""


@dataclass(frozen=True, slots=True)
class ReadinessThresholds:
    """阶段结束门禁阈值；全部字段都可由 CLI 覆盖。"""

    max_hard_budget_ratio: float = 0.10
    max_snapshot_failures: int = 0
    max_state_analysis_degraded_rate: float = 0.01
    max_shaping_p95: float = DEFAULT_SHAPING_P95_LIMIT
    max_truncated_rate: float = 0.02
    min_episodes_for_truncated_rate: int = 50
    consecutive_window_limit: int = 3
    max_cf_reproduction_failure_rate: float = 0.01
    min_cf_results_for_failure_rate: int = 100
    max_q_target_td_magnitude: float = 100.0
    max_replay_cold_gb: float = 16.0
    require_shapley_samples: bool = False

    def __post_init__(self):
        unit_interval_fields = (
            'max_hard_budget_ratio',
            'max_state_analysis_degraded_rate',
            'max_truncated_rate',
            'max_cf_reproduction_failure_rate',
        )
        for name in unit_interval_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f'{name} must be finite and in [0, 1]')
        if (
                not math.isfinite(float(self.max_shaping_p95))
                or float(self.max_shaping_p95) < 0.0):
            raise ValueError(
                'max_shaping_p95 must be finite and non-negative'
            )
        if (
                not math.isfinite(float(self.max_replay_cold_gb))
                or float(self.max_replay_cold_gb) < 0.0):
            raise ValueError(
                'max_replay_cold_gb must be finite and non-negative'
            )
        if (
                not math.isfinite(float(self.max_q_target_td_magnitude))
                or float(self.max_q_target_td_magnitude) <= 0.0):
            raise ValueError(
                'max_q_target_td_magnitude must be finite and positive'
            )
        for name in (
                'max_snapshot_failures',
                'min_episodes_for_truncated_rate',
                'min_cf_results_for_failure_rate'):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        if (
                isinstance(self.consecutive_window_limit, bool)
                or int(self.consecutive_window_limit)
                != self.consecutive_window_limit
                or self.consecutive_window_limit < 1):
            raise ValueError(
                'consecutive_window_limit must be a positive integer'
            )


def parse_args(argv=None):
    """解析只读审计入口。"""

    parser = argparse.ArgumentParser(
        description=(
            '只读检查完整因果训练 run 的 5k/10k/正式阶段结束门禁。'
        ),
    )
    parser.add_argument('--run-dir', required=True)
    parser.add_argument(
        '--stage',
        choices=('auto', '5k', '10k', 'formal'),
        default='auto',
        help='阶段决定 Shapley 零样本的解释；auto 按 total_updates 推断。',
    )
    parser.add_argument(
        '--expected-total-updates',
        type=int,
        default=None,
        help=(
            '仅用于诊断；若与 config.json 的 total_updates 不同，门禁必定失败，'
            '不能把未完成长 run 伪装成短 run。'
        ),
    )
    parser.add_argument(
        '--baseline-config',
        default=str(DEFAULT_BASELINE_CONFIG),
        help=(
            '正式语义基线 TOML；阶段长度、warmup 和日志字段会剔除后比较。'
        ),
    )
    parser.add_argument(
        '--max-hard-budget-ratio',
        type=float,
        default=0.10,
    )
    parser.add_argument(
        '--max-snapshot-failures',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--max-state-analysis-degraded-rate',
        type=float,
        default=0.01,
    )
    parser.add_argument(
        '--max-shaping-p95',
        type=float,
        default=DEFAULT_SHAPING_P95_LIMIT,
    )
    parser.add_argument(
        '--max-truncated-rate',
        type=float,
        default=0.02,
    )
    parser.add_argument(
        '--min-episodes-for-truncated-rate',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--consecutive-window-limit',
        type=int,
        default=3,
        help='shaping/degraded 连续多少个日志窗口越限才阻断下一阶段。',
    )
    parser.add_argument(
        '--max-cf-reproduction-failure-rate',
        type=float,
        default=0.01,
    )
    parser.add_argument(
        '--min-cf-results-for-failure-rate',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--max-q-target-td-magnitude',
        type=float,
        default=100.0,
    )
    parser.add_argument(
        '--max-replay-cold-gb',
        type=float,
        default=16.0,
        help=(
            '超过该体积时警告可能残留 hot-only resume 的旧 cold '
            'generation；只告警，不自动删除。'
        ),
    )
    parser.add_argument(
        '--require-shapley-samples',
        action='store_true',
        help='即使阶段允许零样本，也强制要求至少一个 Shapley 样本。',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='原子 JSON 输出路径；省略或传 "-" 时写 stdout。',
    )
    return parser.parse_args(argv)


def _strict_json_load(path):
    """读取严格 JSON，显式拒绝 Python 默认接受的 NaN/Infinity。"""

    def reject_constant(value):
        raise ValueError(f'non-standard JSON constant: {value}')

    with Path(path).open('r', encoding='utf-8') as file_obj:
        return json.load(file_obj, parse_constant=reject_constant)


def _json_safe_finite_errors(value, *, path='$', limit=25):
    """返回 JSON-like 数据中的非有限数值和不支持对象路径。"""

    errors = []

    def visit(item, item_path):
        if len(errors) >= limit:
            return
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                errors.append(f'{item_path}={item!r}')
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f'{item_path}.{key}')
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f'{item_path}[{index}]')
            return
        errors.append(f'{item_path}=<{type(item).__name__}>')

    visit(value, path)
    return errors


def _json_report_value(value):
    """把损坏 checkpoint 中的异常对象降级为可输出的诊断文本。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_report_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_report_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _read_csv(path):
    with Path(path).open('r', newline='', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = tuple(reader.fieldnames or ())
        if not fieldnames:
            raise ValueError('CSV header is missing')
        if len(set(fieldnames)) != len(fieldnames):
            raise ValueError('CSV header contains duplicate fields')
        rows = list(reader)
        for row_index, row in enumerate(rows, start=2):
            if None in row:
                raise ValueError(
                    f'CSV row {row_index} has fields beyond the header'
                )
        return fieldnames, rows


def _validate_csv_cells(
        rows,
        fieldnames,
        *,
        text_fields,
        json_fields=frozenset(),
        limit=25):
    errors = []
    for row_index, row in enumerate(rows, start=2):
        for field in fieldnames:
            value = (row.get(field) or '').strip()
            if not value:
                continue
            if field in json_fields:
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError) as exc:
                    errors.append(
                        f'row {row_index} {field}: invalid JSON: {exc}'
                    )
                else:
                    errors.extend(
                        f'row {row_index} {field}: {error}'
                        for error in _json_safe_finite_errors(
                            parsed,
                            limit=max(1, limit - len(errors)),
                        )
                    )
            elif field not in text_fields:
                try:
                    number = float(value)
                except ValueError:
                    errors.append(
                        f'row {row_index} {field}: '
                        f'not numeric: {value!r}'
                    )
                else:
                    if not math.isfinite(number):
                        errors.append(
                            f'row {row_index} {field}: '
                            f'non-finite: {value!r}'
                        )
            if len(errors) >= limit:
                return errors
    return errors


def _number(row, field):
    value = row.get(field)
    if value is None or not str(value).strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_count(value):
    """把 JSON/checkpoint 计数规范为 int；无效时返回 ``None``。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    integer = int(value)
    return integer if integer == value else None


def _series(rows, field):
    return tuple(
        number
        for row in rows
        if (number := _number(row, field)) is not None
    )


def _metric_counter(row, field):
    """解析一格 JSON 累计计数；任何非法键值都返回 ``None``。"""

    raw = row.get(field)
    if raw is None or not str(raw).strip():
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    normalized = Counter()
    for key, value in payload.items():
        count = _nonnegative_count(value)
        if not isinstance(key, str) or not key or count is None:
            return None
        normalized[key] = count
    return normalized


def _max_number(rows, field, *, default=0.0):
    values = _series(rows, field)
    return max(values) if values else float(default)


def _segmented_cumulative_total(
        values,
        final_value=None,
        *,
        segment_starts=()):
    """汇总会在 resume 后归零的累计计数。

    ``segment_starts`` 是新进程第一次 metrics 行的下标。累计值下降仍作为
    sidecar 缺失时的保守后备边界，但不能只依赖下降：新段第一行完全可能已经
    追平旧段末值。
    """

    normalized = tuple(float(value) for value in values)
    explicit_starts = {
        int(index)
        for index in segment_starts
        if (
            not isinstance(index, bool)
            and isinstance(index, int)
            and 0 < index < len(normalized)
        )
    }
    completed_segments = 0.0
    segment_max = 0.0
    previous = None
    for index, value in enumerate(normalized):
        if (
                previous is not None
                and (
                    index in explicit_starts
                    or value < previous
                )):
            completed_segments += segment_max
            segment_max = 0.0
        segment_max = max(segment_max, value)
        previous = value
    if final_value is not None:
        final_number = _finite_number(final_value)
        if final_number is not None and final_number >= 0.0:
            # 完整 run 的 latest checkpoint 不应落后于同一进程最后一行指标；
            # 若它更小，说明最后一次 resume 后尚无足够日志显式暴露边界。
            if previous is not None and final_number < previous:
                completed_segments += segment_max
                segment_max = final_number
            else:
                segment_max = max(segment_max, final_number)
    total = completed_segments + segment_max
    return int(total) if total.is_integer() else total


def _reproduction_outcome_audit(
        *,
        prefix,
        metrics_rows,
        cumulative,
        strict_matches,
        results_failed,
        reproduction_failed,
        segment_starts,
        max_semantic_rate,
        min_results_for_rate):
    """汇总 strict / 数值抖动 / 语义分叉三态复现账本。

    ``reproduction_failed`` 是为兼容旧日志保留的复现失败计数，其中应同时
    包含 ``numeric_jitter_dropped`` 和 ``semantic_divergence_dropped``；
    两者之外的差额是未归入三态的未知复现失败。``results_failed`` 还可能包含
    subset、效率门或 runner 失败，因此独立返回其非复现门差额供调用方判断。

    两类 drop 计数和 strict 计数在 resume 后会归零，因此使用显式 sidecar
    分段累计；五类数值误差是累计最大值，只需跨所有分段和最终状态取最大值。
    """

    if not isinstance(cumulative, Mapping):
        cumulative = {}

    count_fields = (
        'numeric_jitter_dropped',
        'semantic_divergence_dropped',
    )
    metric_count_values = {}
    invalid_metric_count_rows = {}
    checkpoint_counts = {}
    invalid_checkpoint_count_fields = []
    aggregated_counts = {}
    for field in count_fields:
        metric_field = f'{prefix}_{field}'
        normalized = tuple(
            _nonnegative_count(_number(row, metric_field))
            for row in metrics_rows
        )
        invalid_rows = [
            index
            for index, value in enumerate(normalized)
            if value is None
        ]
        metric_count_values[field] = tuple(
            0 if value is None else value
            for value in normalized
        )
        invalid_metric_count_rows[field] = invalid_rows

        checkpoint_value = _nonnegative_count(cumulative.get(field))
        checkpoint_counts[field] = checkpoint_value
        if checkpoint_value is None:
            invalid_checkpoint_count_fields.append(field)
        aggregated_counts[field] = _segmented_cumulative_total(
            metric_count_values[field],
            checkpoint_value,
            segment_starts=segment_starts,
        )

    metric_error_values = {}
    invalid_metric_error_rows = {}
    checkpoint_error_maxima = {}
    invalid_checkpoint_error_fields = []
    error_maxima = {}
    for suffix in REPRODUCTION_NUMERIC_ERROR_SUFFIXES:
        field = f'numeric_jitter_max_{suffix}_error'
        metric_field = f'{prefix}_{field}'
        normalized = tuple(
            _number(row, metric_field)
            for row in metrics_rows
        )
        invalid_rows = [
            index
            for index, value in enumerate(normalized)
            if value is None or value < 0.0
        ]
        valid_metric_values = tuple(
            value
            for value in normalized
            if value is not None and value >= 0.0
        )
        metric_error_values[field] = valid_metric_values
        invalid_metric_error_rows[field] = invalid_rows

        checkpoint_value = _finite_number(cumulative.get(field))
        if checkpoint_value is None or checkpoint_value < 0.0:
            invalid_checkpoint_error_fields.append(field)
            checkpoint_error_maxima[field] = None
            valid_checkpoint_values = ()
        else:
            checkpoint_error_maxima[field] = checkpoint_value
            valid_checkpoint_values = (checkpoint_value,)
        error_maxima[field] = max(
            (*valid_metric_values, *valid_checkpoint_values),
            default=0.0,
        )

    strict_count = _nonnegative_count(strict_matches)
    failed_count = _nonnegative_count(results_failed)
    legacy_reproduction_failed = _nonnegative_count(
        reproduction_failed
    )
    numeric_jitter = aggregated_counts['numeric_jitter_dropped']
    semantic_divergence = aggregated_counts[
        'semantic_divergence_dropped'
    ]
    unknown_failed = (
        legacy_reproduction_failed
        - numeric_jitter
        - semantic_divergence
        if legacy_reproduction_failed is not None
        else None
    )
    non_gate_result_failed = (
        failed_count - numeric_jitter - semantic_divergence
        if failed_count is not None
        else None
    )
    semantic_denominator = (
        strict_count + numeric_jitter + semantic_divergence
        if strict_count is not None
        else None
    )
    semantic_rate = (
        semantic_divergence / semantic_denominator
        if semantic_denominator
        else None
    )
    semantic_rate_evaluated = (
        semantic_denominator is not None
        and semantic_denominator >= min_results_for_rate
    )
    semantic_rate_passed = (
        not semantic_rate_evaluated
        or (
            semantic_rate is not None
            and semantic_rate <= max_semantic_rate + 1e-12
        )
    )
    valid = (
        strict_count is not None
        and failed_count is not None
        and legacy_reproduction_failed is not None
        and not invalid_checkpoint_count_fields
        and not any(invalid_metric_count_rows.values())
        and not invalid_checkpoint_error_fields
        and not any(invalid_metric_error_rows.values())
    )
    return {
        'valid': valid,
        'strict_match': strict_count,
        'legacy_results_failed': failed_count,
        'legacy_reproduction_failed': legacy_reproduction_failed,
        'numeric_jitter_dropped': numeric_jitter,
        'semantic_divergence_dropped': semantic_divergence,
        'unknown_failed': unknown_failed,
        'non_gate_result_failed': non_gate_result_failed,
        'failure_outcomes_fully_accounted': (
            valid and unknown_failed == 0
        ),
        'semantic_rate_denominator': semantic_denominator,
        'semantic_divergence_rate': semantic_rate,
        'semantic_rate_evaluated': semantic_rate_evaluated,
        'semantic_rate_passed': semantic_rate_passed,
        'semantic_rate_threshold': max_semantic_rate,
        'semantic_rate_minimum_results': min_results_for_rate,
        'numeric_jitter_error_maxima': error_maxima,
        'checkpoint_counts': checkpoint_counts,
        'checkpoint_numeric_jitter_error_maxima': (
            checkpoint_error_maxima
        ),
        'invalid_metric_count_rows': invalid_metric_count_rows,
        'invalid_checkpoint_count_fields': (
            invalid_checkpoint_count_fields
        ),
        'invalid_metric_numeric_error_rows': (
            invalid_metric_error_rows
        ),
        'invalid_checkpoint_numeric_error_fields': (
            invalid_checkpoint_error_fields
        ),
        'aggregation': (
            'resume-segmented counts plus final state; '
            'cross-segment maxima for numeric errors'
        ),
    }


def _resume_sidecar_audit(run_dir, metric_updates):
    """读取 resume sidecar，并把 saved update 映射为 metrics 分段边界。"""

    paths = sorted(
        path
        for path in Path(run_dir).glob('resume_*.json')
        if not path.name.startswith('resume_config_')
    )
    errors = []
    records = []
    saved_steps = set()
    for path in paths:
        try:
            payload = _strict_json_load(path)
            if not isinstance(payload, Mapping):
                raise TypeError('root must be an object')
            saved_step = _nonnegative_count(
                payload.get('saved_update_step')
            )
            if saved_step is None:
                raise ValueError(
                    'saved_update_step must be a non-negative integer'
                )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(
                f'{path.name}: {type(exc).__name__}: {exc}'
            )
            continue
        saved_steps.add(saved_step)
        records.append(
            {
                'path': str(path),
                'saved_update_step': saved_step,
                'saved_env_steps': payload.get('saved_env_steps'),
                'epsilon_schedule_total_updates': payload.get(
                    'epsilon_schedule_total_updates'
                ),
                'requested_total_updates': payload.get(
                    'requested_total_updates'
                ),
                'epsilon_schedule_extended_without_reexpansion': (
                    payload.get(
                        'epsilon_schedule_extended_without_reexpansion'
                    )
                ),
            }
        )

    starts = set()
    for saved_step in saved_steps:
        for index, update in enumerate(metric_updates):
            if update > saved_step:
                if index > 0:
                    starts.add(index)
                break
    return {
        'passed': not errors,
        'errors': errors,
        'records': records,
        'saved_update_steps': sorted(saved_steps),
        'segment_start_indices': sorted(starts),
        'segment_start_updates': [
            metric_updates[index]
            for index in sorted(starts)
        ],
        'segment_count': 1 + len(starts) if metric_updates else 0,
    }


def _max_consecutive(values, predicate):
    current = 0
    maximum = 0
    for value in values:
        if predicate(value):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _strictly_increasing(values):
    return all(right > left for left, right in zip(values, values[1:]))


def _non_decreasing(values):
    return all(right >= left for left, right in zip(values, values[1:]))


def _artifact_info(path):
    path = Path(path)
    return {
        'path': str(path),
        'exists': path.is_file(),
        'bytes': path.stat().st_size if path.is_file() else 0,
    }


def _add_check(checks, name, passed, *, required=True, **details):
    check = {
        'name': str(name),
        'passed': bool(passed),
        'required': bool(required),
        'details': details,
    }
    checks.append(check)
    return check


def _finite_number(value):
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        return None
    return float(value)


def _attribution_warmup_audit(payload, thresholds):
    """验证 warmup 本身已经覆盖状态分析、快照和 shaping 健康信号。"""

    if not isinstance(payload, Mapping):
        return {
            'passed': False,
            'errors': ['root must be an object'],
        }
    errors = []
    steps = _nonnegative_count(payload.get('steps'))
    snapshot_failures = _nonnegative_count(
        payload.get('counterfactual_snapshot_failures')
    )
    state_calls = _nonnegative_count(
        payload.get('state_analysis_calls')
    )
    degraded_count = _nonnegative_count(
        payload.get('state_analysis_degraded_count')
    )
    degraded_rate = _finite_number(
        payload.get('state_analysis_degraded_rate')
    )
    shaping_p95 = _finite_number(
        payload.get('p95_abs_potential_shaping_reward')
    )

    if payload.get('schema_version') != 1:
        errors.append(
            f'schema_version={payload.get("schema_version")!r}, '
            'expected 1'
        )
    if not isinstance(payload.get('phase'), str) or not payload['phase']:
        errors.append('phase must be a non-empty string')
    if steps is None or steps <= 0:
        errors.append(f'steps={payload.get("steps")!r}, expected > 0')
    if snapshot_failures is None:
        errors.append('counterfactual_snapshot_failures invalid')
    elif snapshot_failures > thresholds.max_snapshot_failures:
        errors.append(
            'counterfactual_snapshot_failures='
            f'{snapshot_failures} exceeds '
            f'{thresholds.max_snapshot_failures}'
        )
    if state_calls is None or state_calls <= 0:
        errors.append(
            f'state_analysis_calls={payload.get("state_analysis_calls")!r}, '
            'expected > 0'
        )
    if degraded_count is None:
        errors.append('state_analysis_degraded_count invalid')
    if degraded_rate is None or not 0.0 <= degraded_rate <= 1.0:
        errors.append('state_analysis_degraded_rate invalid')
    elif (
            degraded_rate
            > thresholds.max_state_analysis_degraded_rate + 1e-12):
        errors.append(
            f'state_analysis_degraded_rate={degraded_rate} exceeds '
            f'{thresholds.max_state_analysis_degraded_rate}'
        )
    if (
            state_calls is not None
            and state_calls > 0
            and degraded_count is not None):
        expected_rate = degraded_count / state_calls
        if (
                degraded_count > state_calls
                or degraded_rate is None
                or not math.isclose(
                    degraded_rate,
                    expected_rate,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )):
            errors.append(
                'state analysis degraded count/rate accounting mismatch'
            )
    if shaping_p95 is None or shaping_p95 < 0.0:
        errors.append('p95_abs_potential_shaping_reward invalid')
    elif shaping_p95 > thresholds.max_shaping_p95 + 1e-12:
        errors.append(
            f'p95_abs_potential_shaping_reward={shaping_p95} exceeds '
            f'{thresholds.max_shaping_p95}'
        )
    return {
        'passed': not errors,
        'errors': errors,
        'schema_version': payload.get('schema_version'),
        'phase': payload.get('phase'),
        'steps': steps,
        'counterfactual_snapshot_failures': snapshot_failures,
        'state_analysis_calls': state_calls,
        'state_analysis_degraded_count': degraded_count,
        'state_analysis_degraded_rate': degraded_rate,
        'p95_abs_potential_shaping_reward': shaping_p95,
        'thresholds': {
            'max_snapshot_failures': thresholds.max_snapshot_failures,
            'max_state_analysis_degraded_rate': (
                thresholds.max_state_analysis_degraded_rate
            ),
            'max_shaping_p95': thresholds.max_shaping_p95,
        },
    }


def _baseline_semantic_audit(config_args, baseline_config):
    """按训练入口解析基线，并剔除纯阶段字段后比较语义指纹。"""

    if baseline_config is None:
        return {
            'passed': True,
            'skipped': True,
            'baseline_config': None,
            'reason': 'Python API caller explicitly disabled baseline audit',
        }
    path = Path(baseline_config).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        return {
            'passed': False,
            'skipped': False,
            'baseline_config': str(path),
            'error': 'baseline config does not exist',
        }
    try:
        from daxigua_rl.scripts.train_dqn import (
            parse_args as parse_training_args,
            validate_args as validate_training_args,
        )
        from daxigua_rl.training.checkpointing import config_fingerprint

        baseline_args = parse_training_args(('--config', str(path)))
        validate_training_args(baseline_args)
        baseline_fingerprint = config_fingerprint(
            vars(baseline_args),
            mutable_fields=BASELINE_STAGE_FIELDS,
        )
        run_fingerprint = config_fingerprint(
            config_args,
            mutable_fields=BASELINE_STAGE_FIELDS,
        )
    except Exception as exc:
        return {
            'passed': False,
            'skipped': False,
            'baseline_config': str(path),
            'error': f'{type(exc).__name__}: {exc}',
        }
    return {
        'passed': run_fingerprint == baseline_fingerprint,
        'skipped': False,
        'baseline_config': str(path),
        'stage_fields_excluded': sorted(BASELINE_STAGE_FIELDS),
        'run_semantic_fingerprint': run_fingerprint,
        'baseline_semantic_fingerprint': baseline_fingerprint,
    }


def _expected_replay_manifests(config_args):
    """按训练入口的构造规则从 config.json 推导两个 replay 契约。"""

    if not isinstance(config_args, Mapping):
        raise TypeError('config args must be a mapping')

    def positive_int(name):
        value = config_args.get(name)
        if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0):
            raise ValueError(
                f'config {name} must be a positive integer'
            )
        return value

    capacity = positive_int('replay_capacity')
    configured_hot = config_args.get('hot_replay_capacity')
    if configured_hot is None:
        hot_capacity = min(10_000, capacity)
    else:
        if (
                isinstance(configured_hot, bool)
                or not isinstance(configured_hot, int)
                or configured_hot <= 0):
            raise ValueError(
                'config hot_replay_capacity must be a positive integer '
                'or null'
            )
        hot_capacity = min(configured_hot, capacity)
    cold_cache_size = config_args.get('replay_cold_cache_size')
    if (
            isinstance(cold_cache_size, bool)
            or not isinstance(cold_cache_size, int)
            or cold_cache_size < 0):
        raise ValueError(
            'config replay_cold_cache_size must be a non-negative integer'
        )
    cold_ratio = _finite_number(
        config_args.get('replay_cold_sample_ratio')
    )
    if cold_ratio is None or not 0.0 <= cold_ratio <= 1.0:
        raise ValueError(
            'config replay_cold_sample_ratio must be in [0, 1]'
        )

    from daxigua_rl.attribution.causal_replay import (
        CAUSAL_SAMPLE_SCHEMA_VERSION,
    )

    return {
        'replay_buffer': {
            'kind': 'dqn-replay-buffer',
            'schema_version': 1,
            'capacity': capacity,
            'hot_capacity': hot_capacity,
            'disk_enabled': hot_capacity < capacity,
            'segment_size': positive_int('replay_segment_size'),
            'cold_cache_size': cold_cache_size,
            'cold_sample_ratio': cold_ratio,
            'cold_cache_refresh_interval': positive_int(
                'replay_cold_cache_refresh_interval'
            ),
        },
        'causal_replay_buffer': {
            'kind': 'causal-replay-buffer',
            'schema_version': 1,
            'sample_schema_version': CAUSAL_SAMPLE_SCHEMA_VERSION,
            'capacity': positive_int('causal_replay_capacity'),
        },
    }


def _load_checkpoint(path, *, config_args=None):
    """可信加载 checkpoint，并只返回门禁需要的轻量摘要。"""

    import torch

    checkpoint_path = Path(path)
    mmap_used = False
    mmap_fallback_reason = None
    try:
        payload = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,
            mmap=True,
        )
        mmap_used = True
    except TypeError as exc:
        mmap_fallback_reason = (
            f'torch.load does not accept mmap: {exc}'
        )
        payload = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,
        )
    except RuntimeError as exc:
        message = str(exc)
        if 'mmap can only be used with files saved with' not in message:
            raise
        mmap_fallback_reason = message
        payload = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,
        )
    if not isinstance(payload, Mapping):
        raise TypeError('checkpoint root must be a mapping')
    state = payload.get('training_state')
    if not isinstance(state, Mapping):
        raise ValueError('checkpoint training_state is missing')

    finite_errors = []
    tensor_count = 0
    tensor_numel = 0

    def scan(value, item_path):
        nonlocal tensor_count, tensor_numel
        if len(finite_errors) >= 25:
            return
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            tensor_numel += int(value.numel())
            if value.numel() and not bool(torch.isfinite(value).all().item()):
                finite_errors.append(f'{item_path}: non-finite tensor')
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                scan(child, f'{item_path}.{key}')
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                scan(child, f'{item_path}[{index}]')
            return
        if isinstance(value, float) and not math.isfinite(value):
            finite_errors.append(f'{item_path}: {value!r}')

    # 主/目标模型和 optimizer 是恢复后继续更新的核心数值状态。replay 可能包含
    # 数万张图，CSV/训练时 fail-fast 已覆盖它，不在本审计中做全量张量扫描。
    for key in ('online_model', 'target_model', 'optimizer'):
        scan(state.get(key), f'$.training_state.{key}')
    for key in ('update_step', 'trainer_update_step', 'env_steps'):
        value = state.get(key)
        if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            finite_errors.append(
                f'$.training_state.{key}: invalid finite number '
                f'{value!r}'
            )

    rng_validation = {
        'passed': False,
        'error': None,
    }
    try:
        from daxigua_rl.training.checkpointing import (
            validate_rng_state,
        )

        rng_validation.update(
            validate_rng_state(payload.get('rng_state'))
        )
        rng_validation['passed'] = True
    except Exception as exc:
        rng_validation['error'] = (
            f'{type(exc).__name__}: {exc}'
        )

    model_optimizer_restore = {
        'passed': False,
        'error': None,
        'online_parameter_count': 0,
        'target_parameter_count': 0,
        'state_key_count': 0,
    }
    restored_online = None
    restored_target = None
    restored_optimizer = None
    try:
        from daxigua_rl.scripts.train_dqn import build_model

        if not isinstance(config_args, Mapping):
            raise ValueError(
                'config args are required for model restore validation'
            )
        online_state = state.get('online_model')
        target_state = state.get('target_model')
        optimizer_state = state.get('optimizer')
        if not isinstance(online_state, Mapping) or not online_state:
            raise ValueError('online_model state must be non-empty')
        if not isinstance(target_state, Mapping) or not target_state:
            raise ValueError('target_model state must be non-empty')
        if set(online_state) != set(target_state):
            raise ValueError(
                'online/target state keys do not match'
            )
        for key in online_state:
            online_value = online_state[key]
            target_value = target_state[key]
            if (
                    not isinstance(online_value, torch.Tensor)
                    or not isinstance(target_value, torch.Tensor)):
                raise TypeError(
                    f'model state {key!r} must contain tensors'
                )
            if (
                    online_value.shape != target_value.shape
                    or online_value.dtype != target_value.dtype):
                raise ValueError(
                    f'online/target shape or dtype mismatch at {key!r}'
                )
        model_args = argparse.Namespace(**dict(config_args))
        restored_online = build_model(model_args).cpu()
        restored_target = build_model(model_args).cpu()
        restored_online.load_state_dict(online_state, strict=True)
        restored_target.load_state_dict(target_state, strict=True)
        if not isinstance(optimizer_state, Mapping):
            raise ValueError('optimizer state must be a mapping')
        learning_rate = _finite_number(
            config_args.get('learning_rate')
        )
        if learning_rate is None or learning_rate <= 0.0:
            raise ValueError('config learning_rate is invalid')
        restored_optimizer = torch.optim.Adam(
            restored_online.parameters(),
            lr=learning_rate,
        )
        restored_optimizer.load_state_dict(optimizer_state)
        model_optimizer_restore.update(
            {
                'passed': True,
                'online_parameter_count': sum(
                    parameter.numel()
                    for parameter in restored_online.parameters()
                ),
                'target_parameter_count': sum(
                    parameter.numel()
                    for parameter in restored_target.parameters()
                ),
                'state_key_count': len(online_state),
            }
        )
    except Exception as exc:
        model_optimizer_restore['error'] = (
            f'{type(exc).__name__}: {exc}'
        )
    finally:
        del restored_optimizer
        del restored_target
        del restored_online

    component_snapshots = payload.get('component_snapshots')
    component_names = (
        sorted(component_snapshots)
        if isinstance(component_snapshots, Mapping)
        else []
    )
    checkpoint_manifest = {
        'passed': False,
        'error': None,
        'schema_version': None,
        'config_fingerprint': None,
        'resume_mutable_fields': [],
    }
    try:
        from daxigua_rl.training.checkpointing import RunManifest

        restored_manifest = RunManifest.from_dict(
            payload.get('run_manifest')
        )
    except Exception as exc:
        checkpoint_manifest['error'] = (
            f'{type(exc).__name__}: {exc}'
        )
    else:
        checkpoint_manifest.update(
            {
                'passed': True,
                'schema_version': restored_manifest.schema_version,
                'config_fingerprint': (
                    restored_manifest.config_fingerprint
                ),
                'resume_mutable_fields': list(
                    restored_manifest.resume_mutable_fields
                ),
            }
        )

    replay_checkpoint = {
        'public_validation': {
            'passed': False,
            'error': 'replay component is missing',
            'state_protocol': None,
            'manifest_present': False,
            'item_count': None,
            'matches_config': False,
        },
    }
    causal_replay_checkpoint = {
        'state_present': False,
        'schema_version': None,
        'items_present': False,
        'total_count': 0,
        'valid_sample_count': 0,
        'invalid_sample_count': 0,
        'sample_errors': [],
        'supervision_kind_counts': {},
        'stratum_counts': {},
        'rule_stratum_counts': {},
        'direction_counts': {},
        'positive_rule_count': 0,
        'negative_rule_count': 0,
        'counterfactual_sample_count': 0,
        'shapley_sample_count': 0,
        'public_restore': {
            'passed': False,
            'error': 'causal replay component is missing',
            'state_protocol': None,
            'manifest_present': False,
            'restored_count': None,
            'matches_config': False,
        },
    }
    expected_replay_manifests = None
    expected_replay_manifest_error = None
    try:
        expected_replay_manifests = _expected_replay_manifests(
            config_args
        )
    except Exception as exc:
        expected_replay_manifest_error = (
            f'{type(exc).__name__}: {exc}'
        )
    if isinstance(component_snapshots, Mapping):
        replay_component = component_snapshots.get('replay_buffer')
        if isinstance(replay_component, Mapping):
            replay_state = replay_component.get('state')
            replay_manifest = replay_component.get('manifest')
            replay_validation = {
                'passed': False,
                'error': None,
                'state_protocol': replay_component.get(
                    'state_protocol'
                ),
                'manifest_present': isinstance(
                    replay_manifest,
                    Mapping,
                ),
                'item_count': None,
                'matches_config': False,
                'expected_manifest': (
                    expected_replay_manifests.get('replay_buffer')
                    if expected_replay_manifests is not None
                    else None
                ),
                'checkpoint_manifest': _json_report_value(
                    replay_manifest
                ),
            }
            try:
                from daxigua_rl.training.replay_buffer import (
                    ReplayBuffer,
                )

                if (
                        replay_component.get('state_protocol')
                        != 'checkpoint_state_dict'):
                    raise ValueError(
                        'replay state_protocol must be '
                        'checkpoint_state_dict'
                    )
                expected_manifest = (
                    expected_replay_manifests['replay_buffer']
                    if expected_replay_manifests is not None
                    else None
                )
                if expected_manifest is None:
                    raise ValueError(
                        'cannot derive replay manifest from config: '
                        f'{expected_replay_manifest_error}'
                    )
                if not isinstance(replay_manifest, Mapping):
                    raise ValueError(
                        'replay checkpoint manifest is missing'
                    )
                manifest_mismatches = {
                    name: {
                        'expected': expected_value,
                        'checkpoint': replay_manifest.get(name),
                    }
                    for name, expected_value in expected_manifest.items()
                    if replay_manifest.get(name) != expected_value
                }
                if manifest_mismatches:
                    replay_validation['config_mismatches'] = (
                        manifest_mismatches
                    )
                    raise ValueError(
                        'replay checkpoint manifest does not match '
                        'config.json'
                    )
                replay_validation['matches_config'] = True
                normalized_replay = (
                    ReplayBuffer.validate_checkpoint_state_dict(
                        replay_state,
                        manifest=replay_manifest,
                    )
                )
                replay_validation.update(
                    {
                        'passed': True,
                        'item_count': len(
                            normalized_replay['items']
                        ),
                        'resume_policy': normalized_replay[
                            'resume_policy'
                        ],
                        'sample_calls': normalized_replay[
                            'sample_calls'
                        ],
                        'source_total_count': normalized_replay[
                            'source_total_count'
                        ],
                        'omitted_cold_count': normalized_replay[
                            'omitted_cold_count'
                        ],
                    }
                )
            except Exception as exc:
                replay_validation['error'] = (
                    f'{type(exc).__name__}: {exc}'
                )
            replay_checkpoint['public_validation'] = (
                replay_validation
            )
            if isinstance(replay_state, Mapping):
                replay_checkpoint.update({
                    key: replay_state.get(key)
                    for key in (
                        'schema_version',
                        'resume_policy',
                        'source_total_count',
                        'omitted_cold_count',
                        'next_segment_index',
                    )
                })
        causal_component = component_snapshots.get(
            'causal_replay_buffer'
        )
        if isinstance(causal_component, Mapping):
            causal_state = causal_component.get('state')
            causal_manifest = causal_component.get('manifest')
            restore_summary = {
                'passed': False,
                'error': None,
                'state_protocol': causal_component.get(
                    'state_protocol'
                ),
                'manifest_present': isinstance(
                    causal_manifest,
                    Mapping,
                ),
                'restored_count': None,
                'matches_config': False,
                'expected_manifest': (
                    expected_replay_manifests.get(
                        'causal_replay_buffer'
                    )
                    if expected_replay_manifests is not None
                    else None
                ),
                'checkpoint_manifest': _json_report_value(
                    causal_manifest
                ),
            }
            try:
                from daxigua_rl.attribution.causal_replay import (
                    CausalReplayBuffer,
                )

                if (
                        causal_component.get('state_protocol')
                        != 'checkpoint_state_dict'):
                    raise ValueError(
                        'causal replay state_protocol must be '
                        'checkpoint_state_dict'
                    )
                if not isinstance(causal_manifest, Mapping):
                    raise ValueError(
                        'causal replay checkpoint manifest is missing'
                    )
                expected_manifest = (
                    expected_replay_manifests[
                        'causal_replay_buffer'
                    ]
                    if expected_replay_manifests is not None
                    else None
                )
                if expected_manifest is None:
                    raise ValueError(
                        'cannot derive causal replay manifest from config: '
                        f'{expected_replay_manifest_error}'
                    )
                manifest_mismatches = {
                    name: {
                        'expected': expected_value,
                        'checkpoint': causal_manifest.get(name),
                    }
                    for name, expected_value in expected_manifest.items()
                    if causal_manifest.get(name) != expected_value
                }
                if manifest_mismatches:
                    restore_summary['config_mismatches'] = (
                        manifest_mismatches
                    )
                    raise ValueError(
                        'causal replay checkpoint manifest does not '
                        'match config.json'
                    )
                restore_summary['matches_config'] = True
                capacity = _nonnegative_count(
                    causal_manifest.get('capacity')
                )
                if capacity is None or capacity <= 0:
                    raise ValueError(
                        'causal replay manifest capacity is invalid'
                    )
                restored_replay = CausalReplayBuffer(
                    capacity=capacity,
                )
                restored_replay.validate_checkpoint_manifest(
                    causal_manifest
                )
                restored_replay.load_checkpoint_state_dict(
                    causal_state
                )
                restore_summary.update(
                    {
                        'passed': True,
                        'restored_count': len(restored_replay),
                    }
                )
            except Exception as exc:
                restore_summary['error'] = (
                    f'{type(exc).__name__}: {exc}'
                )
            finally:
                try:
                    del restored_replay
                except UnboundLocalError:
                    pass
            causal_replay_checkpoint['public_restore'] = (
                restore_summary
            )
            if isinstance(causal_state, Mapping):
                causal_replay_checkpoint['state_present'] = True
                causal_replay_checkpoint['schema_version'] = (
                    causal_state.get('schema_version')
                )
                items = causal_state.get('items')
                if isinstance(items, (list, tuple)):
                    causal_replay_checkpoint['items_present'] = True
                    causal_replay_checkpoint['total_count'] = len(items)
                    kind_counts = Counter()
                    stratum_counts = Counter()
                    rule_stratum_counts = Counter()
                    direction_counts = Counter()
                    sample_errors = []
                    for index, sample in enumerate(items):
                        def sample_field(name):
                            if isinstance(sample, Mapping):
                                return sample.get(name)
                            return getattr(sample, name, None)

                        kind = sample_field('supervision_kind')
                        stratum = sample_field('stratum')
                        direction = sample_field('direction')
                        item_errors = []
                        if kind not in (
                                'rule',
                                'counterfactual',
                                'shapley'):
                            item_errors.append(
                                f'unknown supervision_kind {kind!r}'
                            )
                        if stratum not in (
                                'positive_setup',
                                'negative_blocking',
                                'counterfactual'):
                            item_errors.append(
                                f'unknown stratum {stratum!r}'
                            )
                        if (
                                isinstance(direction, bool)
                                or direction not in (-1, 1)):
                            item_errors.append(
                                f'invalid direction {direction!r}'
                            )
                        for field_name in (
                                'target_margin',
                                'confidence',
                                'target_delta'):
                            value = sample_field(field_name)
                            if (
                                    value is not None
                                    and (
                                        isinstance(value, bool)
                                        or not isinstance(
                                            value,
                                            (int, float),
                                        )
                                        or not math.isfinite(float(value))
                                    )):
                                item_errors.append(
                                    f'non-finite {field_name}={value!r}'
                                )
                        if item_errors:
                            if len(sample_errors) < 25:
                                sample_errors.extend(
                                    f'items[{index}]: {error}'
                                    for error in item_errors[
                                        :25 - len(sample_errors)
                                    ]
                                )
                            continue
                        kind_counts[kind] += 1
                        stratum_counts[stratum] += 1
                        direction_counts[str(direction)] += 1
                        if kind == 'rule':
                            rule_stratum_counts[stratum] += 1
                    invalid_count = (
                        len(items) - sum(kind_counts.values())
                    )
                    causal_replay_checkpoint.update(
                        {
                            'valid_sample_count': sum(
                                kind_counts.values()
                            ),
                            'invalid_sample_count': invalid_count,
                            'sample_errors': sample_errors,
                            'supervision_kind_counts': dict(
                                sorted(kind_counts.items())
                            ),
                            'stratum_counts': dict(
                                sorted(stratum_counts.items())
                            ),
                            'rule_stratum_counts': dict(
                                sorted(rule_stratum_counts.items())
                            ),
                            'direction_counts': dict(
                                sorted(direction_counts.items())
                            ),
                            'positive_rule_count': (
                                rule_stratum_counts['positive_setup']
                            ),
                            'negative_rule_count': (
                                rule_stratum_counts[
                                    'negative_blocking'
                                ]
                            ),
                            'counterfactual_sample_count': (
                                kind_counts['counterfactual']
                            ),
                            'shapley_sample_count': (
                                kind_counts['shapley']
                            ),
                        }
                    )
                else:
                    causal_replay_checkpoint['sample_errors'] = [
                        'state.items must be a list or tuple'
                    ]
    result = {
        'schema_version': payload.get('schema_version'),
        'update_step': state.get('update_step'),
        'trainer_update_step': state.get('trainer_update_step'),
        'env_steps': state.get('env_steps'),
        'has_online_model': isinstance(
            state.get('online_model'),
            Mapping,
        ),
        'has_target_model': isinstance(
            state.get('target_model'),
            Mapping,
        ),
        'has_optimizer': isinstance(state.get('optimizer'), Mapping),
        'has_rng_state': rng_validation['passed'],
        'rng_validation': rng_validation,
        'model_optimizer_restore': model_optimizer_restore,
        'component_names': component_names,
        'checkpoint_manifest': checkpoint_manifest,
        'replay_checkpoint': replay_checkpoint,
        'causal_replay_checkpoint': causal_replay_checkpoint,
        'expected_replay_manifests': expected_replay_manifests,
        'expected_replay_manifest_error': (
            expected_replay_manifest_error
        ),
        'mmap_used': mmap_used,
        'mmap_fallback_reason': mmap_fallback_reason,
        'finite_errors': finite_errors,
        'scanned_tensor_count': tensor_count,
        'scanned_tensor_numel': tensor_numel,
        'shapley': _json_report_value(state.get('shapley')),
    }
    del state
    del component_snapshots
    del payload
    gc.collect()
    return result


def _replay_cold_storage_audit(
        run_dir,
        *,
        config_args,
        replay_checkpoint,
        max_gb):
    """统计 cold segment，不删除疑似旧 generation。"""

    cold_dir_value = config_args.get('replay_cold_dir')
    cold_dir = (
        Path(cold_dir_value)
        if isinstance(cold_dir_value, str) and cold_dir_value
        else Path(run_dir) / 'replay_cold'
    )
    if not cold_dir.is_absolute():
        cold_dir = (PROJECT_ROOT / cold_dir).resolve()
    else:
        cold_dir = cold_dir.resolve()

    segment_pattern = re.compile(r'^segment_(\d+)\.pt$')
    segments = []
    unknown_files = []
    if cold_dir.is_dir():
        for path in sorted(cold_dir.iterdir()):
            if not path.is_file():
                continue
            match = segment_pattern.fullmatch(path.name)
            if match is None:
                unknown_files.append(path.name)
                continue
            segments.append(
                {
                    'index': int(match.group(1)),
                    'name': path.name,
                    'bytes': path.stat().st_size,
                }
            )
    indices = [item['index'] for item in segments]
    total_bytes = sum(item['bytes'] for item in segments)
    total_gb = total_bytes / (1024 ** 3)
    minimum_index = min(indices) if indices else None
    maximum_index = max(indices) if indices else None
    contiguous = (
        not indices
        or (
            len(set(indices)) == len(indices)
            and maximum_index - minimum_index + 1 == len(indices)
        )
    )

    capacity = config_args.get('replay_capacity')
    hot_capacity = config_args.get('hot_replay_capacity')
    segment_size = config_args.get('replay_segment_size')
    max_live_segments = None
    if all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in (capacity, hot_capacity, segment_size)):
        cold_capacity = max(0, capacity - hot_capacity)
        max_live_segments = math.ceil(cold_capacity / segment_size)

    next_segment_index = (
        replay_checkpoint.get('next_segment_index')
        if isinstance(replay_checkpoint, Mapping)
        else None
    )
    suspicious_reasons = []
    if (
            max_live_segments is not None
            and len(segments) > max_live_segments):
        suspicious_reasons.append(
            'segment_count_exceeds_one_live_generation'
        )
    if total_gb > float(max_gb) + 1e-12:
        suspicious_reasons.append('total_bytes_exceed_warning_limit')
    if indices and not contiguous:
        suspicious_reasons.append('segment_indices_have_gaps')
    if (
            indices
            and isinstance(next_segment_index, int)
            and maximum_index >= next_segment_index):
        suspicious_reasons.append(
            'segment_index_not_below_checkpoint_next_index'
        )
    if unknown_files:
        suspicious_reasons.append('unknown_files_in_replay_cold')

    generation_estimate = None
    if max_live_segments:
        generation_estimate = math.ceil(
            len(segments) / max_live_segments
        )
    return {
        'passed': not suspicious_reasons,
        'path': str(cold_dir),
        'exists': cold_dir.is_dir(),
        'segment_count': len(segments),
        'total_bytes': total_bytes,
        'total_gb': total_gb,
        'min_segment_index': minimum_index,
        'max_segment_index': maximum_index,
        'indices_contiguous': contiguous,
        'first_segment_indices': indices[:10],
        'last_segment_indices': indices[-10:],
        'unknown_files': unknown_files,
        'max_live_segments_from_config': max_live_segments,
        'estimated_generation_count_lower_bound': generation_estimate,
        'checkpoint_resume_policy': (
            replay_checkpoint.get('resume_policy')
            if isinstance(replay_checkpoint, Mapping)
            else None
        ),
        'checkpoint_omitted_cold_count': (
            replay_checkpoint.get('omitted_cold_count')
            if isinstance(replay_checkpoint, Mapping)
            else None
        ),
        'checkpoint_next_segment_index': next_segment_index,
        'warning_limit_gb': float(max_gb),
        'suspicious_reasons': suspicious_reasons,
        'interpretation': (
            'hot-only resume intentionally leaves unregistered older cold '
            'segments on disk; this audit reports possible generations but '
            'never deletes them'
        ),
    }


def _attribution_shutdown_audit(payload, expected_worker_count):
    """验证每个 collector worker 都被收口，且所有取消事件汇总守恒。"""

    errors = []
    if not isinstance(payload, Mapping):
        return {
            'passed': False,
            'errors': ['root must be an object'],
            'cancelled_pending_count': None,
        }
    workers = payload.get('workers')
    if not isinstance(workers, list):
        return {
            'passed': False,
            'errors': ['workers must be a list'],
            'cancelled_pending_count': payload.get(
                'cancelled_pending_count'
            ),
        }
    if len(workers) != expected_worker_count:
        errors.append(
            f'worker finalizations={len(workers)} '
            f'expected={expected_worker_count}'
        )

    worker_ids = []
    cancelled_total = 0
    n_step_total = 0
    aggregate_events = {}
    aggregate_reasons = {}
    for index, worker in enumerate(workers):
        if not isinstance(worker, Mapping):
            errors.append(f'workers[{index}] is not an object')
            continue
        worker_id = _nonnegative_count(worker.get('worker_id'))
        if worker_id is None:
            errors.append(f'workers[{index}].worker_id invalid')
            worker_ids.append(f'<invalid-{index}>')
        else:
            worker_ids.append(worker_id)
        cancelled = _nonnegative_count(
            worker.get('cancelled_pending_count')
        )
        n_step = _nonnegative_count(
            worker.get('n_step_flush_emitted')
        )
        if cancelled is None:
            errors.append(
                f'workers[{index}].cancelled_pending_count invalid'
            )
            continue
        if n_step is None:
            errors.append(
                f'workers[{index}].n_step_flush_emitted invalid'
            )
            continue
        event_counts = worker.get('event_type_counts')
        reason_counts = worker.get('resolution_reason_counts')
        if not isinstance(event_counts, Mapping):
            errors.append(f'workers[{index}].event_type_counts invalid')
            event_counts = {}
        if not isinstance(reason_counts, Mapping):
            errors.append(
                f'workers[{index}].resolution_reason_counts invalid'
            )
            reason_counts = {}
        normalized_events = {}
        for name, count in event_counts.items():
            normalized = _nonnegative_count(count)
            if normalized is None:
                errors.append(
                    f'workers[{index}].event_type_counts[{name!r}] '
                    'invalid'
                )
            else:
                normalized_events[str(name)] = normalized
        normalized_reasons = {}
        for name, count in reason_counts.items():
            normalized = _nonnegative_count(count)
            if normalized is None:
                errors.append(
                    f'workers[{index}].resolution_reason_counts'
                    f'[{name!r}] invalid'
                )
            else:
                normalized_reasons[str(name)] = normalized
        if sum(normalized_events.values()) != cancelled:
            errors.append(
                f'workers[{index}] event counts do not match cancelled'
            )
        if sum(normalized_reasons.values()) != cancelled:
            errors.append(
                f'workers[{index}] reason counts do not match cancelled'
            )
        cancelled_total += cancelled
        n_step_total += n_step
        for name, count in normalized_events.items():
            aggregate_events[name] = aggregate_events.get(name, 0) + count
        for name, count in normalized_reasons.items():
            aggregate_reasons[name] = (
                aggregate_reasons.get(name, 0) + count
            )

    if len(set(worker_ids)) != len(worker_ids):
        errors.append('worker_id values are not unique')
    expected_ids = set(range(expected_worker_count))
    if set(worker_ids) != expected_ids:
        errors.append(
            f'worker_id set={sorted(worker_ids, key=str)!r} '
            f'expected={sorted(expected_ids)!r}'
        )
    if (
            _nonnegative_count(payload.get('cancelled_pending_count'))
            != cancelled_total):
        errors.append('top-level cancelled_pending_count mismatch')
    if (
            _nonnegative_count(payload.get('n_step_flush_emitted'))
            != n_step_total):
        errors.append('top-level n_step_flush_emitted mismatch')
    top_events = payload.get('event_type_counts')
    if not isinstance(top_events, Mapping):
        top_events = {}
        errors.append('top-level event_type_counts invalid')
    top_reasons = payload.get('resolution_reason_counts')
    if not isinstance(top_reasons, Mapping):
        top_reasons = {}
        errors.append('top-level resolution_reason_counts invalid')
    if dict(top_events) != aggregate_events:
        errors.append('top-level event_type_counts mismatch')
    if dict(top_reasons) != aggregate_reasons:
        errors.append('top-level resolution_reason_counts mismatch')

    return {
        'passed': not errors,
        'errors': errors,
        'worker_count': len(workers),
        'worker_ids': worker_ids,
        'cancelled_pending_count': cancelled_total,
        'n_step_flush_emitted': n_step_total,
        'resolution_reason_counts': aggregate_reasons,
        'interpretation': (
            'cancelled pending may be non-zero when every item is recorded '
            'by a worker finalization; it is not an untracked live task'
        ),
    }


def _counterfactual_shutdown_audit(payload, ratio_limit):
    """验证物理任务、队列和共享 token reservation 在关闭后全部归零。"""

    errors = []
    if not isinstance(payload, Mapping):
        return {
            'passed': False,
            'errors': ['root must be an object'],
        }
    scheduler = payload.get('scheduler')
    if not isinstance(scheduler, Mapping):
        scheduler = {}
        errors.append('scheduler must be an object')

    zero_fields = {
        'pending_task_count': payload.get('pending_task_count'),
        'candidate_pool_count': payload.get('candidate_pool_count'),
        'scheduler.queued': scheduler.get('queued'),
        'scheduler.inflight': scheduler.get('inflight'),
        'scheduler.tokens_reserved': scheduler.get('tokens_reserved'),
        'scheduler.external_active_reservations': scheduler.get(
            'external_active_reservations'
        ),
    }
    if payload.get('closed') is not True:
        errors.append('coordinator is not closed')
    if payload.get('circuit_open') is not False:
        errors.append(
            f'circuit_open={payload.get("circuit_open")!r}, '
            'expected false'
        )
    active_ids = payload.get('active_task_ids')
    if active_ids not in ([], ()):
        errors.append(f'active_task_ids is not empty: {active_ids!r}')
    for field, value in zero_fields.items():
        if value != 0:
            errors.append(f'{field}={value!r}, expected 0')
    if scheduler.get('token_overrun') != 0:
        errors.append(
            f'scheduler.token_overrun={scheduler.get("token_overrun")!r}'
        )
    if payload.get('hard_budget_respected') is not True:
        errors.append('hard_budget_respected is not true')
    for field in ('actual_token_ratio', 'projected_token_ratio'):
        value = payload.get(field)
        if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) > ratio_limit + 1e-12):
            errors.append(
                f'{field}={value!r} exceeds {ratio_limit}'
            )
    return {
        'passed': not errors,
        'errors': errors,
        'active_task_ids': active_ids,
        **zero_fields,
        'scheduler_failed': scheduler.get('failed'),
        'failure_record_count': payload.get(
            'failure_record_count',
            0,
        ),
        'failure_records_created': (
            payload.get('cumulative', {}).get(
                'failure_records_created',
                0,
            )
            if isinstance(payload.get('cumulative'), Mapping)
            else None
        ),
        'failure_reason_counts': (
            payload.get('cumulative', {}).get(
                'failure_reason_counts',
                {},
            )
            if isinstance(payload.get('cumulative'), Mapping)
            else {}
        ),
        'failure_diagnostic_code_counts': (
            payload.get('cumulative', {}).get(
                'failure_diagnostic_code_counts',
                {},
            )
            if isinstance(payload.get('cumulative'), Mapping)
            else {}
        ),
        'failure_trigger_reason_counts': (
            payload.get('cumulative', {}).get(
                'failure_trigger_reason_counts',
                {},
            )
            if isinstance(payload.get('cumulative'), Mapping)
            else {}
        ),
        'actual_token_ratio': payload.get('actual_token_ratio'),
        'projected_token_ratio': payload.get(
            'projected_token_ratio'
        ),
        'hard_budget_respected': payload.get(
            'hard_budget_respected'
        ),
        'circuit_open': payload.get('circuit_open'),
    }


def _shapley_audit(
        *,
        stage,
        config_args,
        metrics_rows,
        checkpoint_state,
        require_samples,
        max_semantic_rate,
        min_results_for_rate,
        resume_segment_starts=()):
    """使用最终 checkpoint 中关闭后的 Shapley 状态给出阶段化结论。"""

    configured = bool(config_args.get('shapley_enabled'))
    metrics_enabled = all(
        _number(row, 'shapley_enabled') == 1.0
        for row in metrics_rows
    ) if metrics_rows else False
    state = (
        checkpoint_state
        if isinstance(checkpoint_state, Mapping)
        else {}
    )
    cumulative = state.get('cumulative')
    if not isinstance(cumulative, Mapping):
        cumulative = {}

    checkpoint_counts = {
        'observed': state.get(
            'observed_event_count',
            0,
        ),
        'selected': state.get(
            'selected_event_count',
            0,
        ),
        'completed': cumulative.get(
            'results_completed',
            0,
        ),
        'failed': cumulative.get(
            'results_failed',
            0,
        ),
        'terminal_dropped': cumulative.get(
            'selected_terminal_dropped',
            0,
        ),
        'reproduction_passed': cumulative.get(
            'reproduction_passed',
            0,
        ),
        'reproduction_failed': cumulative.get(
            'reproduction_failed',
            0,
        ),
        'numeric_jitter_dropped': cumulative.get(
            'numeric_jitter_dropped',
        ),
        'semantic_divergence_dropped': cumulative.get(
            'semantic_divergence_dropped',
        ),
        'samples': cumulative.get(
            'samples_inserted',
            0,
        ),
    }
    normalized_checkpoint_counts = {
        name: _nonnegative_count(value)
        for name, value in checkpoint_counts.items()
    }
    invalid_counts = [
        name
        for name, value in normalized_checkpoint_counts.items()
        if value is None
    ]
    metric_fields = {
        'observed': 'shapley_events_observed',
        'selected': 'shapley_events_selected',
        'completed': 'shapley_tasks_completed',
        'failed': 'shapley_tasks_failed',
        'terminal_dropped': 'shapley_terminal_dropped',
        'reproduction_passed': 'shapley_reproduction_passed',
        'reproduction_failed': 'shapley_reproduction_failed',
        'numeric_jitter_dropped': (
            'shapley_numeric_jitter_dropped'
        ),
        'semantic_divergence_dropped': (
            'shapley_semantic_divergence_dropped'
        ),
        'samples': 'shapley_samples_inserted',
    }
    aggregated_counts = {
        name: _segmented_cumulative_total(
            _series(metrics_rows, field),
            normalized_checkpoint_counts[name],
            segment_starts=resume_segment_starts,
        )
        for name, field in metric_fields.items()
    }
    observed = aggregated_counts['observed']
    selected = aggregated_counts['selected']
    completed = aggregated_counts['completed']
    failed = aggregated_counts['failed']
    terminal_dropped = aggregated_counts['terminal_dropped']
    reproduction_passed = aggregated_counts[
        'reproduction_passed'
    ]
    reproduction_failed = aggregated_counts[
        'reproduction_failed'
    ]
    numeric_jitter_dropped = aggregated_counts[
        'numeric_jitter_dropped'
    ]
    semantic_divergence_dropped = aggregated_counts[
        'semantic_divergence_dropped'
    ]
    samples = aggregated_counts['samples']
    reproduction_outcomes = _reproduction_outcome_audit(
        prefix='shapley',
        metrics_rows=metrics_rows,
        cumulative=cumulative,
        strict_matches=reproduction_passed,
        results_failed=failed,
        reproduction_failed=reproduction_failed,
        segment_starts=resume_segment_starts,
        max_semantic_rate=max_semantic_rate,
        min_results_for_rate=min_results_for_rate,
    )
    # “runner 结果闭环”和“所有 selected 终态闭环”必须分开表达。
    # terminal_dropped 能解释任务去了哪里，但绝不能冒充有效物理结果，
    # 否则一次全部被预算饿死的运行会被错误放行。
    selected_result_accounting_closed = (
        selected == completed + failed
    )
    selected_terminal_accounting_closed = (
        selected == completed + failed + terminal_dropped
    )
    optimizer_consumed = any(
        _number(row, 'causal_update_applied') == 1.0
        and (_number(row, 'shapley_batch_size') or 0.0) > 0.0
        for row in metrics_rows
    )
    active = state.get('active_task_id')
    pending = state.get('pending_task_ids')
    cleanup_complete = (
        state.get('closed') is True
        and active is None
        and pending in ([], ())
    )

    warnings = []
    if selected == 0:
        if stage == '5k':
            interpretation = (
                'zero Shapley selections are permitted in 5k because the '
                'configured cumulative quota is intentionally 0.05%'
            )
            signal_passed = not require_samples
        elif stage == '10k':
            interpretation = (
                'zero Shapley selections are recorded for calibration; '
                'inspect observed/drop reasons and, if needed, launch an '
                'independent 25k calibration run from update 0'
            )
            warnings.append('shapley_zero_selection')
            signal_passed = not require_samples
        else:
            interpretation = (
                'a formal run cannot finish with zero Shapley selections; '
                'inspect selector eligibility, quota and drop reasons'
            )
            signal_passed = False
    else:
        interpretation = (
            'every selected Shapley task must produce a runner result; '
            'completed work must reproduce, pass its efficiency gate, '
            'insert a sample and enter an optimizer batch'
        )
        signal_passed = (
            completed > 0
            and selected_result_accounting_closed
            and terminal_dropped == 0
            and reproduction_passed > 0
            and reproduction_outcomes[
                'failure_outcomes_fully_accounted'
            ]
            and reproduction_outcomes[
                'non_gate_result_failed'
            ] == 0
            and reproduction_outcomes['semantic_rate_passed']
            and samples > 0
            and optimizer_consumed
        )
        if (
                semantic_divergence_dropped > 0
                and not reproduction_outcomes[
                    'semantic_rate_evaluated'
                ]):
            warnings.append(
                'shapley_reproduction_failure_rate_sample_size'
            )
        if terminal_dropped > 0:
            warnings.append('shapley_selected_terminal_drops')
    if require_samples:
        signal_passed = signal_passed and samples > 0

    passed = (
        configured
        and metrics_enabled
        and cleanup_complete
        and not invalid_counts
        and reproduction_outcomes['valid']
        and reproduction_outcomes['failure_outcomes_fully_accounted']
        and reproduction_outcomes['non_gate_result_failed'] == 0
        and reproduction_outcomes['semantic_rate_passed']
        and selected_terminal_accounting_closed
        and signal_passed
    )
    return {
        'passed': passed,
        'configured': configured,
        'metrics_enabled_all_rows': metrics_enabled,
        'checkpoint_state_present': bool(state),
        'invalid_count_fields': invalid_counts,
        'checkpoint_counts': normalized_checkpoint_counts,
        'aggregation': 'resume-segmented metrics plus final checkpoint',
        'closed': state.get('closed'),
        'active_task_id': active,
        'pending_task_ids': pending,
        'observed': observed,
        'selected': selected,
        'completed': completed,
        'failed': failed,
        'terminal_dropped': terminal_dropped,
        'reproduction_passed': reproduction_passed,
        'reproduction_failed': reproduction_failed,
        'numeric_jitter_dropped': numeric_jitter_dropped,
        'semantic_divergence_dropped': (
            semantic_divergence_dropped
        ),
        'reproduction_outcomes': reproduction_outcomes,
        'samples_inserted': samples,
        'selected_result_accounting_closed': (
            selected_result_accounting_closed
        ),
        'selected_result_accounting_delta': (
            selected - completed - failed
        ),
        'selected_terminal_accounting_closed': (
            selected_terminal_accounting_closed
        ),
        'selected_terminal_accounting_delta': (
            selected - completed - failed - terminal_dropped
        ),
        'optimizer_consumed': optimizer_consumed,
        'max_shapley_batch_size': _max_number(
            metrics_rows,
            'shapley_batch_size',
        ),
        'interpretation': interpretation,
        'warnings': warnings,
    }


def _resolve_stage(stage, expected_updates):
    if stage != 'auto':
        return stage
    if expected_updates is None:
        return 'formal'
    if expected_updates <= 5_000:
        return '5k'
    if expected_updates <= 25_000:
        return '10k'
    return 'formal'


def audit_training_run(
        run_dir,
        *,
        stage='auto',
        expected_total_updates=None,
        baseline_config=None,
        thresholds=None):
    """读取 ``run_dir`` 并返回 JSON-safe 门禁报告，不写任何文件。"""

    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise ReadinessInputError(
            f'run directory does not exist: {run_dir}'
        )
    thresholds = thresholds or ReadinessThresholds()
    if not isinstance(thresholds, ReadinessThresholds):
        raise TypeError('thresholds must be ReadinessThresholds')
    requested_expected_total_updates = expected_total_updates
    if (
            expected_total_updates is not None
            and (
                isinstance(expected_total_updates, bool)
                or int(expected_total_updates) != expected_total_updates
                or expected_total_updates <= 0)):
        raise ValueError(
            'expected_total_updates must be a positive integer'
        )
    if stage not in {'auto', '5k', '10k', 'formal'}:
        raise ValueError(
            'stage must be one of auto, 5k, 10k, formal'
        )

    checks = []
    artifacts = {
        'config': _artifact_info(run_dir / 'config.json'),
        'metrics': _artifact_info(run_dir / 'metrics.csv'),
        'episodes': _artifact_info(run_dir / 'episode_metrics.csv'),
        'attribution_warmup': _artifact_info(
            run_dir / 'attribution_warmup.json'
        ),
        'attribution_shutdown': _artifact_info(
            run_dir / 'attribution_shutdown.json'
        ),
        'counterfactual_shutdown': _artifact_info(
            run_dir / 'counterfactual_shutdown.json'
        ),
        'latest_checkpoint': _artifact_info(
            run_dir / 'checkpoints' / 'latest.pt'
        ),
        'training_curves': _artifact_info(
            run_dir / 'plots' / 'training_curves.png'
        ),
        'reward_curves': _artifact_info(
            run_dir / 'plots' / 'reward_breakdown_curves.png'
        ),
        'structure_learning_curves': _artifact_info(
            run_dir / 'plots' / 'structure_learning_curves.png'
        ),
    }
    missing_or_empty = [
        name
        for name, info in artifacts.items()
        if not info['exists'] or info['bytes'] <= 0
    ]
    _add_check(
        checks,
        'required_artifacts',
        not missing_or_empty,
        missing_or_empty=missing_or_empty,
        artifacts=artifacts,
    )

    failure_latest = run_dir / 'failure_latest.json'
    failure_records = sorted(
        path
        for path in run_dir.glob('failure_*.json')
        if path.name != 'failure_latest.json'
    )
    failure_detail = None
    if failure_latest.is_file():
        try:
            failure_detail = _strict_json_load(failure_latest)
        except (OSError, TypeError, ValueError) as exc:
            failure_detail = {'parse_error': f'{type(exc).__name__}: {exc}'}
    _add_check(
        checks,
        'no_failure_latest',
        not failure_latest.exists(),
        failure_latest=str(failure_latest),
        failure_detail=failure_detail,
        historical_failure_records=[
            str(path) for path in failure_records
        ],
    )
    if failure_records and not failure_latest.exists():
        _add_check(
            checks,
            'historical_failure_records',
            False,
            required=False,
            count=len(failure_records),
            paths=[str(path) for path in failure_records],
        )

    config = {}
    config_error = None
    if artifacts['config']['exists']:
        try:
            config = _strict_json_load(artifacts['config']['path'])
            if not isinstance(config, Mapping):
                raise TypeError('config root must be an object')
        except (OSError, TypeError, ValueError) as exc:
            config_error = f'{type(exc).__name__}: {exc}'
            config = {}
    config_finite_errors = _json_safe_finite_errors(config)
    config_args = (
        config.get('args')
        if isinstance(config.get('args'), Mapping)
        else {}
    )
    config_total = config_args.get('total_updates')
    config_total_valid = (
        isinstance(config_total, int)
        and not isinstance(config_total, bool)
        and config_total > 0
    )
    if expected_total_updates is None:
        if config_total_valid:
            expected_total_updates = config_total
    expected_matches_config = (
        config_total_valid
        and expected_total_updates == config_total
    )
    inferred_stage = _resolve_stage(
        'auto',
        config_total if config_total_valid else None,
    )
    stage_matches_config = (
        stage == 'auto' or stage == inferred_stage
    )
    # 所有放宽策略只信任 config total 推导结果；显式声明仅用于一致性检查。
    resolved_stage = inferred_stage
    _add_check(
        checks,
        'config_json',
        (
            config_error is None
            and not config_finite_errors
            and bool(config_args)
            and expected_total_updates is not None
            and expected_matches_config
        ),
        parse_error=config_error,
        finite_errors=config_finite_errors,
        configured_total_updates=config_total,
        expected_total_updates=expected_total_updates,
        stage=resolved_stage,
    )
    _add_check(
        checks,
        'declared_stage_matches_config',
        stage_matches_config,
        declared_stage=stage,
        inferred_stage=inferred_stage,
        configured_total_updates=config_total,
        interpretation=(
            'stage-specific relaxations always use the stage inferred '
            'from config total_updates'
        ),
    )
    _add_check(
        checks,
        'expected_total_updates_matches_config',
        expected_matches_config,
        configured_total_updates=config_total,
        requested_expected_total_updates=(
            requested_expected_total_updates
        ),
        effective_expected_total_updates=expected_total_updates,
        diagnostic_override_used=(
            requested_expected_total_updates is not None
        ),
        unsafe_override=(
            requested_expected_total_updates is not None
            and requested_expected_total_updates != config_total
        ),
        interpretation=(
            'a diagnostic override never authorizes a partial run; ready '
            'requires exact equality with config.json'
        ),
    )

    config_manifest_audit = {
        'passed': False,
        'error': None,
        'manifest_fingerprint': None,
        'recorded_training_fingerprint': None,
        'args_fingerprint': None,
        'resume_mutable_fields': [],
    }
    try:
        from daxigua_rl.training.checkpointing import (
            RunManifest,
            config_fingerprint,
        )

        config_manifest = RunManifest.from_dict(
            config.get('run_manifest')
        )
        recorded_fingerprints = config.get('fingerprints')
        if not isinstance(recorded_fingerprints, Mapping):
            raise ValueError('config fingerprints must be an object')
        recorded_training_fingerprint = (
            recorded_fingerprints.get('training_config')
        )
        args_fingerprint = config_fingerprint(
            config_args,
            mutable_fields=config_manifest.resume_mutable_fields,
        )
        config_manifest_audit.update(
            {
                'manifest_fingerprint': (
                    config_manifest.config_fingerprint
                ),
                'recorded_training_fingerprint': (
                    recorded_training_fingerprint
                ),
                'args_fingerprint': args_fingerprint,
                'resume_mutable_fields': list(
                    config_manifest.resume_mutable_fields
                ),
            }
        )
        config_manifest_audit['passed'] = (
            isinstance(recorded_training_fingerprint, str)
            and recorded_training_fingerprint
            == config_manifest.config_fingerprint
            == args_fingerprint
        )
        if not config_manifest_audit['passed']:
            config_manifest_audit['error'] = (
                'config fingerprint fields do not agree'
            )
    except Exception as exc:
        config_manifest_audit['error'] = (
            f'{type(exc).__name__}: {exc}'
        )
    _add_check(
        checks,
        'config_run_manifest_fingerprints',
        config_manifest_audit['passed'],
        **{
            key: value
            for key, value in config_manifest_audit.items()
            if key != 'passed'
        },
    )

    baseline_audit = _baseline_semantic_audit(
        config_args,
        baseline_config,
    )
    _add_check(
        checks,
        'baseline_training_semantics_match',
        baseline_audit['passed'],
        **{
            key: value
            for key, value in baseline_audit.items()
            if key != 'passed'
        },
    )
    _add_check(
        checks,
        'full_causal_config_enabled',
        (
            config_args.get('counterfactual_enabled') is True
            and config_args.get('shapley_enabled') is True
            and isinstance(config_args.get('lambda_rule'), (int, float))
            and not isinstance(config_args.get('lambda_rule'), bool)
            and float(config_args.get('lambda_rule')) > 0.0
            and isinstance(config_args.get('lambda_cf'), (int, float))
            and not isinstance(config_args.get('lambda_cf'), bool)
            and float(config_args.get('lambda_cf')) > 0.0
        ),
        counterfactual_enabled=config_args.get(
            'counterfactual_enabled'
        ),
        shapley_enabled=config_args.get('shapley_enabled'),
        lambda_rule=config_args.get('lambda_rule'),
        lambda_cf=config_args.get('lambda_cf'),
    )

    attribution_warmup = {}
    attribution_warmup_error = None
    if artifacts['attribution_warmup']['exists']:
        try:
            attribution_warmup = _strict_json_load(
                artifacts['attribution_warmup']['path']
            )
            if not isinstance(attribution_warmup, Mapping):
                raise TypeError(
                    'attribution warmup root must be an object'
                )
        except (OSError, TypeError, ValueError) as exc:
            attribution_warmup_error = (
                f'{type(exc).__name__}: {exc}'
            )
            attribution_warmup = {}
    attribution_warmup_finite_errors = _json_safe_finite_errors(
        attribution_warmup
    )
    attribution_warmup_audit = _attribution_warmup_audit(
        attribution_warmup,
        thresholds,
    )
    _add_check(
        checks,
        'attribution_warmup_health',
        (
            attribution_warmup_error is None
            and not attribution_warmup_finite_errors
            and attribution_warmup_audit['passed']
        ),
        parse_error=attribution_warmup_error,
        finite_errors=attribution_warmup_finite_errors,
        **{
            key: value
            for key, value in attribution_warmup_audit.items()
            if key != 'passed'
        },
    )

    metrics_fields = ()
    metrics_rows = []
    metrics_error = None
    if artifacts['metrics']['exists']:
        try:
            metrics_fields, metrics_rows = _read_csv(
                artifacts['metrics']['path']
            )
        except (OSError, TypeError, ValueError, csv.Error) as exc:
            metrics_error = f'{type(exc).__name__}: {exc}'
    missing_metric_fields = sorted(
        REQUIRED_METRIC_FIELDS.difference(metrics_fields)
    )
    metrics_cell_errors = _validate_csv_cells(
        metrics_rows,
        metrics_fields,
        text_fields=METRIC_TEXT_FIELDS,
        json_fields=METRIC_JSON_FIELDS,
    )
    _add_check(
        checks,
        'metrics_schema_and_finite_values',
        (
            metrics_error is None
            and bool(metrics_rows)
            and not missing_metric_fields
            and not metrics_cell_errors
        ),
        row_count=len(metrics_rows),
        parse_error=metrics_error,
        missing_fields=missing_metric_fields,
        cell_errors=metrics_cell_errors,
    )

    metric_updates = _series(metrics_rows, 'update_step')
    metric_env_steps = _series(metrics_rows, 'env_steps')
    resume_sidecars = _resume_sidecar_audit(
        run_dir,
        metric_updates,
    )
    resume_segment_starts = tuple(
        resume_sidecars['segment_start_indices']
    )
    _add_check(
        checks,
        'resume_sidecars_valid',
        resume_sidecars['passed'],
        **{
            key: value
            for key, value in resume_sidecars.items()
            if key != 'passed'
        },
    )
    last_update = metric_updates[-1] if metric_updates else None
    last_env_steps = metric_env_steps[-1] if metric_env_steps else None
    progress_passed = (
        expected_total_updates is not None
        and last_update == float(expected_total_updates)
        and len(metric_updates) == len(metrics_rows)
        and len(metric_env_steps) == len(metrics_rows)
        and _strictly_increasing(metric_updates)
        and _strictly_increasing(metric_env_steps)
    )
    _add_check(
        checks,
        'training_progress_complete_and_monotonic',
        progress_passed,
        expected_total_updates=expected_total_updates,
        first_update=metric_updates[0] if metric_updates else None,
        last_update=last_update,
        first_env_steps=(
            metric_env_steps[0] if metric_env_steps else None
        ),
        last_env_steps=last_env_steps,
        updates_strictly_increasing=_strictly_increasing(
            metric_updates
        ),
        env_steps_strictly_increasing=_strictly_increasing(
            metric_env_steps
        ),
    )

    td_updated = any(
        _number(row, 'td_loss') is not None
        and _number(row, 'td_loss') >= 0.0
        for row in metrics_rows
    )
    rule_updated = any(
        _number(row, 'causal_update_applied') == 1.0
        and (_number(row, 'rule_batch_size') or 0.0) > 0.0
        for row in metrics_rows
    )
    cf_updated = any(
        _number(row, 'causal_update_applied') == 1.0
        and (_number(row, 'counterfactual_batch_size') or 0.0) > 0.0
        for row in metrics_rows
    )
    _add_check(
        checks,
        'td_rule_counterfactual_updates',
        td_updated and rule_updated and cf_updated,
        td_updated=td_updated,
        rule_updated=rule_updated,
        counterfactual_updated=cf_updated,
        max_rule_batch_size=_max_number(
            metrics_rows,
            'rule_batch_size',
        ),
        max_counterfactual_batch_size=_max_number(
            metrics_rows,
            'counterfactual_batch_size',
        ),
    )

    # V2 的结构头只在实际执行动作上学习一步结构变化。仅有字段存在并不能证明
    # 链路贯通：必须在同一个 optimizer step 中看到有效维度、非零原始 loss，
    # 以及乘上正式 lambda 后的非零加权项。逐行检查还会阻止计数越界或日志中的
    # weighted loss 与配置不一致。
    lambda_structural = _finite_number(
        config_args.get('lambda_structural')
    )
    structural_rows = []
    structural_invalid_rows = []
    structural_optimizer_rows = []
    for row_index, row in enumerate(metrics_rows, start=1):
        values = {
            field: _number(row, field)
            for field in STRUCTURAL_SUPERVISION_METRIC_FIELDS
        }
        missing = sorted(
            field
            for field, value in values.items()
            if value is None
        )
        reasons = []
        if missing:
            reasons.append(f'missing_or_nonfinite={missing!r}')
        else:
            loss = values['structural_loss']
            weighted_loss = values['weighted_structural_loss']
            valid_count = values['structural_valid_count']
            sample_count = values['structural_sample_count']
            mean_abs_error = values[
                'structural_mean_abs_error'
            ]
            valid_integer = _nonnegative_count(valid_count)
            sample_integer = _nonnegative_count(sample_count)
            if loss < 0.0:
                reasons.append('structural_loss_negative')
            if weighted_loss < 0.0:
                reasons.append('weighted_structural_loss_negative')
            if mean_abs_error < 0.0:
                reasons.append('structural_mean_abs_error_negative')
            if valid_integer is None:
                reasons.append('structural_valid_count_not_count')
            if sample_integer is None:
                reasons.append('structural_sample_count_not_count')
            if (
                    valid_integer is not None
                    and sample_integer is not None
                    and valid_integer > 6 * sample_integer):
                reasons.append(
                    'structural_valid_count_exceeds_six_per_sample'
                )
            if (
                    lambda_structural is not None
                    and not math.isclose(
                        weighted_loss,
                        lambda_structural * loss,
                        rel_tol=1e-6,
                        abs_tol=1e-9,
                    )):
                reasons.append(
                    'weighted_structural_loss_lambda_mismatch'
                )
            if (
                    not reasons
                    # 第六维仅凭 PhysicsResult 即可有效；要求平均至少五维，
                    # 才能证明依赖相邻完整 StateAnalysis 的前五维不曾全部
                    # 失效，避免“只有 chain/terminal 标签”冒充完整结构监督。
                    and valid_integer >= 5 * sample_integer
                    and sample_integer > 0
                    and loss > 0.0
                    and weighted_loss > 0.0
                    and mean_abs_error > 0.0):
                structural_optimizer_rows.append(row_index)
        structural_rows.append(values)
        if reasons:
            structural_invalid_rows.append(
                {
                    'row': row_index,
                    'update_step': _number(row, 'update_step'),
                    'reasons': reasons,
                }
            )
    structural_config_enabled = (
        lambda_structural is not None
        and lambda_structural > 0.0
    )
    structural_gate_passed = (
        structural_config_enabled
        and len(structural_rows) == len(metrics_rows)
        and not structural_invalid_rows
        and bool(structural_optimizer_rows)
    )
    _add_check(
        checks,
        'structural_supervision_targets_reached_optimizer',
        structural_gate_passed,
        lambda_structural=lambda_structural,
        config_enabled=structural_config_enabled,
        optimizer_evidence_rows=structural_optimizer_rows,
        optimizer_evidence_updates=[
            _number(metrics_rows[index - 1], 'update_step')
            for index in structural_optimizer_rows
        ],
        invalid_rows=structural_invalid_rows,
        max_valid_count=_max_number(
            metrics_rows,
            'structural_valid_count',
        ),
        max_sample_count=_max_number(
            metrics_rows,
            'structural_sample_count',
        ),
        max_structural_loss=_max_number(
            metrics_rows,
            'structural_loss',
        ),
        max_weighted_structural_loss=_max_number(
            metrics_rows,
            'weighted_structural_loss',
        ),
        minimum_mean_valid_dimensions_for_evidence=5.0,
        interpretation=(
            'a non-zero weighted structural loss with a valid target '
            'on the same logged update proves that the auxiliary term '
            'was part of that optimizer objective; at least five valid '
            'dimensions per sampled target are required so the always-'
            'available chain/terminal dimension cannot pass alone'
        ),
    )

    # 集中 actor 的统计是进程段累计值，resume 后允许按 sidecar 划分的新段归零。
    # 异步采集可能已预取下一批，因此不能要求它与当前 metrics 行的 collect_steps
    # 一一对齐；但累计统计自身必须闭合。正式证据至少要出现一次 greedy 请求和
    # 实际微批，同时每行都应满足 requests / batches == mean_batch_size，且最大
    # 批量不超过配置。
    centralized_actor_enabled = (
        config_args.get('centralized_actor_inference') is True
    )
    configured_actor_batch_size = _nonnegative_count(
        config_args.get('actor_batch_size')
    )
    actor_rows = []
    actor_invalid_rows = []
    actor_activity_rows = []
    actor_counter_fields = (
        'actor_inference_requests',
        'actor_inference_batches',
        'actor_inference_max_batch',
    )
    for row_index, row in enumerate(metrics_rows, start=1):
        values = {
            field: _number(row, field)
            for field in CENTRAL_ACTOR_METRIC_FIELDS
        }
        reasons = []
        if any(value is None for value in values.values()):
            reasons.append('missing_or_nonfinite_actor_metric')
        else:
            normalized_counts = {
                field: _nonnegative_count(values[field])
                for field in actor_counter_fields
            }
            if any(
                    value is None
                    for value in normalized_counts.values()):
                reasons.append('actor_counter_not_nonnegative_integer')
            requests = normalized_counts[
                'actor_inference_requests'
            ]
            batches = normalized_counts['actor_inference_batches']
            maximum = normalized_counts[
                'actor_inference_max_batch'
            ]
            mean = values['actor_inference_mean_batch_size']
            seconds = values['actor_inference_seconds']
            if mean < 0.0:
                reasons.append('actor_mean_batch_size_negative')
            if seconds < 0.0:
                reasons.append('actor_inference_seconds_negative')
            if requests is not None and batches is not None:
                if batches > requests:
                    reasons.append('actor_batches_exceed_requests')
                if requests == 0:
                    if (
                            batches != 0
                            or maximum not in {0, None}
                            or mean != 0.0
                            or seconds != 0.0):
                        reasons.append(
                            'zero_requests_have_nonzero_activity'
                        )
                elif batches == 0:
                    reasons.append(
                        'positive_requests_without_batches'
                    )
                else:
                    expected_mean = requests / batches
                    if not math.isclose(
                            mean,
                            expected_mean,
                            rel_tol=1e-6,
                            abs_tol=1e-9):
                        reasons.append(
                            'actor_mean_batch_size_mismatch'
                        )
                    if maximum is None or maximum <= 0:
                        reasons.append('actor_max_batch_not_positive')
                    elif not 1.0 <= mean <= maximum:
                        reasons.append(
                            'actor_mean_outside_one_to_max'
                        )
                    if (
                            configured_actor_batch_size is not None
                            and maximum is not None
                            and maximum
                            > configured_actor_batch_size):
                        reasons.append(
                            'actor_max_batch_exceeds_config'
                        )
                    if seconds <= 0.0:
                        reasons.append(
                            'actor_activity_without_positive_seconds'
                        )
                    if not reasons:
                        actor_activity_rows.append(row_index)
        actor_rows.append(values)
        if reasons:
            actor_invalid_rows.append(
                {
                    'row': row_index,
                    'update_step': _number(row, 'update_step'),
                    'reasons': reasons,
                }
            )

    actor_reset_fields = (
        'actor_inference_requests',
        'actor_inference_batches',
        'actor_inference_max_batch',
        'actor_inference_seconds',
    )
    actor_unexpected_resets = {}
    for field in actor_reset_fields:
        values = tuple(
            _number(row, field)
            for row in metrics_rows
        )
        actor_unexpected_resets[field] = [
            index
            for index, (left, right) in enumerate(
                zip(values, values[1:]),
                start=1,
            )
            if (
                left is not None
                and right is not None
                and right < left
                and index not in resume_segment_starts
            )
        ]
    actor_gate_passed = (
        centralized_actor_enabled
        and configured_actor_batch_size is not None
        and configured_actor_batch_size > 0
        and len(actor_rows) == len(metrics_rows)
        and not actor_invalid_rows
        and bool(actor_activity_rows)
        and not any(actor_unexpected_resets.values())
    )
    _add_check(
        checks,
        'centralized_actor_inference_activity',
        actor_gate_passed,
        centralized_actor_inference=centralized_actor_enabled,
        configured_actor_batch_size=configured_actor_batch_size,
        activity_rows=actor_activity_rows,
        activity_updates=[
            _number(metrics_rows[index - 1], 'update_step')
            for index in actor_activity_rows
        ],
        invalid_rows=actor_invalid_rows,
        unexpected_reset_indices=actor_unexpected_resets,
        resume_segment_start_indices=list(
            resume_segment_starts
        ),
        max_requests=_max_number(
            metrics_rows,
            'actor_inference_requests',
        ),
        max_batches=_max_number(
            metrics_rows,
            'actor_inference_batches',
        ),
        max_observed_batch=_max_number(
            metrics_rows,
            'actor_inference_max_batch',
        ),
        interpretation=(
            'positive centralized actor requests are direct evidence '
            'that at least one non-random (greedy) action used the '
            'main-process micro-batch service; counters are cumulative '
            'within a process segment and may include a prefetched next '
            'rollout, so they are not matched to per-row collect_steps'
        ),
    )

    value_magnitude_fields = (
        'mean_q',
        'mean_target',
        'mean_abs_td_error',
    )
    value_magnitude_streaks = {
        field: _max_consecutive(
            _series(metrics_rows, field),
            lambda value: (
                abs(value)
                > thresholds.max_q_target_td_magnitude
            ),
        )
        for field in value_magnitude_fields
    }
    _add_check(
        checks,
        'q_target_td_error_magnitude',
        (
            all(
                len(_series(metrics_rows, field))
                == len(metrics_rows)
                for field in value_magnitude_fields
            )
            and max(value_magnitude_streaks.values(), default=0)
            < thresholds.consecutive_window_limit
        ),
        maximum_absolute_values={
            field: max(
                (
                    abs(value)
                    for value in _series(metrics_rows, field)
                ),
                default=0.0,
            )
            for field in value_magnitude_fields
        },
        consecutive_violations=value_magnitude_streaks,
        magnitude_threshold=thresholds.max_q_target_td_magnitude,
        consecutive_window_limit=thresholds.consecutive_window_limit,
    )

    max_positive = _max_number(
        metrics_rows,
        'causal_replay_positive_count',
    )
    max_negative = _max_number(
        metrics_rows,
        'causal_replay_negative_count',
    )
    max_rule = _max_number(metrics_rows, 'causal_replay_rule_count')
    max_cf = _max_number(metrics_rows, 'causal_replay_cf_count')

    hard_flags = _series(
        metrics_rows,
        'counterfactual_hard_budget_respected',
    )
    actual_ratios = _series(
        metrics_rows,
        'counterfactual_actual_token_ratio',
    )
    projected_ratios = _series(
        metrics_rows,
        'counterfactual_projected_token_ratio',
    )
    budget_passed = (
        len(hard_flags) == len(metrics_rows)
        and all(value == 1.0 for value in hard_flags)
        and len(actual_ratios) == len(metrics_rows)
        and len(projected_ratios) == len(metrics_rows)
        and max(actual_ratios, default=0.0)
        <= thresholds.max_hard_budget_ratio + 1e-12
        and max(projected_ratios, default=0.0)
        <= thresholds.max_hard_budget_ratio + 1e-12
    )
    _add_check(
        checks,
        'counterfactual_budget_all_windows',
        budget_passed,
        rows=len(metrics_rows),
        hard_true_rows=sum(value == 1.0 for value in hard_flags),
        max_actual_token_ratio=max(actual_ratios, default=0.0),
        max_projected_token_ratio=max(
            projected_ratios,
            default=0.0,
        ),
        limit=thresholds.max_hard_budget_ratio,
    )

    pending_tasks = _series(
        metrics_rows,
        'counterfactual_pending_tasks',
    )
    candidate_capacities = _series(
        metrics_rows,
        'counterfactual_candidate_pool_capacity',
    )
    candidate_counts = _series(
        metrics_rows,
        'counterfactual_candidate_pool_count',
    )
    admission_slots_used = _series(
        metrics_rows,
        'counterfactual_admission_slots_used',
    )
    admission_slots_available = _series(
        metrics_rows,
        'counterfactual_admission_slots_available',
    )
    candidate_dispatch_admitted = _series(
        metrics_rows,
        'counterfactual_candidate_dispatch_admitted',
    )
    candidate_dispatch_attempts = _series(
        metrics_rows,
        'counterfactual_candidate_dispatch_attempts',
    )
    queue_capacity = config_args.get(
        'counterfactual_queue_capacity'
    )
    valid_queue_capacity = (
        isinstance(queue_capacity, int)
        and not isinstance(queue_capacity, bool)
        and queue_capacity > 0
    )
    pending_saturation_flags = tuple(
        valid_queue_capacity and value >= queue_capacity
        for value in pending_tasks
    )
    candidate_saturation_flags = tuple(
        capacity > 0.0 and count >= capacity
        for count, capacity in zip(
            candidate_counts,
            candidate_capacities,
        )
    )
    pending_saturation_streak = _max_consecutive(
        pending_saturation_flags,
        bool,
    )
    candidate_saturation_streak = _max_consecutive(
        candidate_saturation_flags,
        bool,
    )
    dispatch_stall_flags = tuple(
        (
            index > 0
            and index not in resume_segment_starts
            and admission_slots_available[index] > 0.0
            and candidate_counts[index] > 0.0
            and candidate_dispatch_attempts[index]
            == candidate_dispatch_attempts[index - 1]
        )
        for index in range(min(
            len(admission_slots_available),
            len(candidate_counts),
            len(candidate_dispatch_attempts),
        ))
    )
    dispatch_stall_streak = _max_consecutive(
        dispatch_stall_flags,
        bool,
    )
    dispatch_admitted_reset_indices = tuple(
        index
        for index, (left, right) in enumerate(
            zip(
                candidate_dispatch_admitted,
                candidate_dispatch_admitted[1:],
            ),
            start=1,
        )
        if right < left
    )
    dispatch_attempt_reset_indices = tuple(
        index
        for index, (left, right) in enumerate(
            zip(
                candidate_dispatch_attempts,
                candidate_dispatch_attempts[1:],
            ),
            start=1,
        )
        if right < left
    )
    unexpected_dispatch_reset_indices = tuple(sorted(set(
        dispatch_admitted_reset_indices
        + dispatch_attempt_reset_indices
    ).difference(resume_segment_starts)))
    saturation_data_valid = (
        valid_queue_capacity
        and len(pending_tasks) == len(metrics_rows)
        and len(candidate_capacities) == len(metrics_rows)
        and len(candidate_counts) == len(metrics_rows)
        and len(admission_slots_used) == len(metrics_rows)
        and len(admission_slots_available) == len(metrics_rows)
        and len(candidate_dispatch_attempts) == len(metrics_rows)
        and len(candidate_dispatch_admitted) == len(metrics_rows)
        and all(value >= 0.0 for value in pending_tasks)
        and all(value >= 0.0 for value in admission_slots_used)
        and all(
            value >= 0.0
            for value in admission_slots_available
        )
        and all(
            value >= 0.0
            for value in candidate_dispatch_attempts
        )
        and all(
            value >= 0.0
            for value in candidate_dispatch_admitted
        )
        and all(value > 0.0 for value in candidate_capacities)
        and all(
            0.0 <= count <= capacity
            for count, capacity in zip(
                candidate_counts,
                candidate_capacities,
            )
        )
        and not unexpected_dispatch_reset_indices
    )
    saturation_hard_passed = (
        saturation_data_valid
        and pending_saturation_streak
        < thresholds.consecutive_window_limit
        and dispatch_stall_streak
        < thresholds.consecutive_window_limit
    )
    _add_check(
        checks,
        'counterfactual_execution_pending_saturation',
        saturation_hard_passed,
        queue_capacity=queue_capacity,
        max_pending_tasks=max(pending_tasks, default=0.0),
        max_candidate_pool_count=max(
            candidate_counts,
            default=0.0,
        ),
        max_candidate_pool_capacity=max(
            candidate_capacities,
            default=0.0,
        ),
        pending_saturated_windows=sum(pending_saturation_flags),
        candidate_saturated_windows=sum(
            candidate_saturation_flags
        ),
        pending_max_consecutive=pending_saturation_streak,
        candidate_max_consecutive=candidate_saturation_streak,
        max_admission_slots_used=max(
            admission_slots_used,
            default=0.0,
        ),
        max_admission_slots_available=max(
            admission_slots_available,
            default=0.0,
        ),
        max_candidate_dispatch_attempts=max(
            candidate_dispatch_attempts,
            default=0.0,
        ),
        dispatch_stall_windows=sum(dispatch_stall_flags),
        dispatch_stall_max_consecutive=dispatch_stall_streak,
        resume_segment_start_indices=list(
            resume_segment_starts
        ),
        dispatch_counter_reset_count=len(
            dispatch_admitted_reset_indices
        ),
        dispatch_attempt_counter_reset_count=len(
            dispatch_attempt_reset_indices
        ),
        unexpected_dispatch_reset_indices=list(
            unexpected_dispatch_reset_indices
        ),
        consecutive_window_limit=thresholds.consecutive_window_limit,
        candidate_pool_interpretation=(
            'the candidate pool is an intentional cross-window top-K '
            'reservoir; being full is informational, not a failure'
        ),
    )
    if (
            saturation_hard_passed
            and any(pending_saturation_flags)):
        _add_check(
            checks,
            'counterfactual_transient_pending_saturation',
            False,
            required=False,
            pending_saturated_windows=sum(
                pending_saturation_flags
            ),
            candidate_saturated_windows=sum(
                candidate_saturation_flags
            ),
            interpretation=(
                'execution pending was transiently saturated but did not '
                'reach the hard consecutive limit'
            ),
        )
    if (
            saturation_hard_passed
            and any(dispatch_stall_flags)):
        _add_check(
            checks,
            'counterfactual_transient_dispatch_stall',
            False,
            required=False,
            stalled_windows=sum(dispatch_stall_flags),
            maximum_consecutive=dispatch_stall_streak,
            interpretation=(
                'candidate work and free admission slots coexisted without '
                'a dispatch attempt, but not for the hard consecutive limit'
            ),
        )

    snapshot_failures = _series(
        metrics_rows,
        'collect_counterfactual_snapshot_failures',
    )
    snapshot_failure_total = sum(snapshot_failures)
    _add_check(
        checks,
        'counterfactual_snapshot_failures',
        (
            len(snapshot_failures) == len(metrics_rows)
            and snapshot_failure_total
            <= thresholds.max_snapshot_failures
        ),
        total=snapshot_failure_total,
        maximum_allowed=thresholds.max_snapshot_failures,
    )

    materialization_fields = (
        'checkpoint_step_materialization',
        'checkpoint_extra_materialization',
    )
    materializations = [
        (row_index, field, (row.get(field) or '').strip())
        for row_index, row in enumerate(metrics_rows, start=2)
        for field in materialization_fields
        if (row.get(field) or '').strip()
    ]
    invalid_materializations = [
        {
            'row': row_index,
            'field': field,
            'value': value,
        }
        for row_index, field, value in materializations
        if value not in {'hardlink', 'copy'}
    ]
    copy_materializations = [
        {
            'row': row_index,
            'field': field,
        }
        for row_index, field, value in materializations
        if value == 'copy'
    ]
    hardlink_materialization_count = sum(
        value == 'hardlink'
        for _row_index, _field, value in materializations
    )
    max_copies_in_one_row = max(
        (
            sum(
                (row.get(field) or '').strip() == 'copy'
                for field in materialization_fields
            )
            for row in metrics_rows
        ),
        default=0,
    )
    max_checkpoint_bytes = _max_number(
        metrics_rows,
        'checkpoint_bytes',
    )
    save_interval = config_args.get('save_interval')
    periodic_materialization_expected = (
        isinstance(save_interval, int)
        and not isinstance(save_interval, bool)
        and save_interval > 0
        and expected_total_updates is not None
        and expected_total_updates >= save_interval
    )
    _add_check(
        checks,
        'checkpoint_materialization_values',
        (
            not invalid_materializations
            and (
                not periodic_materialization_expected
                or bool(materializations)
            )
        ),
        observed_count=len(materializations),
        periodic_materialization_expected=(
            periodic_materialization_expected
        ),
        save_interval=save_interval,
        hardlink_count=hardlink_materialization_count,
        copy_count=len(copy_materializations),
        invalid=invalid_materializations,
    )
    if copy_materializations:
        _add_check(
            checks,
            'checkpoint_copy_materialization',
            False,
            required=False,
            occurrences=copy_materializations,
            checkpoint_bytes_max=max_checkpoint_bytes,
            estimated_peak_checkpoint_multiplier=(
                1 + max_copies_in_one_row
            ),
            interpretation=(
                'same-volume hardlink was unavailable and the atomic '
                'fallback copied checkpoint bytes; reserve the reported '
                'higher transient disk peak before a longer run'
            ),
        )

    degraded_rates = _series(
        metrics_rows,
        'collect_state_analysis_degraded_rate',
    )
    degraded_consecutive = _max_consecutive(
        degraded_rates,
        lambda value: (
            value > thresholds.max_state_analysis_degraded_rate
        ),
    )
    _add_check(
        checks,
        'state_analysis_degraded_rate',
        (
            len(degraded_rates) == len(metrics_rows)
            and degraded_consecutive
            < thresholds.consecutive_window_limit
        ),
        max_rate=max(degraded_rates, default=0.0),
        threshold=thresholds.max_state_analysis_degraded_rate,
        max_consecutive_violations=degraded_consecutive,
        consecutive_window_limit=thresholds.consecutive_window_limit,
    )

    shaping_values = _series(
        metrics_rows,
        'collect_p95_abs_potential_shaping_reward',
    )
    shaping_consecutive = _max_consecutive(
        shaping_values,
        lambda value: value > thresholds.max_shaping_p95,
    )
    _add_check(
        checks,
        'potential_shaping_p95',
        (
            len(shaping_values) == len(metrics_rows)
            and shaping_consecutive
            < thresholds.consecutive_window_limit
        ),
        maximum=max(shaping_values, default=0.0),
        threshold=thresholds.max_shaping_p95,
        max_consecutive_violations=shaping_consecutive,
        consecutive_window_limit=thresholds.consecutive_window_limit,
    )

    episode_fields = ()
    episode_rows = []
    episode_error = None
    if artifacts['episodes']['exists']:
        try:
            episode_fields, episode_rows = _read_csv(
                artifacts['episodes']['path']
            )
        except (OSError, TypeError, ValueError, csv.Error) as exc:
            episode_error = f'{type(exc).__name__}: {exc}'
    missing_episode_fields = sorted(
        REQUIRED_EPISODE_FIELDS.difference(episode_fields)
    )
    episode_cell_errors = _validate_csv_cells(
        episode_rows,
        episode_fields,
        text_fields=EPISODE_TEXT_FIELDS,
    )
    episode_indices = _series(episode_rows, 'episode_index')
    episode_updates = _series(episode_rows, 'update_step')
    episode_env_steps = _series(episode_rows, 'env_steps')
    episode_end_flags_valid = all(
        {
            _number(row, 'terminated'),
            _number(row, 'truncated'),
        }.issubset({0.0, 1.0})
        and (
            (_number(row, 'terminated') or 0.0)
            + (_number(row, 'truncated') or 0.0)
            == 1.0
        )
        for row in episode_rows
    )
    episode_schema_passed = (
        episode_error is None
        and bool(episode_rows)
        and not missing_episode_fields
        and not episode_cell_errors
        and len(episode_indices) == len(episode_rows)
        and _strictly_increasing(episode_indices)
        and len(episode_updates) == len(episode_rows)
        and _non_decreasing(episode_updates)
        and len(episode_env_steps) == len(episode_rows)
        and _non_decreasing(episode_env_steps)
        and episode_end_flags_valid
    )
    _add_check(
        checks,
        'episode_metrics_schema_finite_and_monotonic',
        episode_schema_passed,
        row_count=len(episode_rows),
        parse_error=episode_error,
        missing_fields=missing_episode_fields,
        cell_errors=episode_cell_errors,
        indices_strictly_increasing=_strictly_increasing(
            episode_indices
        ),
        updates_non_decreasing=_non_decreasing(episode_updates),
        env_steps_non_decreasing=_non_decreasing(
            episode_env_steps
        ),
        end_flags_valid=episode_end_flags_valid,
    )

    truncated_count = sum(
        _number(row, 'truncated') == 1.0
        for row in episode_rows
    )
    truncated_rate = (
        truncated_count / len(episode_rows)
        if episode_rows
        else None
    )
    truncated_evaluated = (
        len(episode_rows)
        >= thresholds.min_episodes_for_truncated_rate
    )
    if truncated_evaluated:
        _add_check(
            checks,
            'episode_truncated_rate',
            truncated_rate <= thresholds.max_truncated_rate + 1e-12,
            evaluated=True,
            episodes=len(episode_rows),
            truncated_count=truncated_count,
            truncated_rate=truncated_rate,
            threshold=thresholds.max_truncated_rate,
        )
    else:
        _add_check(
            checks,
            'episode_truncated_rate_sample_size',
            False,
            required=False,
            evaluated=False,
            episodes=len(episode_rows),
            minimum=thresholds.min_episodes_for_truncated_rate,
            truncated_count=truncated_count,
            truncated_rate=truncated_rate,
        )

    attribution_shutdown = {}
    attribution_error = None
    if artifacts['attribution_shutdown']['exists']:
        try:
            attribution_shutdown = _strict_json_load(
                artifacts['attribution_shutdown']['path']
            )
        except (OSError, TypeError, ValueError) as exc:
            attribution_error = f'{type(exc).__name__}: {exc}'
    expected_workers = config_args.get('num_envs', 1)
    if (
            not isinstance(expected_workers, int)
            or isinstance(expected_workers, bool)
            or expected_workers < 1):
        expected_workers = 1
    attribution_audit = _attribution_shutdown_audit(
        attribution_shutdown,
        expected_workers,
    )
    attribution_finite_errors = _json_safe_finite_errors(
        attribution_shutdown
    )
    _add_check(
        checks,
        'attribution_worker_finalization',
        (
            attribution_error is None
            and not attribution_finite_errors
            and attribution_audit['passed']
        ),
        parse_error=attribution_error,
        finite_errors=attribution_finite_errors,
        **{
            key: value
            for key, value in attribution_audit.items()
            if key != 'passed'
        },
    )

    cf_shutdown = {}
    cf_shutdown_error = None
    if artifacts['counterfactual_shutdown']['exists']:
        try:
            cf_shutdown = _strict_json_load(
                artifacts['counterfactual_shutdown']['path']
            )
            if not isinstance(cf_shutdown, Mapping):
                raise TypeError(
                    'counterfactual shutdown root must be an object'
                )
        except (OSError, TypeError, ValueError) as exc:
            cf_shutdown_error = f'{type(exc).__name__}: {exc}'
            cf_shutdown = {}
    cf_shutdown_audit = _counterfactual_shutdown_audit(
        cf_shutdown,
        thresholds.max_hard_budget_ratio,
    )
    cf_shutdown_finite_errors = _json_safe_finite_errors(cf_shutdown)
    _add_check(
        checks,
        'counterfactual_shutdown_drained',
        (
            cf_shutdown_error is None
            and not cf_shutdown_finite_errors
            and cf_shutdown_audit['passed']
        ),
        parse_error=cf_shutdown_error,
        finite_errors=cf_shutdown_finite_errors,
        **{
            key: value
            for key, value in cf_shutdown_audit.items()
            if key != 'passed'
        },
    )

    cumulative = (
        cf_shutdown.get('cumulative')
        if isinstance(cf_shutdown.get('cumulative'), Mapping)
        else {}
    )
    cf_count_values = {
        'results_completed': _nonnegative_count(
            cumulative.get('results_completed')
        ),
        'results_failed': _nonnegative_count(
            cumulative.get('results_failed')
        ),
        'reproduction_passed': _nonnegative_count(
            cumulative.get('reproduction_passed')
        ),
        'reproduction_failed': _nonnegative_count(
            cumulative.get('reproduction_failed')
        ),
        'samples_inserted': _nonnegative_count(
            cumulative.get('samples_inserted')
        ),
        'candidate_offers': _nonnegative_count(
            cumulative.get('candidate_offers')
        ),
        'candidate_close_dropped': _nonnegative_count(
            cumulative.get('candidate_close_dropped')
        ),
    }
    cf_invalid_counts = [
        name
        for name, value in cf_count_values.items()
        if value is None
    ]
    cf_completed = _segmented_cumulative_total(
        _series(metrics_rows, 'counterfactual_results_completed'),
        cf_count_values['results_completed'],
        segment_starts=resume_segment_starts,
    )
    cf_results_failed = _segmented_cumulative_total(
        _series(metrics_rows, 'counterfactual_results_failed'),
        cf_count_values['results_failed'],
        segment_starts=resume_segment_starts,
    )
    cf_reproduced = _segmented_cumulative_total(
        _series(
            metrics_rows,
            'counterfactual_reproduction_passed',
        ),
        cf_count_values['reproduction_passed'],
        segment_starts=resume_segment_starts,
    )
    cf_reproduction_failed = _segmented_cumulative_total(
        _series(
            metrics_rows,
            'counterfactual_reproduction_failed',
        ),
        cf_count_values['reproduction_failed'],
        segment_starts=resume_segment_starts,
    )
    cf_samples = _segmented_cumulative_total(
        _series(metrics_rows, 'counterfactual_samples_inserted'),
        cf_count_values['samples_inserted'],
        segment_starts=resume_segment_starts,
    )
    cf_candidate_offers = _segmented_cumulative_total(
        _series(metrics_rows, 'counterfactual_candidate_offers'),
        cf_count_values['candidate_offers'],
        segment_starts=resume_segment_starts,
    )
    cf_candidate_close_dropped = _segmented_cumulative_total(
        _series(
            metrics_rows,
            'counterfactual_candidate_close_dropped',
        ),
        cf_count_values['candidate_close_dropped'],
        segment_starts=resume_segment_starts,
    )
    cf_reproduction_outcomes = _reproduction_outcome_audit(
        prefix='counterfactual',
        metrics_rows=metrics_rows,
        cumulative=cumulative,
        strict_matches=cf_reproduced,
        results_failed=cf_results_failed,
        reproduction_failed=cf_reproduction_failed,
        segment_starts=resume_segment_starts,
        max_semantic_rate=(
            thresholds.max_cf_reproduction_failure_rate
        ),
        min_results_for_rate=(
            thresholds.min_cf_results_for_failure_rate
        ),
    )
    _add_check(
        checks,
        'counterfactual_completed_reproduced_samples',
        (
            not cf_invalid_counts
            and cf_completed > 0
            and cf_reproduced > 0
            and cf_samples > 0
        ),
        results_completed=cf_completed,
        reproduction_passed=cf_reproduced,
        reproduction_failed=cf_reproduction_failed,
        samples_inserted=cf_samples,
        invalid_count_fields=cf_invalid_counts,
        aggregation='resume-segmented metrics plus shutdown',
    )
    _add_check(
        checks,
        'counterfactual_reproduction_outcome_accounting',
        (
            cf_reproduction_outcomes['valid']
            and cf_reproduction_outcomes[
                'failure_outcomes_fully_accounted'
            ]
        ),
        **cf_reproduction_outcomes,
    )
    metric_cf_failures = _series(
        metrics_rows,
        'counterfactual_results_failed',
    )
    metric_failure_reason_counts = tuple(
        _metric_counter(row, 'counterfactual_failure_reasons')
        for row in metrics_rows
    )
    metric_failure_diagnostic_counts = tuple(
        _metric_counter(
            row,
            'counterfactual_failure_diagnostic_codes',
        )
        for row in metrics_rows
    )
    metric_failure_trigger_counts = tuple(
        _metric_counter(
            row,
            'counterfactual_failure_trigger_reasons',
        )
        for row in metrics_rows
    )
    metric_drop_reason_counts = tuple(
        _metric_counter(row, 'counterfactual_drop_reasons')
        for row in metrics_rows
    )
    unclassified_metric_rows = []
    for index, (
            row,
            reason_counts,
    ) in enumerate(zip(metrics_rows, metric_failure_reason_counts)):
        failed_count = _number(
            row,
            'counterfactual_results_failed',
        )
        if (
                failed_count is None
                or reason_counts is None
                or failed_count > sum(reason_counts.values())):
            unclassified_metric_rows.append(index)

    shutdown_failure_reasons = (
        cumulative.get('failure_reason_counts')
        if isinstance(
            cumulative.get('failure_reason_counts'),
            Mapping,
        )
        else {}
    )
    normalized_shutdown_failure_reasons = {}
    shutdown_failure_reason_counts_valid = True
    for reason, value in shutdown_failure_reasons.items():
        count = _nonnegative_count(value)
        if (
                not isinstance(reason, str)
                or not reason
                or count is None):
            shutdown_failure_reason_counts_valid = False
            break
        normalized_shutdown_failure_reasons[reason] = count

    shutdown_drop_reasons = (
        cumulative.get('drop_reason_counts')
        if isinstance(cumulative.get('drop_reason_counts'), Mapping)
        else {}
    )
    normalized_shutdown_drop_reasons = {}
    shutdown_drop_reason_counts_valid = True
    for reason, value in shutdown_drop_reasons.items():
        count = _nonnegative_count(value)
        if (
                not isinstance(reason, str)
                or not reason
                or count is None):
            shutdown_drop_reason_counts_valid = False
            break
        normalized_shutdown_drop_reasons[reason] = count
    infrastructure_reason_names = frozenset(
        {
            'executor_submit_failure',
            'runner_failure',
            'pending_without_result',
            'close_pending_without_result',
            'orphan_result',
            'duplicate_result',
            'label_conversion_failure',
            'causal_replay_push_failure',
        }
    )
    infrastructure_failure_counts = {}
    for reason in sorted(infrastructure_reason_names):
        observed = []
        for reason_counts in metric_drop_reason_counts:
            if reason_counts is not None:
                observed.append(reason_counts.get(reason, 0))
        shutdown_count = _nonnegative_count(
            normalized_shutdown_drop_reasons.get(reason, 0)
        )
        if shutdown_count is not None:
            observed.append(shutdown_count)
        infrastructure_failure_counts[reason] = max(
            observed,
            default=0,
        )

    shutdown_scheduler_failed = _nonnegative_count(
        cf_shutdown.get('scheduler', {}).get('failed')
        if isinstance(cf_shutdown.get('scheduler'), Mapping)
        else None
    )
    shutdown_result_failed = cf_count_values['results_failed']
    expected_shutdown_scheduler_failed = None
    if shutdown_result_failed is not None:
        expected_shutdown_scheduler_failed = (
            shutdown_result_failed
            + (
                _nonnegative_count(
                    normalized_shutdown_drop_reasons.get(
                        'executor_submit_failure',
                        0,
                    )
                )
                or 0
            )
            + (
                _nonnegative_count(
                    normalized_shutdown_drop_reasons.get(
                        'runner_failure',
                        0,
                    )
                )
                or 0
            )
        )
    shutdown_failed_results_classified = (
        shutdown_result_failed is not None
        and shutdown_failure_reason_counts_valid
        and shutdown_result_failed
        <= sum(normalized_shutdown_failure_reasons.values())
    )
    _add_check(
        checks,
        'counterfactual_failures_classified_and_infrastructure_clean',
        (
            len(metric_cf_failures) == len(metrics_rows)
            and all(
                counts is not None
                for counts in metric_failure_reason_counts
            )
            and all(
                counts is not None
                for counts in metric_failure_diagnostic_counts
            )
            and all(
                counts is not None
                for counts in metric_failure_trigger_counts
            )
            and all(
                counts is not None
                for counts in metric_drop_reason_counts
            )
            and shutdown_drop_reason_counts_valid
            and not unclassified_metric_rows
            and shutdown_failed_results_classified
            and shutdown_scheduler_failed
            == expected_shutdown_scheduler_failed
            and not any(infrastructure_failure_counts.values())
        ),
        metrics_max_results_failed=max(
            metric_cf_failures,
            default=0.0,
        ),
        shutdown_results_failed=cf_results_failed,
        unclassified_metric_row_indices=unclassified_metric_rows,
        invalid_failure_reason_metric_rows=[
            index
            for index, counts in enumerate(
                metric_failure_reason_counts
            )
            if counts is None
        ],
        invalid_failure_diagnostic_metric_rows=[
            index
            for index, counts in enumerate(
                metric_failure_diagnostic_counts
            )
            if counts is None
        ],
        invalid_failure_trigger_metric_rows=[
            index
            for index, counts in enumerate(
                metric_failure_trigger_counts
            )
            if counts is None
        ],
        shutdown_failure_reason_counts=(
            normalized_shutdown_failure_reasons
        ),
        shutdown_failed_results_classified=(
            shutdown_failed_results_classified
        ),
        shutdown_scheduler_failed=shutdown_scheduler_failed,
        expected_shutdown_scheduler_failed=(
            expected_shutdown_scheduler_failed
        ),
        infrastructure_failure_counts=(
            infrastructure_failure_counts
        ),
        drop_reason_counts=normalized_shutdown_drop_reasons,
        interpretation=(
            'a physics result rejected by the original-action gate is safe '
            'only when its reason is classified and its aggregate '
            'reproduction failure rate remains within the separate limit; '
            'executor, transport, accounting, conversion and replay '
            'failures always block scale-up'
        ),
    )

    candidate_close_drop_rate = (
        cf_candidate_close_dropped / cf_candidate_offers
        if cf_candidate_offers > 0
        else 0.0
    )
    if cf_candidate_close_dropped > 0:
        _add_check(
            checks,
            'counterfactual_candidate_close_dropped',
            False,
            required=False,
            candidate_offers=cf_candidate_offers,
            candidate_close_dropped=cf_candidate_close_dropped,
            drop_rate=candidate_close_drop_rate,
            interpretation=(
                'the cross-window top-K reservoir was cleared during '
                'graceful close; this is reported but is not a failure'
            ),
        )
    cf_reproduction_total = cf_reproduction_outcomes[
        'semantic_rate_denominator'
    ]
    cf_failure_rate = cf_reproduction_outcomes[
        'semantic_divergence_rate'
    ]
    cf_semantic_divergence = cf_reproduction_outcomes[
        'semantic_divergence_dropped'
    ]
    if cf_reproduction_outcomes['semantic_rate_evaluated']:
        _add_check(
            checks,
            'counterfactual_reproduction_failure_rate',
            cf_reproduction_outcomes['semantic_rate_passed'],
            evaluated=True,
            total=cf_reproduction_total,
            strict_matches=cf_reproduced,
            numeric_jitter_dropped=cf_reproduction_outcomes[
                'numeric_jitter_dropped'
            ],
            failures=cf_semantic_divergence,
            rate=cf_failure_rate,
            threshold=thresholds.max_cf_reproduction_failure_rate,
            interpretation=(
                'only semantic divergence consumes the physical '
                'reproduction failure-rate allowance; numeric jitter is '
                'reported and dropped without consuming that allowance'
            ),
        )
    elif cf_semantic_divergence > 0:
        _add_check(
            checks,
            'counterfactual_reproduction_failure_rate_sample_size',
            False,
            required=False,
            evaluated=False,
            total=cf_reproduction_total,
            minimum=thresholds.min_cf_results_for_failure_rate,
            strict_matches=cf_reproduced,
            numeric_jitter_dropped=cf_reproduction_outcomes[
                'numeric_jitter_dropped'
            ],
            failures=cf_semantic_divergence,
            rate=cf_failure_rate,
        )

    checkpoint_summary = {}
    checkpoint_error = None
    checkpoint_loaded = None
    if artifacts['latest_checkpoint']['exists']:
        try:
            checkpoint_loaded = _load_checkpoint(
                artifacts['latest_checkpoint']['path'],
                config_args=config_args,
            )
        except Exception as exc:
            checkpoint_error = f'{type(exc).__name__}: {exc}'
    if checkpoint_loaded is not None:
        checkpoint_summary = {
            key: value
            for key, value in checkpoint_loaded.items()
            if key != 'shapley'
        }
    required_components = {
        'replay_buffer',
        'causal_replay_buffer',
    }
    causal_checkpoint = (
        checkpoint_loaded.get('causal_replay_checkpoint', {})
        if checkpoint_loaded is not None
        else {}
    )
    causal_checkpoint_complete = (
        causal_checkpoint.get('state_present') is True
        and causal_checkpoint.get('schema_version') == 1
        and causal_checkpoint.get('items_present') is True
        and causal_checkpoint.get('invalid_sample_count') == 0
        and causal_checkpoint.get(
            'public_restore',
            {},
        ).get('passed') is True
        and causal_checkpoint.get(
            'public_restore',
            {},
        ).get('matches_config') is True
    )
    replay_checkpoint_validation = (
        checkpoint_loaded.get('replay_checkpoint', {}).get(
            'public_validation',
            {},
        )
        if checkpoint_loaded is not None
        else {}
    )
    replay_checkpoint_complete = (
        replay_checkpoint_validation.get('passed') is True
        and replay_checkpoint_validation.get(
            'matches_config'
        ) is True
        and (
            replay_checkpoint_validation.get('item_count') or 0
        ) > 0
    )
    checkpoint_manifest_audit = (
        checkpoint_loaded.get('checkpoint_manifest', {})
        if checkpoint_loaded is not None
        else {}
    )
    checkpoint_config_fingerprint_matches = (
        config_manifest_audit['passed']
        and checkpoint_manifest_audit.get('passed') is True
        and checkpoint_manifest_audit.get('config_fingerprint')
        == config_manifest_audit.get('manifest_fingerprint')
    )
    checkpoint_passed = (
        checkpoint_error is None
        and checkpoint_loaded is not None
        and checkpoint_loaded['schema_version'] == 1
        and checkpoint_loaded['update_step'] == expected_total_updates
        and (
            checkpoint_loaded['trainer_update_step']
            == expected_total_updates
        )
        and checkpoint_loaded['env_steps'] == last_env_steps
        and checkpoint_loaded['has_online_model']
        and checkpoint_loaded['has_target_model']
        and checkpoint_loaded['has_optimizer']
        and checkpoint_loaded['has_rng_state']
        and checkpoint_loaded.get(
            'model_optimizer_restore',
            {},
        ).get('passed') is True
        and checkpoint_config_fingerprint_matches
        and required_components.issubset(
            checkpoint_loaded['component_names']
        )
        and replay_checkpoint_complete
        and causal_checkpoint_complete
        and not checkpoint_loaded['finite_errors']
    )
    _add_check(
        checks,
        'latest_checkpoint_complete_and_finite',
        checkpoint_passed,
        parse_error=checkpoint_error,
        expected_update_step=expected_total_updates,
        expected_env_steps=last_env_steps,
        required_components=sorted(required_components),
        **checkpoint_summary,
    )
    _add_check(
        checks,
        'config_checkpoint_manifest_fingerprints',
        checkpoint_config_fingerprint_matches,
        config_manifest_fingerprint=config_manifest_audit.get(
            'manifest_fingerprint'
        ),
        config_recorded_training_fingerprint=(
            config_manifest_audit.get(
                'recorded_training_fingerprint'
            )
        ),
        config_args_fingerprint=config_manifest_audit.get(
            'args_fingerprint'
        ),
        checkpoint_manifest=checkpoint_manifest_audit,
    )
    _add_check(
        checks,
        'checkpoint_replay_manifests_match_config',
        (
            replay_checkpoint_validation.get(
                'matches_config'
            ) is True
            and causal_checkpoint.get(
                'public_restore',
                {},
            ).get('matches_config') is True
        ),
        expected_manifests=(
            checkpoint_loaded.get('expected_replay_manifests')
            if checkpoint_loaded is not None
            else None
        ),
        expected_manifest_error=(
            checkpoint_loaded.get(
                'expected_replay_manifest_error'
            )
            if checkpoint_loaded is not None
            else checkpoint_error
        ),
        replay_validation=replay_checkpoint_validation,
        causal_replay_validation=causal_checkpoint.get(
            'public_restore',
            {},
        ),
    )

    _add_check(
        checks,
        'causal_replay_positive_negative_rule_samples',
        (
            causal_checkpoint_complete
            and causal_checkpoint.get('positive_rule_count', 0) > 0
            and causal_checkpoint.get('negative_rule_count', 0) > 0
            and causal_checkpoint.get(
                'counterfactual_sample_count',
                0,
            ) > 0
            and max_positive > 0
            and max_negative > 0
            and max_rule > 0
            and max_cf > 0
        ),
        checkpoint_exact_counts=causal_checkpoint,
        metrics_max_counts={
            'positive_stratum': max_positive,
            'negative_stratum': max_negative,
            'rule_supervision': max_rule,
            'counterfactual_supervision': max_cf,
        },
        evidence_note=(
            'checkpoint items provide the exact supervision-kind by '
            'stratum cross-tab; CSV maxima are independent corroboration'
        ),
    )

    replay_cold = _replay_cold_storage_audit(
        run_dir,
        config_args=config_args,
        replay_checkpoint=(
            checkpoint_loaded.get('replay_checkpoint', {})
            if checkpoint_loaded is not None
            else {}
        ),
        max_gb=thresholds.max_replay_cold_gb,
    )
    _add_check(
        checks,
        'replay_cold_storage_growth',
        replay_cold['passed'],
        required=False,
        **{
            key: value
            for key, value in replay_cold.items()
            if key != 'passed'
        },
    )

    shapley = _shapley_audit(
        stage=resolved_stage,
        config_args=config_args,
        metrics_rows=metrics_rows,
        checkpoint_state=(
            checkpoint_loaded.get('shapley')
            if checkpoint_loaded is not None
            else None
        ),
        require_samples=thresholds.require_shapley_samples,
        max_semantic_rate=(
            thresholds.max_cf_reproduction_failure_rate
        ),
        min_results_for_rate=(
            thresholds.min_cf_results_for_failure_rate
        ),
        resume_segment_starts=resume_segment_starts,
    )
    _add_check(
        checks,
        'shapley_stage_evidence_and_shutdown',
        shapley['passed'],
        stage=resolved_stage,
        **{
            key: value
            for key, value in shapley.items()
            if key != 'passed'
        },
    )
    for warning in shapley['warnings']:
        _add_check(
            checks,
            warning,
            False,
            required=False,
            stage=resolved_stage,
            interpretation=shapley['interpretation'],
            observed=shapley['observed'],
            selected=shapley['selected'],
            samples=shapley['samples_inserted'],
        )

    required_failures = [
        check['name']
        for check in checks
        if check['required'] and not check['passed']
    ]
    warnings = [
        check['name']
        for check in checks
        if not check['required'] and not check['passed']
    ]
    ready = not required_failures
    payload = {
        'schema_version': READINESS_SCHEMA_VERSION,
        'created_at': datetime.now().astimezone().isoformat(
            timespec='seconds'
        ),
        'run_dir': str(run_dir),
        'stage': resolved_stage,
        'ready': ready,
        'exit_code': 0 if ready else 1,
        'required_failures': required_failures,
        'warnings': warnings,
        'thresholds': asdict(thresholds),
        'config': {
            'configured_total_updates': config_total,
            'expected_total_updates': expected_total_updates,
            'num_envs': config_args.get('num_envs'),
            'counterfactual_enabled': config_args.get(
                'counterfactual_enabled'
            ),
            'shapley_enabled': config_args.get('shapley_enabled'),
            'git': config.get('git'),
            'fingerprints': config.get('fingerprints'),
            'manifest_audit': config_manifest_audit,
            'baseline_semantic_audit': baseline_audit,
            'diagnostic_expected_override': {
                'requested': requested_expected_total_updates,
                'matches_config': expected_matches_config,
                'unsafe': (
                    requested_expected_total_updates is not None
                    and requested_expected_total_updates != config_total
                ),
            },
        },
        'attribution_warmup': attribution_warmup_audit,
        'metrics': {
            'row_count': len(metrics_rows),
            'first_update': (
                metric_updates[0] if metric_updates else None
            ),
            'last_update': last_update,
            'last_env_steps': last_env_steps,
            'max_snapshot_failures_in_window': max(
                snapshot_failures,
                default=0.0,
            ),
            'snapshot_failure_total': snapshot_failure_total,
            'max_state_analysis_degraded_rate': max(
                degraded_rates,
                default=0.0,
            ),
            'max_shaping_p95': max(shaping_values, default=0.0),
            'max_actual_token_ratio': max(
                actual_ratios,
                default=0.0,
            ),
            'max_projected_token_ratio': max(
                projected_ratios,
                default=0.0,
            ),
            'causal_replay_positive_max': max_positive,
            'causal_replay_negative_max': max_negative,
            'causal_replay_rule_max': max_rule,
            'causal_replay_cf_max': max_cf,
            'checkpoint_materialization': {
                'observed_count': len(materializations),
                'hardlink_count': hardlink_materialization_count,
                'copy_count': len(copy_materializations),
                'copy_occurrences': copy_materializations,
                'estimated_peak_checkpoint_multiplier': (
                    1 + max_copies_in_one_row
                    if copy_materializations
                    else 1
                ),
            },
        },
        'episodes': {
            'row_count': len(episode_rows),
            'truncated_count': truncated_count,
            'truncated_rate': truncated_rate,
            'threshold_evaluated': truncated_evaluated,
        },
        'counterfactual': {
            'shutdown': cf_shutdown_audit,
            'results_completed': cf_completed,
            'results_failed': cf_results_failed,
            'reproduction_passed': cf_reproduced,
            'reproduction_failed': cf_reproduction_failed,
            'numeric_jitter_dropped': cf_reproduction_outcomes[
                'numeric_jitter_dropped'
            ],
            'semantic_divergence_dropped': (
                cf_semantic_divergence
            ),
            'unknown_failed': cf_reproduction_outcomes[
                'unknown_failed'
            ],
            'non_gate_result_failed': cf_reproduction_outcomes[
                'non_gate_result_failed'
            ],
            'numeric_jitter_error_maxima': (
                cf_reproduction_outcomes[
                    'numeric_jitter_error_maxima'
                ]
            ),
            'reproduction_failure_rate': cf_failure_rate,
            'samples_inserted': cf_samples,
            'candidate_offers': cf_candidate_offers,
            'candidate_close_dropped': (
                cf_candidate_close_dropped
            ),
            'candidate_close_drop_rate': (
                candidate_close_drop_rate
            ),
        },
        'shapley': shapley,
        'attribution_shutdown': attribution_audit,
        'checkpoint': checkpoint_summary,
        'replay_cold': replay_cold,
        'artifacts': artifacts,
        'resource_monitor_coverage': {
            'automatically_audited': False,
            'host_rss_threshold_audited': False,
            'gpu_memory_threshold_audited': False,
            'ready_includes_resource_health': False,
            'required_external_evidence': [
                'resource monitor summary and process_metrics.csv',
                'host peak RSS / available memory / swap trend',
                'GPU peak memory and Xid/OOM evidence',
            ],
            'interpretation': (
                'ready only covers artifacts inside the run directory; '
                'host RSS and GPU thresholds require the sidecar resource '
                'monitor and human review'
            ),
        },
        'checks': checks,
    }
    return _json_report_value(payload)


def write_json_atomic(path, payload, *, protected_run_dir=None):
    """原子写 JSON；拒绝写入被审计 run，保持工具的只读边界。"""

    destination = Path(path).expanduser().resolve()
    if protected_run_dir is not None:
        protected = Path(protected_run_dir).expanduser().resolve()
        if destination == protected or destination.is_relative_to(protected):
            raise ReadinessInputError(
                '--output must be outside the audited run directory'
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.',
        suffix='.tmp',
        dir=destination.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file_obj:
            json.dump(
                payload,
                file_obj,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file_obj.write('\n')
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, destination)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def _thresholds_from_args(args):
    return ReadinessThresholds(
        max_hard_budget_ratio=args.max_hard_budget_ratio,
        max_snapshot_failures=args.max_snapshot_failures,
        max_state_analysis_degraded_rate=(
            args.max_state_analysis_degraded_rate
        ),
        max_shaping_p95=args.max_shaping_p95,
        max_truncated_rate=args.max_truncated_rate,
        min_episodes_for_truncated_rate=(
            args.min_episodes_for_truncated_rate
        ),
        consecutive_window_limit=args.consecutive_window_limit,
        max_cf_reproduction_failure_rate=(
            args.max_cf_reproduction_failure_rate
        ),
        min_cf_results_for_failure_rate=(
            args.min_cf_results_for_failure_rate
        ),
        max_q_target_td_magnitude=(
            args.max_q_target_td_magnitude
        ),
        max_replay_cold_gb=args.max_replay_cold_gb,
        require_shapley_samples=args.require_shapley_samples,
    )


def _emit_payload(payload, *, output, run_dir):
    if output in (None, '-'):
        print(json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ))
        return None
    destination = write_json_atomic(
        output,
        payload,
        protected_run_dir=run_dir,
    )
    print(
        f'readiness_output={destination} '
        f'ready={str(bool(payload.get("ready"))).lower()}',
        flush=True,
    )
    return destination


def main(argv=None):
    """CLI：0=通过，1=门禁未通过，2=输入/审计器错误。"""

    args = parse_args(argv)
    try:
        thresholds = _thresholds_from_args(args)
        payload = audit_training_run(
            args.run_dir,
            stage=args.stage,
            expected_total_updates=args.expected_total_updates,
            baseline_config=args.baseline_config,
            thresholds=thresholds,
        )
        _emit_payload(
            payload,
            output=args.output,
            run_dir=args.run_dir,
        )
        return int(payload['exit_code'])
    except (
            OSError,
            TypeError,
            ValueError,
            ReadinessInputError,
    ) as exc:
        payload = {
            'schema_version': READINESS_SCHEMA_VERSION,
            'created_at': datetime.now().astimezone().isoformat(
                timespec='seconds'
            ),
            'run_dir': str(
                Path(args.run_dir).expanduser().resolve()
            ),
            'ready': False,
            'exit_code': 2,
            'error': f'{type(exc).__name__}: {exc}',
        }
        try:
            _emit_payload(
                payload,
                output=args.output,
                run_dir=args.run_dir,
            )
        except (OSError, ReadinessInputError):
            print(json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
