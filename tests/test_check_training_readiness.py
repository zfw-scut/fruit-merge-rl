"""训练产物只读门禁工具的合成 run 回归测试。"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from tools.check_training_readiness import (
    ReadinessThresholds,
    audit_training_run,
    main,
)
from daxigua_rl.attribution.causal_replay import (
    CausalReplayBuffer,
    CausalSample,
    graph_schema_fingerprint,
)
from daxigua_rl.graph.tensor import GraphTensor
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.scripts.train_dqn import (
    parse_args as parse_training_args,
    validate_args as validate_training_args,
)
from daxigua_rl.training.checkpointing import (
    capture_rng_state,
    create_run_manifest,
)
from daxigua_rl.training.identity import TransitionKey
from daxigua_rl.training.replay_buffer import ReplayBuffer
from daxigua_rl.training.tensor_transition import TensorTransition


METRIC_FIELDS = (
    'update_step',
    'env_steps',
    'td_loss',
    'structural_loss',
    'weighted_structural_loss',
    'structural_valid_count',
    'structural_sample_count',
    'structural_mean_abs_error',
    'mean_q',
    'mean_target',
    'mean_abs_td_error',
    'actor_inference_requests',
    'actor_inference_batches',
    'actor_inference_mean_batch_size',
    'actor_inference_max_batch',
    'actor_inference_seconds',
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
    'counterfactual_numeric_jitter_dropped',
    'counterfactual_semantic_divergence_dropped',
    'counterfactual_numeric_jitter_max_merge_event_position_error',
    'counterfactual_numeric_jitter_max_fruit_position_error',
    'counterfactual_numeric_jitter_max_linear_velocity_error',
    'counterfactual_numeric_jitter_max_orientation_error',
    'counterfactual_numeric_jitter_max_angular_velocity_error',
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
    'shapley_numeric_jitter_dropped',
    'shapley_semantic_divergence_dropped',
    'shapley_numeric_jitter_max_merge_event_position_error',
    'shapley_numeric_jitter_max_fruit_position_error',
    'shapley_numeric_jitter_max_linear_velocity_error',
    'shapley_numeric_jitter_max_orientation_error',
    'shapley_numeric_jitter_max_angular_velocity_error',
    'shapley_samples_inserted',
    'checkpoint_bytes',
    'checkpoint_step_materialization',
    'checkpoint_extra_materialization',
)

EPISODE_FIELDS = (
    'episode_index',
    'phase',
    'update_step',
    'env_steps',
    'score',
    'episode_reward',
    'episode_length',
    'terminated',
    'truncated',
)


def _graph():
    return GraphTensor(
        node_features=torch.zeros((2, 2), dtype=torch.float16),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_features=torch.empty((0, 1), dtype=torch.float16),
        action_node_indices=torch.tensor((0, 1), dtype=torch.long),
        action_indices=torch.tensor((0, 1), dtype=torch.long),
        node_feature_names=('x', 'y'),
        edge_feature_names=('distance',),
    )


def _causal_sample(
        *,
        step,
        supervision_kind,
        stratum,
        direction):
    graph = _graph()
    return CausalSample(
        graph=graph,
        actual_action_offset=0,
        comparison_action_offset=1,
        direction=direction,
        target_margin=1.0,
        confidence=0.9,
        cause_type=f'fixture-{supervision_kind}-{stratum}',
        delay=1,
        transition_key=TransitionKey(0, 0, step),
        attribution_version='fixture-v1',
        supervision_kind=supervision_kind,
        stratum=stratum,
        event_key=f'event:{step}',
        budget_key=f'budget:{step}',
        target_delta=(
            float(direction)
            if supervision_kind != 'rule'
            else None
        ),
        graph_schema_fingerprint=graph_schema_fingerprint(graph),
    )


class SyntheticTrainingRun:
    """生成体积很小、但契约完整的训练产物 fixture。"""

    def __init__(
            self,
            root,
            *,
            name='synthetic_run',
            total_updates=5,
            worker_count=2):
        self.run_dir = Path(root) / name
        self.run_dir.mkdir()
        (self.run_dir / 'checkpoints').mkdir()
        (self.run_dir / 'plots').mkdir()
        self.total_updates = int(total_updates)
        self.worker_count = int(worker_count)
        self.final_env_steps = 105
        self.baseline_config = (
            self.run_dir.parent / f'{name}_baseline.toml'
        )
        self._write_baseline_config()
        self._write_config()
        self._write_metrics(self._default_metric_rows())
        self._write_episodes()
        self._write_attribution_shutdown()
        self._write_counterfactual_shutdown()
        self._write_checkpoint(self._default_shapley_state())
        self._write_json(
            'attribution_warmup.json',
            {
                'schema_version': 1,
                'phase': 'warmup',
                'steps': 100,
                'counterfactual_snapshot_failures': 0,
                'state_analysis_calls': 100,
                'state_analysis_degraded_count': 0,
                'state_analysis_degraded_rate': 0.0,
                'p95_abs_potential_shaping_reward': 0.1,
            },
        )
        (self.run_dir / 'plots' / 'training_curves.png').write_bytes(
            b'png-training'
        )
        (
            self.run_dir
            / 'plots'
            / 'reward_breakdown_curves.png'
        ).write_bytes(b'png-reward')
        (
            self.run_dir
            / 'plots'
            / 'structure_learning_curves.png'
        ).write_bytes(b'png-structure')

    def _write_json(self, relative_path, payload):
        (self.run_dir / relative_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _write_baseline_config(self):
        project_root = Path(__file__).resolve().parents[1]
        extends_path = (
            project_root
            / 'configs'
            / 'train_dqn_causal_250k.toml'
        ).as_posix()
        self.baseline_config.write_text(
            '\n'.join(
                (
                    f'extends = {json.dumps(extends_path)}',
                    '',
                    '[runtime]',
                    f'run_dir = {json.dumps(self.run_dir.as_posix())}',
                    'device = "cpu"',
                    '',
                    '[training]',
                    f'total_updates = {self.total_updates}',
                    'warmup_steps = 2',
                    '',
                    '[replay]',
                    'replay_capacity = 10',
                    'hot_replay_capacity = 2',
                    'replay_segment_size = 4',
                    '',
                    '[counterfactual]',
                    'counterfactual_queue_capacity = 8',
                    '',
                    '[model]',
                    'hidden_dim = 8',
                    'message_layers = 1',
                    '',
                    '[parallel]',
                    f'num_envs = {self.worker_count}',
                    '',
                )
            ),
            encoding='utf-8',
        )

    def _write_config(self):
        parsed = parse_training_args(
            ('--config', str(self.baseline_config))
        )
        validate_training_args(parsed)
        args = vars(parsed)
        self.config_args = dict(args)
        manifest = create_run_manifest(
            args,
            created_at_utc='2026-07-27T00:00:00Z',
            metadata={'fixture': True},
        )
        self.run_manifest = manifest.to_dict()
        self._write_json(
            'config.json',
            {
                'args': args,
                'run_manifest': self.run_manifest,
                'git': {
                    'commit': 'fixture-commit',
                    'branch': 'codex/work-1',
                    'dirty': False,
                },
                'fingerprints': {
                    'training_config': (
                        manifest.config_fingerprint
                    ),
                },
            },
        )

    def _default_metric_rows(self):
        updates = (1, max(2, self.total_updates - 2), self.total_updates)
        env_steps = (101, 103, self.final_env_steps)
        rows = []
        for index, (update, env_step) in enumerate(
                zip(updates, env_steps)):
            rows.append(
                {
                    'update_step': update,
                    'env_steps': env_step,
                    'td_loss': 1.0 / (index + 1),
                    'structural_loss': (
                        0.0 if index == 0 else 0.3 / index
                    ),
                    'weighted_structural_loss': (
                        0.0 if index == 0 else 0.045 / index
                    ),
                    'structural_valid_count': (
                        0 if index == 0 else 12 + 6 * index
                    ),
                    'structural_sample_count': (
                        0 if index == 0 else 4
                    ),
                    'structural_mean_abs_error': (
                        0.0 if index == 0 else 0.4 / index
                    ),
                    'mean_q': 0.2 + index / 10,
                    'mean_target': 0.3 + index / 10,
                    'mean_abs_td_error': 0.1,
                    'actor_inference_requests': (1, 4, 8)[index],
                    'actor_inference_batches': (1, 3, 5)[index],
                    'actor_inference_mean_batch_size': (
                        (1, 4, 8)[index] / (1, 3, 5)[index]
                    ),
                    'actor_inference_max_batch': (1, 2, 2)[index],
                    'actor_inference_seconds': (
                        0.01 * (index + 1)
                    ),
                    'causal_update_applied': int(index > 0),
                    'rule_batch_size': 4 if index > 0 else 0,
                    'counterfactual_batch_size': (
                        2 if index == 2 else 0
                    ),
                    'shapley_batch_size': 0,
                    'causal_replay_positive_count': index + 1,
                    'causal_replay_negative_count': index + 1,
                    'causal_replay_rule_count': 2 + index,
                    'causal_replay_cf_count': 2 if index == 2 else 0,
                    'collect_counterfactual_snapshot_failures': 0,
                    'collect_p95_abs_potential_shaping_reward': 0.1,
                    'collect_state_analysis_degraded_rate': 0.0,
                    'counterfactual_results_completed': (
                        1 if index == 2 else 0
                    ),
                    'counterfactual_results_failed': 0,
                    'counterfactual_reproduction_passed': (
                        1 if index == 2 else 0
                    ),
                    'counterfactual_reproduction_failed': 0,
                    'counterfactual_numeric_jitter_dropped': 0,
                    'counterfactual_semantic_divergence_dropped': 0,
                    'counterfactual_numeric_jitter_max_merge_event_position_error': 0.0,
                    'counterfactual_numeric_jitter_max_fruit_position_error': 0.0,
                    'counterfactual_numeric_jitter_max_linear_velocity_error': 0.0,
                    'counterfactual_numeric_jitter_max_orientation_error': 0.0,
                    'counterfactual_numeric_jitter_max_angular_velocity_error': 0.0,
                    'counterfactual_samples_inserted': (
                        2 if index == 2 else 0
                    ),
                    'counterfactual_pending_tasks': 0,
                    'counterfactual_admission_slots_used': (
                        0 if index == 0 else 1
                    ),
                    'counterfactual_admission_slots_available': (
                        3 if index == 0 else 2
                    ),
                    'counterfactual_candidate_pool_capacity': 8,
                    'counterfactual_candidate_pool_count': (
                        1 if index < 2 else 0
                    ),
                    'counterfactual_candidate_offers': index + 1,
                    'counterfactual_candidate_dispatch_attempts': (
                        index + 1
                    ),
                    'counterfactual_candidate_dispatch_admitted': (
                        0 if index == 0 else 1
                    ),
                    'counterfactual_candidate_close_dropped': 0,
                    'counterfactual_actual_token_ratio': 0.05,
                    'counterfactual_projected_token_ratio': 0.06,
                    'counterfactual_hard_budget_respected': 1,
                    'counterfactual_drop_reasons': '{}',
                    'counterfactual_failure_reasons': '{}',
                    'counterfactual_failure_diagnostic_codes': '{}',
                    'counterfactual_failure_trigger_reasons': '{}',
                    'shapley_enabled': 1,
                    'shapley_events_observed': 100 * (index + 1),
                    'shapley_events_selected': 0,
                    'shapley_tasks_completed': 0,
                    'shapley_tasks_failed': 0,
                    'shapley_terminal_dropped': 0,
                    'shapley_reproduction_passed': 0,
                    'shapley_reproduction_failed': 0,
                    'shapley_numeric_jitter_dropped': 0,
                    'shapley_semantic_divergence_dropped': 0,
                    'shapley_numeric_jitter_max_merge_event_position_error': 0.0,
                    'shapley_numeric_jitter_max_fruit_position_error': 0.0,
                    'shapley_numeric_jitter_max_linear_velocity_error': 0.0,
                    'shapley_numeric_jitter_max_orientation_error': 0.0,
                    'shapley_numeric_jitter_max_angular_velocity_error': 0.0,
                    'shapley_samples_inserted': 0,
                    'checkpoint_bytes': 1024,
                    'checkpoint_step_materialization': (
                        'hardlink' if index == 1 else ''
                    ),
                    'checkpoint_extra_materialization': '',
                }
            )
        return rows

    def read_metrics(self):
        with (self.run_dir / 'metrics.csv').open(
                'r',
                newline='',
                encoding='utf-8') as file_obj:
            return list(csv.DictReader(file_obj))

    def _write_metrics(self, rows):
        with (self.run_dir / 'metrics.csv').open(
                'w',
                newline='',
                encoding='utf-8') as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def write_metrics(self, rows):
        self._write_metrics(rows)

    def _write_episodes(self):
        with (self.run_dir / 'episode_metrics.csv').open(
                'w',
                newline='',
                encoding='utf-8') as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            for index in range(1, 51):
                writer.writerow(
                    {
                        'episode_index': index,
                        'phase': 'train',
                        'update_step': 1 + (
                            (index - 1)
                            * (self.total_updates - 1)
                            // 49
                        ),
                        'env_steps': 55 + index,
                        'score': 100 + index,
                        'episode_reward': 10 + index / 10,
                        'episode_length': 20 + index,
                        'terminated': 1,
                        'truncated': 0,
                    }
                )

    def _write_attribution_shutdown(self):
        workers = []
        for worker_id in range(self.worker_count):
            workers.append(
                {
                    'worker_id': worker_id,
                    'cancelled_pending_count': 2,
                    'n_step_flush_emitted': 2,
                    'event_type_counts': {
                        'REACHABILITY_SEALED': 2,
                    },
                    'resolution_reason_counts': {
                        'worker_shutdown': 2,
                    },
                }
            )
        self._write_json(
            'attribution_shutdown.json',
            {
                'created_at': '2026-07-27T00:00:00+08:00',
                'cancelled_pending_count': 2 * self.worker_count,
                'n_step_flush_emitted': 2 * self.worker_count,
                'event_type_counts': {
                    'REACHABILITY_SEALED': 2 * self.worker_count,
                },
                'resolution_reason_counts': {
                    'worker_shutdown': 2 * self.worker_count,
                },
                'workers': workers,
            },
        )

    def _default_cf_shutdown(self):
        return {
            'schema_version': 1,
            'enabled': True,
            'closed': True,
            'active_task_ids': [],
            'pending_task_count': 0,
            'candidate_pool_capacity': 8,
            'candidate_pool_count': 0,
            'scheduler': {
                'real_steps': self.final_env_steps,
                'queued': 0,
                'inflight': 0,
                'failed': 0,
                'tokens_reserved': 0,
                'tokens_consumed': 5,
                'token_overrun': 0,
                'external_active_reservations': 0,
            },
            'cumulative': {
                'candidate_offers': 3,
                'candidate_close_dropped': 0,
                'results_completed': 1,
                'results_failed': 0,
                'reproduction_passed': 1,
                'reproduction_failed': 0,
                'numeric_jitter_dropped': 0,
                'semantic_divergence_dropped': 0,
                'numeric_jitter_max_merge_event_position_error': 0.0,
                'numeric_jitter_max_fruit_position_error': 0.0,
                'numeric_jitter_max_linear_velocity_error': 0.0,
                'numeric_jitter_max_orientation_error': 0.0,
                'numeric_jitter_max_angular_velocity_error': 0.0,
                'samples_inserted': 2,
            },
            'actual_token_ratio': 0.05,
            'projected_token_ratio': 0.05,
            'hard_budget_respected': True,
            'circuit_open': False,
        }

    def _write_counterfactual_shutdown(self, payload=None):
        self._write_json(
            'counterfactual_shutdown.json',
            payload or self._default_cf_shutdown(),
        )

    def read_counterfactual_shutdown(self):
        return json.loads(
            (self.run_dir / 'counterfactual_shutdown.json').read_text(
                encoding='utf-8'
            )
        )

    def write_counterfactual_shutdown(self, payload):
        self._write_counterfactual_shutdown(payload)

    @staticmethod
    def _default_shapley_state():
        return {
            'enabled': True,
            'closed': True,
            'active_task_id': None,
            'pending_task_ids': (),
            'observed_event_count': 300,
            'selected_event_count': 0,
            'selected_ratio': 0.0,
            'cumulative': {
                'results_completed': 0,
                'results_failed': 0,
                'selected_terminal_dropped': 0,
                'reproduction_passed': 0,
                'reproduction_failed': 0,
                'numeric_jitter_dropped': 0,
                'semantic_divergence_dropped': 0,
                'numeric_jitter_max_merge_event_position_error': 0.0,
                'numeric_jitter_max_fruit_position_error': 0.0,
                'numeric_jitter_max_linear_velocity_error': 0.0,
                'numeric_jitter_max_orientation_error': 0.0,
                'numeric_jitter_max_angular_velocity_error': 0.0,
                'samples_inserted': 0,
            },
        }

    def _checkpoint_payload(self, shapley_state):
        online_model = GNNQNetwork(
            hidden_dim=8,
            message_layers=1,
            activation='silu',
            dropout=0.0,
        )
        target_model = GNNQNetwork(
            hidden_dim=8,
            message_layers=1,
            activation='silu',
            dropout=0.0,
        )
        target_model.load_state_dict(online_model.state_dict())
        optimizer = torch.optim.Adam(
            online_model.parameters(),
            lr=1e-4,
        )
        replay = ReplayBuffer(
            capacity=self.config_args['replay_capacity'],
            seed=0,
            hot_capacity=self.config_args[
                'hot_replay_capacity'
            ],
            cold_dir=self.run_dir / 'replay_cold',
            segment_size=self.config_args['replay_segment_size'],
            cold_cache_size=self.config_args[
                'replay_cold_cache_size'
            ],
            cold_sample_ratio=self.config_args[
                'replay_cold_sample_ratio'
            ],
            cold_cache_refresh_interval=self.config_args[
                'replay_cold_cache_refresh_interval'
            ],
        )
        replay.push(TensorTransition(
            graph=_graph(),
            action_offset=0,
            reward=1.0,
            next_graph=_graph(),
            terminated=False,
            truncated=False,
        ))
        causal = CausalReplayBuffer(
            capacity=self.config_args['causal_replay_capacity'],
            seed=0,
        )
        causal.extend((
            _causal_sample(
                step=1,
                supervision_kind='rule',
                stratum='positive_setup',
                direction=1,
            ),
            _causal_sample(
                step=2,
                supervision_kind='rule',
                stratum='negative_blocking',
                direction=-1,
            ),
            _causal_sample(
                step=3,
                supervision_kind='counterfactual',
                stratum='counterfactual',
                direction=1,
            ),
        ))
        return {
            'schema_version': 1,
            'run_manifest': self.run_manifest,
            'training_state': {
                'online_model': online_model.state_dict(),
                'target_model': target_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'update_step': self.total_updates,
                'trainer_update_step': self.total_updates,
                'env_steps': self.final_env_steps,
                'shapley': shapley_state,
            },
            'rng_state': capture_rng_state(),
            'component_snapshots': {
                'replay_buffer': {
                    'state_protocol': 'checkpoint_state_dict',
                    'state': replay.checkpoint_state_dict(),
                    'manifest': replay.checkpoint_manifest(),
                },
                'causal_replay_buffer': {
                    'state_protocol': 'checkpoint_state_dict',
                    'state': causal.checkpoint_state_dict(),
                    'manifest': causal.checkpoint_manifest(),
                },
            },
        }

    def _write_checkpoint(self, shapley_state):
        torch.save(
            self._checkpoint_payload(shapley_state),
            self.run_dir / 'checkpoints' / 'latest.pt',
        )

    def write_shapley_checkpoint(self, shapley_state):
        self._write_checkpoint(shapley_state)

    def write_causal_checkpoint(self, items):
        payload = self._checkpoint_payload(
            self._default_shapley_state()
        )
        causal = CausalReplayBuffer(
            capacity=self.config_args['causal_replay_capacity'],
            seed=0,
        )
        causal.extend(tuple(items))
        payload['component_snapshots']['causal_replay_buffer'] = {
            'state_protocol': 'checkpoint_state_dict',
            'state': causal.checkpoint_state_dict(),
            'manifest': causal.checkpoint_manifest(),
        }
        torch.save(
            payload,
            self.run_dir / 'checkpoints' / 'latest.pt',
        )

    def write_checkpoint_payload(self, payload):
        torch.save(
            payload,
            self.run_dir / 'checkpoints' / 'latest.pt',
        )


class CheckTrainingReadinessTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.fixture = SyntheticTrainingRun(self.root)

    def _check(self, payload, name):
        return next(
            check
            for check in payload['checks']
            if check['name'] == name
        )

    def _set_counterfactual_reproduction_outcomes(
            self,
            *,
            strict,
            numeric,
            semantic,
            numeric_errors=None):
        """把 fixture 最后一段改成指定的三态累计账本。"""

        numeric_errors = dict(numeric_errors or {})
        failed = numeric + semantic
        reason_counts = {}
        if numeric:
            reason_counts['original_reproduction_numeric_jitter'] = numeric
        if semantic:
            reason_counts[
                'original_reproduction_semantic_divergence'
            ] = semantic

        rows = self.fixture.read_metrics()
        row = rows[-1]
        row['counterfactual_results_completed'] = str(strict)
        row['counterfactual_results_failed'] = str(failed)
        row['counterfactual_reproduction_passed'] = str(strict)
        row['counterfactual_reproduction_failed'] = str(failed)
        row['counterfactual_numeric_jitter_dropped'] = str(numeric)
        row['counterfactual_semantic_divergence_dropped'] = str(
            semantic
        )
        row['counterfactual_failure_reasons'] = json.dumps(
            reason_counts
        )
        row['counterfactual_failure_diagnostic_codes'] = json.dumps(
            {'reproduction_outcome_classified': failed}
            if failed
            else {}
        )
        row['counterfactual_failure_trigger_reasons'] = json.dumps(
            {'fixture': failed} if failed else {}
        )
        for suffix in (
                'merge_event_position',
                'fruit_position',
                'linear_velocity',
                'orientation',
                'angular_velocity'):
            row[
                f'counterfactual_numeric_jitter_max_{suffix}_error'
            ] = str(numeric_errors.get(suffix, 0.0))
        self.fixture.write_metrics(rows)

        shutdown = self.fixture.read_counterfactual_shutdown()
        shutdown['scheduler']['failed'] = failed
        shutdown['cumulative'].update(
            {
                'results_completed': strict,
                'results_failed': failed,
                'reproduction_passed': strict,
                'reproduction_failed': failed,
                'numeric_jitter_dropped': numeric,
                'semantic_divergence_dropped': semantic,
                'failure_reason_counts': reason_counts,
                'failure_diagnostic_code_counts': (
                    {'reproduction_outcome_classified': failed}
                    if failed
                    else {}
                ),
                'failure_trigger_reason_counts': (
                    {'fixture': failed} if failed else {}
                ),
            }
        )
        for suffix in (
                'merge_event_position',
                'fruit_position',
                'linear_velocity',
                'orientation',
                'angular_velocity'):
            shutdown['cumulative'][
                f'numeric_jitter_max_{suffix}_error'
            ] = numeric_errors.get(suffix, 0.0)
        self.fixture.write_counterfactual_shutdown(shutdown)

    def _set_shapley_reproduction_outcomes(
            self,
            *,
            strict,
            numeric,
            semantic,
            numeric_errors=None):
        """生成既有严格样本、又有可审计 drop 的 Shapley 账本。"""

        numeric_errors = dict(numeric_errors or {})
        failed = numeric + semantic
        selected = strict + failed
        rows = self.fixture.read_metrics()
        row = rows[-1]
        row['causal_update_applied'] = '1'
        row['shapley_batch_size'] = '1'
        row['shapley_events_selected'] = str(selected)
        row['shapley_tasks_completed'] = str(strict)
        row['shapley_tasks_failed'] = str(failed)
        row['shapley_reproduction_passed'] = str(strict)
        row['shapley_reproduction_failed'] = str(failed)
        row['shapley_numeric_jitter_dropped'] = str(numeric)
        row['shapley_semantic_divergence_dropped'] = str(semantic)
        row['shapley_samples_inserted'] = str(max(1, strict))
        for suffix in (
                'merge_event_position',
                'fruit_position',
                'linear_velocity',
                'orientation',
                'angular_velocity'):
            row[
                f'shapley_numeric_jitter_max_{suffix}_error'
            ] = str(numeric_errors.get(suffix, 0.0))
        self.fixture.write_metrics(rows)

        state = self.fixture._default_shapley_state()
        state.update(
            {
                'selected_event_count': selected,
                'selected_ratio': selected / 300,
            }
        )
        state['cumulative'].update(
            {
                'results_completed': strict,
                'results_failed': failed,
                'reproduction_passed': strict,
                'reproduction_failed': failed,
                'numeric_jitter_dropped': numeric,
                'semantic_divergence_dropped': semantic,
                'samples_inserted': max(1, strict),
            }
        )
        for suffix in (
                'merge_event_position',
                'fruit_position',
                'linear_velocity',
                'orientation',
                'angular_velocity'):
            state['cumulative'][
                f'numeric_jitter_max_{suffix}_error'
            ] = numeric_errors.get(suffix, 0.0)
        self.fixture.write_shapley_checkpoint(state)

    def test_complete_5k_fixture_passes_with_recorded_shutdown_cancellations(self):
        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(payload['ready'])
        self.assertEqual(payload['schema_version'], 3)
        self.assertEqual(payload['exit_code'], 0)
        self.assertGreater(
            payload['attribution_shutdown']['cancelled_pending_count'],
            0,
        )
        self.assertTrue(
            self._check(
                payload,
                'attribution_worker_finalization',
            )['passed']
        )
        self.assertIn(
            'permitted in 5k',
            payload['shapley']['interpretation'],
        )
        self.assertEqual(payload['warnings'], [])
        self.assertTrue(
            self._check(
                payload,
                'structural_supervision_targets_reached_optimizer',
            )['passed']
        )
        self.assertTrue(
            self._check(
                payload,
                'centralized_actor_inference_activity',
            )['passed']
        )

    def test_structure_learning_plot_is_required_artifact(self):
        (
            self.fixture.run_dir
            / 'plots'
            / 'structure_learning_curves.png'
        ).unlink()

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        artifact_gate = self._check(payload, 'required_artifacts')
        self.assertIn(
            'structure_learning_curves',
            artifact_gate['details']['missing_or_empty'],
        )

    def test_structural_supervision_requires_valid_nonzero_optimizer_term(self):
        rows = self.fixture.read_metrics()
        for row in rows:
            row['structural_loss'] = '0'
            row['weighted_structural_loss'] = '0'
            row['structural_valid_count'] = '0'
            row['structural_sample_count'] = '0'
            row['structural_mean_abs_error'] = '0'
        self.fixture.write_metrics(rows)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        self.assertIn(
            'structural_supervision_targets_reached_optimizer',
            payload['required_failures'],
        )
        gate = self._check(
            payload,
            'structural_supervision_targets_reached_optimizer',
        )
        self.assertEqual(
            gate['details']['optimizer_evidence_rows'],
            [],
        )

    def test_structural_supervision_rejects_sixth_dimension_only(self):
        rows = self.fixture.read_metrics()
        for row in rows:
            row['structural_loss'] = '0.2'
            row['weighted_structural_loss'] = '0.03'
            row['structural_valid_count'] = '4'
            row['structural_sample_count'] = '4'
            row['structural_mean_abs_error'] = '0.3'
        self.fixture.write_metrics(rows)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        gate = self._check(
            payload,
            'structural_supervision_targets_reached_optimizer',
        )
        self.assertEqual(
            gate['details']['optimizer_evidence_rows'],
            [],
        )
        self.assertEqual(
            gate['details'][
                'minimum_mean_valid_dimensions_for_evidence'
            ],
            5.0,
        )

    def test_v2_structural_and_actor_metrics_are_required_schema(self):
        rows = self.fixture.read_metrics()
        omitted = {
            'structural_valid_count',
            'actor_inference_requests',
        }
        fields = tuple(
            field
            for field in METRIC_FIELDS
            if field not in omitted
        )
        with (self.fixture.run_dir / 'metrics.csv').open(
                'w',
                newline='',
                encoding='utf-8') as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=fields,
                extrasaction='ignore',
            )
            writer.writeheader()
            writer.writerows(rows)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        schema = self._check(
            payload,
            'metrics_schema_and_finite_values',
        )
        self.assertFalse(schema['passed'])
        self.assertEqual(
            set(schema['details']['missing_fields']),
            omitted,
        )

    def test_structural_supervision_rejects_invalid_count_and_weighting(self):
        rows = self.fixture.read_metrics()
        rows[-1]['structural_valid_count'] = '25'
        rows[-1]['structural_sample_count'] = '4'
        rows[-1]['weighted_structural_loss'] = '0.5'
        self.fixture.write_metrics(rows)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        gate = self._check(
            payload,
            'structural_supervision_targets_reached_optimizer',
        )
        reasons = gate['details']['invalid_rows'][0]['reasons']
        self.assertIn(
            'structural_valid_count_exceeds_six_per_sample',
            reasons,
        )
        self.assertIn(
            'weighted_structural_loss_lambda_mismatch',
            reasons,
        )

    def test_centralized_actor_requires_real_consistent_batches(self):
        rows = self.fixture.read_metrics()
        for row in rows:
            row['actor_inference_requests'] = '0'
            row['actor_inference_batches'] = '0'
            row['actor_inference_mean_batch_size'] = '0'
            row['actor_inference_max_batch'] = '0'
            row['actor_inference_seconds'] = '0'
        self.fixture.write_metrics(rows)

        inactive = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )
        self.assertFalse(inactive['ready'])
        self.assertIn(
            'centralized_actor_inference_activity',
            inactive['required_failures'],
        )

        rows = self.fixture.read_metrics()
        rows[-1]['actor_inference_requests'] = '8'
        rows[-1]['actor_inference_batches'] = '9'
        rows[-1]['actor_inference_mean_batch_size'] = '3'
        rows[-1]['actor_inference_max_batch'] = '17'
        rows[-1]['actor_inference_seconds'] = '0.1'
        self.fixture.write_metrics(rows)

        inconsistent = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )
        gate = self._check(
            inconsistent,
            'centralized_actor_inference_activity',
        )
        self.assertFalse(gate['passed'])
        reasons = gate['details']['invalid_rows'][-1]['reasons']
        self.assertIn('actor_batches_exceed_requests', reasons)
        self.assertIn('actor_mean_batch_size_mismatch', reasons)
        self.assertIn('actor_max_batch_exceeds_config', reasons)

    def test_metric_corruption_and_budget_violations_fail_gate(self):
        rows = self.fixture.read_metrics()
        rows[0]['td_loss'] = 'nan'
        rows[1]['update_step'] = rows[0]['update_step']
        rows[1]['counterfactual_hard_budget_respected'] = '0'
        rows[1]['counterfactual_actual_token_ratio'] = '0.11'
        rows[1]['collect_counterfactual_snapshot_failures'] = '1'
        self.fixture.write_metrics(rows)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        failures = set(payload['required_failures'])
        self.assertIn('metrics_schema_and_finite_values', failures)
        self.assertIn(
            'training_progress_complete_and_monotonic',
            failures,
        )
        self.assertIn('counterfactual_budget_all_windows', failures)
        self.assertIn('counterfactual_snapshot_failures', failures)

    def test_threshold_override_and_copy_materialization_warning(self):
        rows = self.fixture.read_metrics()
        for row in rows:
            row['collect_p95_abs_potential_shaping_reward'] = '0.8'
        rows[1]['checkpoint_step_materialization'] = 'copy'
        self.fixture.write_metrics(rows)

        default_payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )
        relaxed_payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
            thresholds=ReadinessThresholds(max_shaping_p95=1.0),
        )

        self.assertFalse(default_payload['ready'])
        self.assertIn(
            'potential_shaping_p95',
            default_payload['required_failures'],
        )
        self.assertTrue(relaxed_payload['ready'])
        self.assertIn(
            'checkpoint_copy_materialization',
            relaxed_payload['warnings'],
        )
        warning = self._check(
            relaxed_payload,
            'checkpoint_copy_materialization',
        )
        self.assertEqual(
            warning['details']['estimated_peak_checkpoint_multiplier'],
            2,
        )

    def test_counterfactual_shutdown_must_be_fully_drained(self):
        shutdown = self.fixture.read_counterfactual_shutdown()
        shutdown['pending_task_count'] = 1
        shutdown['candidate_pool_count'] = 1
        shutdown['scheduler']['queued'] = 1
        shutdown['circuit_open'] = True
        self.fixture.write_counterfactual_shutdown(shutdown)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        check = self._check(
            payload,
            'counterfactual_shutdown_drained',
        )
        self.assertFalse(check['passed'])
        self.assertTrue(any(
            'pending_task_count' in error
            for error in check['details']['errors']
        ))
        self.assertTrue(any(
            'candidate_pool_count' in error
            for error in check['details']['errors']
        ))
        self.assertTrue(any(
            'circuit_open' in error
            for error in check['details']['errors']
        ))

    def test_classified_physics_failure_uses_rate_gate_not_infra_gate(self):
        rows = self.fixture.read_metrics()
        rows[-1]['counterfactual_results_failed'] = '1'
        rows[-1]['counterfactual_reproduction_failed'] = '1'
        rows[-1][
            'counterfactual_semantic_divergence_dropped'
        ] = '1'
        rows[-1]['counterfactual_failure_reasons'] = json.dumps(
            {'original_reproduction_mismatch': 1}
        )
        rows[-1][
            'counterfactual_failure_diagnostic_codes'
        ] = json.dumps({'original_mismatch_state_checksum': 1})
        rows[-1][
            'counterfactual_failure_trigger_reasons'
        ] = json.dumps({'random_rule_audit': 1})
        self.fixture.write_metrics(rows)

        shutdown = self.fixture.read_counterfactual_shutdown()
        shutdown['scheduler']['failed'] = 1
        shutdown['cumulative'].update(
            {
                'results_failed': 1,
                'reproduction_failed': 1,
                'semantic_divergence_dropped': 1,
                'failure_records_created': 1,
                'failure_reason_counts': {
                    'original_reproduction_mismatch': 1,
                },
                'failure_diagnostic_code_counts': {
                    'original_mismatch_state_checksum': 1,
                },
                'failure_trigger_reason_counts': {
                    'random_rule_audit': 1,
                },
            }
        )
        self.fixture.write_counterfactual_shutdown(shutdown)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(payload['ready'])
        gate = self._check(
            payload,
            'counterfactual_failures_classified_and_infrastructure_clean',
        )
        self.assertTrue(gate['passed'])
        self.assertIn(
            'counterfactual_reproduction_failure_rate_sample_size',
            payload['warnings'],
        )

        rows[-1]['counterfactual_failure_reasons'] = '{}'
        self.fixture.write_metrics(rows)
        unclassified = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )
        self.assertFalse(unclassified['ready'])
        self.assertIn(
            'counterfactual_failures_classified_and_infrastructure_clean',
            unclassified['required_failures'],
        )

    def test_counterfactual_numeric_jitter_is_reported_without_warning(self):
        self._set_counterfactual_reproduction_outcomes(
            strict=1,
            numeric=1,
            semantic=0,
            numeric_errors={
                'merge_event_position': 0.09,
                'fruit_position': 0.03,
                'linear_velocity': 0.008,
                'orientation': 0.0006,
                'angular_velocity': 0.0004,
            },
        )

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(payload['ready'])
        self.assertNotIn(
            'counterfactual_reproduction_failure_rate_sample_size',
            payload['warnings'],
        )
        outcome = self._check(
            payload,
            'counterfactual_reproduction_outcome_accounting',
        )['details']
        self.assertTrue(outcome['failure_outcomes_fully_accounted'])
        self.assertEqual(outcome['numeric_jitter_dropped'], 1)
        self.assertEqual(outcome['semantic_divergence_dropped'], 0)
        self.assertEqual(outcome['unknown_failed'], 0)
        self.assertEqual(
            outcome['numeric_jitter_error_maxima'][
                'numeric_jitter_max_merge_event_position_error'
            ],
            0.09,
        )

    def test_counterfactual_semantic_rate_uses_all_three_outcomes(self):
        self._set_counterfactual_reproduction_outcomes(
            strict=98,
            numeric=1,
            semantic=1,
        )
        passing = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(passing['ready'])
        gate = self._check(
            passing,
            'counterfactual_reproduction_failure_rate',
        )
        self.assertEqual(gate['details']['total'], 100)
        self.assertEqual(gate['details']['strict_matches'], 98)
        self.assertEqual(
            gate['details']['numeric_jitter_dropped'],
            1,
        )
        self.assertAlmostEqual(gate['details']['rate'], 0.01)

        self._set_counterfactual_reproduction_outcomes(
            strict=97,
            numeric=1,
            semantic=2,
        )
        failing = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )
        self.assertFalse(failing['ready'])
        self.assertIn(
            'counterfactual_reproduction_failure_rate',
            failing['required_failures'],
        )

    def test_counterfactual_unknown_failure_and_negative_maximum_block(self):
        rows = self.fixture.read_metrics()
        rows[-1]['counterfactual_results_failed'] = '1'
        rows[-1]['counterfactual_reproduction_failed'] = '1'
        rows[-1]['counterfactual_failure_reasons'] = json.dumps(
            {'original_reproduction_mismatch': 1}
        )
        rows[-1][
            'counterfactual_numeric_jitter_max_fruit_position_error'
        ] = '-0.1'
        self.fixture.write_metrics(rows)
        shutdown = self.fixture.read_counterfactual_shutdown()
        shutdown['scheduler']['failed'] = 1
        shutdown['cumulative'].update(
            {
                'results_failed': 1,
                'reproduction_failed': 1,
                'failure_reason_counts': {
                    'original_reproduction_mismatch': 1,
                },
            }
        )
        self.fixture.write_counterfactual_shutdown(shutdown)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        self.assertIn(
            'counterfactual_reproduction_outcome_accounting',
            payload['required_failures'],
        )
        outcome = self._check(
            payload,
            'counterfactual_reproduction_outcome_accounting',
        )['details']
        self.assertEqual(outcome['unknown_failed'], 1)
        self.assertEqual(
            outcome['invalid_metric_numeric_error_rows'][
                'numeric_jitter_max_fruit_position_error'
            ],
            [2],
        )

    def test_counterfactual_runner_exception_is_infrastructure_failure(self):
        rows = self.fixture.read_metrics()
        rows[-1]['counterfactual_drop_reasons'] = json.dumps(
            {'runner_failure': 1}
        )
        self.fixture.write_metrics(rows)
        shutdown = self.fixture.read_counterfactual_shutdown()
        shutdown['scheduler']['failed'] = 1
        shutdown['cumulative']['drop_reason_counts'] = {
            'runner_failure': 1,
        }
        self.fixture.write_counterfactual_shutdown(shutdown)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        gate = self._check(
            payload,
            'counterfactual_failures_classified_and_infrastructure_clean',
        )
        self.assertFalse(gate['passed'])
        self.assertEqual(
            gate['details']['infrastructure_failure_counts'][
                'runner_failure'
            ],
            1,
        )

    def test_rule_sign_gate_uses_exact_checkpoint_cross_tab(self):
        self.fixture.write_causal_checkpoint(
            (
                _causal_sample(
                    step=11,
                    supervision_kind='rule',
                    stratum='positive_setup',
                    direction=1,
                ),
                _causal_sample(
                    step=12,
                    supervision_kind='counterfactual',
                    stratum='counterfactual',
                    direction=-1,
                ),
            )
        )

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        check = self._check(
            payload,
            'causal_replay_positive_negative_rule_samples',
        )
        self.assertFalse(check['passed'])
        exact = check['details']['checkpoint_exact_counts']
        self.assertEqual(exact['positive_rule_count'], 1)
        self.assertEqual(exact['negative_rule_count'], 0)

    def test_replay_cold_growth_is_reported_but_never_deleted(self):
        cold_dir = self.fixture.run_dir / 'replay_cold'
        cold_dir.mkdir(exist_ok=True)
        for index in range(3):
            (cold_dir / f'segment_{index:08d}.pt').write_bytes(
                b'cold-segment'
            )

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(payload['ready'])
        self.assertIn(
            'replay_cold_storage_growth',
            payload['warnings'],
        )
        self.assertEqual(payload['replay_cold']['segment_count'], 3)
        self.assertEqual(
            payload['replay_cold'][
                'max_live_segments_from_config'
            ],
            2,
        )
        self.assertIn(
            'segment_count_exceeds_one_live_generation',
            payload['replay_cold']['suspicious_reasons'],
        )
        self.assertEqual(len(list(cold_dir.glob('segment_*.pt'))), 3)

    def test_10k_records_zero_shapley_but_selected_failure_blocks(self):
        fixture = SyntheticTrainingRun(
            self.root,
            name='synthetic_10k',
            total_updates=10_000,
        )
        zero_payload = audit_training_run(
            fixture.run_dir,
            stage='10k',
        )
        self.assertTrue(zero_payload['ready'])
        self.assertIn('shapley_zero_selection', zero_payload['warnings'])

        failed_state = fixture._default_shapley_state()
        failed_state.update(
            {
                'selected_event_count': 1,
                'selected_ratio': 1 / 300,
            }
        )
        failed_state['cumulative'] = {
            'results_completed': 0,
            'results_failed': 1,
            'reproduction_passed': 1,
            'reproduction_failed': 0,
            'numeric_jitter_dropped': 0,
            'semantic_divergence_dropped': 0,
            'numeric_jitter_max_merge_event_position_error': 0.0,
            'numeric_jitter_max_fruit_position_error': 0.0,
            'numeric_jitter_max_linear_velocity_error': 0.0,
            'numeric_jitter_max_orientation_error': 0.0,
            'numeric_jitter_max_angular_velocity_error': 0.0,
            'samples_inserted': 0,
        }
        fixture.write_shapley_checkpoint(failed_state)

        failed_payload = audit_training_run(
            fixture.run_dir,
            stage='10k',
        )
        self.assertFalse(failed_payload['ready'])
        self.assertIn(
            'shapley_stage_evidence_and_shutdown',
            failed_payload['required_failures'],
        )
        self.assertEqual(
            failed_payload['shapley']['reproduction_outcomes'][
                'unknown_failed'
            ],
            0,
        )
        self.assertEqual(
            failed_payload['shapley']['reproduction_outcomes'][
                'non_gate_result_failed'
            ],
            1,
        )

    def test_shapley_numeric_jitter_does_not_replace_strict_evidence(self):
        self._set_shapley_reproduction_outcomes(
            strict=1,
            numeric=1,
            semantic=0,
            numeric_errors={'fruit_position': 0.02},
        )

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(payload['ready'])
        self.assertEqual(payload['shapley']['completed'], 1)
        self.assertEqual(payload['shapley']['reproduction_passed'], 1)
        self.assertEqual(payload['shapley']['numeric_jitter_dropped'], 1)
        self.assertTrue(payload['shapley']['optimizer_consumed'])
        self.assertNotIn(
            'shapley_reproduction_failure_rate_sample_size',
            payload['warnings'],
        )

    def test_shapley_low_sample_semantic_divergence_is_warning(self):
        self._set_shapley_reproduction_outcomes(
            strict=1,
            numeric=0,
            semantic=1,
        )

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(payload['ready'])
        self.assertIn(
            'shapley_reproduction_failure_rate_sample_size',
            payload['warnings'],
        )
        outcome = payload['shapley']['reproduction_outcomes']
        self.assertEqual(outcome['semantic_rate_denominator'], 2)
        self.assertAlmostEqual(
            outcome['semantic_divergence_rate'],
            0.5,
        )

    def test_shapley_semantic_rate_blocks_at_sufficient_sample_size(self):
        self._set_shapley_reproduction_outcomes(
            strict=98,
            numeric=0,
            semantic=2,
        )

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        self.assertIn(
            'shapley_stage_evidence_and_shutdown',
            payload['required_failures'],
        )
        outcome = payload['shapley']['reproduction_outcomes']
        self.assertTrue(outcome['semantic_rate_evaluated'])
        self.assertFalse(outcome['semantic_rate_passed'])
        self.assertAlmostEqual(
            outcome['semantic_divergence_rate'],
            0.02,
        )

    def test_resume_sidecar_breaks_equal_counter_segments(self):
        rows = self.fixture.read_metrics()
        for row in rows:
            row['counterfactual_candidate_pool_count'] = '1'
            row['counterfactual_admission_slots_available'] = '2'
            row['counterfactual_candidate_dispatch_attempts'] = '1'
            row['counterfactual_candidate_dispatch_admitted'] = '1'
        inserted = dict(rows[0])
        inserted['update_step'] = '2'
        inserted['env_steps'] = '102'
        rows.insert(1, inserted)
        self.fixture.write_metrics(rows)
        self.fixture._write_json(
            'resume_20260727_010203_000001.json',
            {
                'saved_update_step': 2,
                'saved_env_steps': 102,
            },
        )

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(payload['ready'])
        check = self._check(
            payload,
            'counterfactual_execution_pending_saturation',
        )
        self.assertEqual(
            check['details']['resume_segment_start_indices'],
            [2],
        )
        self.assertEqual(
            check['details']['dispatch_stall_max_consecutive'],
            1,
        )

    def test_budget_retries_are_not_misclassified_as_dispatch_stall(self):
        rows = self.fixture.read_metrics()
        for index, row in enumerate(rows):
            row['counterfactual_candidate_pool_count'] = '1'
            row['counterfactual_admission_slots_available'] = '2'
            row['counterfactual_candidate_dispatch_attempts'] = str(
                10 + index
            )
            row['counterfactual_candidate_dispatch_admitted'] = '1'
        self.fixture.write_metrics(rows)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(payload['ready'])
        check = self._check(
            payload,
            'counterfactual_execution_pending_saturation',
        )
        self.assertEqual(
            check['details']['dispatch_stall_max_consecutive'],
            0,
        )

    def test_unexplained_dispatch_counter_reset_fails_gate(self):
        rows = self.fixture.read_metrics()
        rows[-1]['counterfactual_candidate_dispatch_attempts'] = '0'
        self.fixture.write_metrics(rows)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        self.assertIn(
            'counterfactual_execution_pending_saturation',
            payload['required_failures'],
        )
        check = self._check(
            payload,
            'counterfactual_execution_pending_saturation',
        )
        self.assertEqual(
            check['details']['unexpected_dispatch_reset_indices'],
            [2],
        )

    def test_shapley_selected_result_accounting_is_closed_across_resume(self):
        rows = self.fixture.read_metrics()
        for index, row in enumerate(rows):
            row['shapley_events_observed'] = (
                '100' if index == 2 else str(100 * (index + 1))
            )
            row['shapley_events_selected'] = '1'
            row['shapley_tasks_completed'] = '1'
            row['shapley_reproduction_passed'] = '1'
            row['shapley_samples_inserted'] = '1'
            row['shapley_batch_size'] = '1'
            row['causal_update_applied'] = '1'
        self.fixture.write_metrics(rows)
        self.fixture._write_json(
            'resume_20260727_010203_000002.json',
            {
                'saved_update_step': int(rows[1]['update_step']),
                'saved_env_steps': int(rows[1]['env_steps']),
            },
        )
        state = self.fixture._default_shapley_state()
        state.update(
            {
                'observed_event_count': 100,
                'selected_event_count': 1,
                'selected_ratio': 0.01,
            }
        )
        state['cumulative'].update(
            {
                'results_completed': 1,
                'reproduction_passed': 1,
                'samples_inserted': 1,
            }
        )
        self.fixture.write_shapley_checkpoint(state)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertTrue(payload['ready'])
        self.assertEqual(payload['shapley']['selected'], 2)
        self.assertEqual(payload['shapley']['completed'], 2)
        self.assertTrue(
            payload['shapley'][
                'selected_result_accounting_closed'
            ]
        )

        state['selected_event_count'] = 2
        self.fixture.write_shapley_checkpoint(state)
        failed = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )
        self.assertFalse(failed['ready'])
        self.assertFalse(
            failed['shapley'][
                'selected_result_accounting_closed'
            ]
        )

    def test_shapley_terminal_drop_closes_audit_but_blocks_stage(self):
        rows = self.fixture.read_metrics()
        rows[-1]['shapley_events_selected'] = '1'
        rows[-1]['shapley_terminal_dropped'] = '1'
        self.fixture.write_metrics(rows)
        state = self.fixture._default_shapley_state()
        state.update(
            {
                'selected_event_count': 1,
                'selected_ratio': 1 / 300,
            }
        )
        state['cumulative']['selected_terminal_dropped'] = 1
        self.fixture.write_shapley_checkpoint(state)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        self.assertFalse(
            payload['shapley'][
                'selected_result_accounting_closed'
            ]
        )
        self.assertTrue(
            payload['shapley'][
                'selected_terminal_accounting_closed'
            ]
        )
        self.assertEqual(
            payload['shapley'][
                'selected_result_accounting_delta'
            ],
            1,
        )
        self.assertIn(
            'shapley_stage_evidence_and_shutdown',
            payload['required_failures'],
        )

    def test_checkpoint_replay_manifests_must_match_config(self):
        checkpoint = self.fixture._checkpoint_payload(
            self.fixture._default_shapley_state()
        )
        components = checkpoint['component_snapshots']
        components['replay_buffer']['manifest']['capacity'] += 1
        components['causal_replay_buffer']['manifest']['capacity'] += 1
        self.fixture.write_checkpoint_payload(checkpoint)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        check = self._check(
            payload,
            'checkpoint_replay_manifests_match_config',
        )
        self.assertFalse(check['passed'])
        self.assertIn(
            'capacity',
            check['details']['replay_validation'][
                'config_mismatches'
            ],
        )
        self.assertIn(
            'capacity',
            check['details']['causal_replay_validation'][
                'config_mismatches'
            ],
        )

    def test_formal_stage_requires_real_shapley_selection(self):
        fixture = SyntheticTrainingRun(
            self.root,
            name='synthetic_formal',
            total_updates=250_000,
        )
        payload = audit_training_run(
            fixture.run_dir,
            stage='formal',
        )

        self.assertFalse(payload['ready'])
        self.assertIn(
            'shapley_stage_evidence_and_shutdown',
            payload['required_failures'],
        )

    def test_malformed_shutdown_and_nonfinite_checkpoint_are_gate_failures(self):
        self.fixture._write_json(
            'counterfactual_shutdown.json',
            [],
        )
        checkpoint = self.fixture._checkpoint_payload(
            self.fixture._default_shapley_state()
        )
        checkpoint['training_state']['update_step'] = float('nan')
        self.fixture.write_checkpoint_payload(checkpoint)

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )
        serialized = json.dumps(payload, allow_nan=False)

        self.assertTrue(serialized)
        self.assertFalse(payload['ready'])
        self.assertIn(
            'counterfactual_shutdown_drained',
            payload['required_failures'],
        )
        checkpoint_check = self._check(
            payload,
            'latest_checkpoint_complete_and_finite',
        )
        self.assertFalse(checkpoint_check['passed'])
        self.assertIn(
            'invalid finite number',
            ' '.join(checkpoint_check['details']['finite_errors']),
        )

    def test_cli_writes_atomic_report_outside_run_without_mutating_run(self):
        before = {
            path.relative_to(self.fixture.run_dir): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in self.fixture.run_dir.rglob('*')
            if path.is_file()
        }
        output_path = self.root / 'reports' / 'readiness.json'

        with redirect_stdout(io.StringIO()):
            exit_code = main(
                (
                    '--run-dir',
                    str(self.fixture.run_dir),
                    '--stage',
                    '5k',
                    '--baseline-config',
                    str(self.fixture.baseline_config),
                    '--output',
                    str(output_path),
                )
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output_path.read_text(encoding='utf-8'))
        self.assertTrue(payload['ready'])
        after = {
            path.relative_to(self.fixture.run_dir): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in self.fixture.run_dir.rglob('*')
            if path.is_file()
        }
        self.assertEqual(after, before)
        self.assertEqual(
            list(output_path.parent.glob('*.tmp')),
            [],
        )

        forbidden = self.fixture.run_dir / 'readiness.json'
        with redirect_stdout(io.StringIO()):
            forbidden_exit_code = main(
                (
                    '--run-dir',
                    str(self.fixture.run_dir),
                    '--stage',
                    '5k',
                    '--baseline-config',
                    str(self.fixture.baseline_config),
                    '--output',
                    str(forbidden),
                )
            )
        self.assertEqual(forbidden_exit_code, 2)
        self.assertFalse(forbidden.exists())

    def test_failure_latest_causes_exit_one_and_is_summarized(self):
        self.fixture._write_json(
            'failure_latest.json',
            {
                'exception_type': 'FloatingPointError',
                'update_step': 4,
            },
        )

        payload = audit_training_run(
            self.fixture.run_dir,
            stage='5k',
        )

        self.assertFalse(payload['ready'])
        self.assertEqual(payload['exit_code'], 1)
        check = self._check(payload, 'no_failure_latest')
        self.assertEqual(
            check['details']['failure_detail']['exception_type'],
            'FloatingPointError',
        )


if __name__ == '__main__':
    unittest.main()
