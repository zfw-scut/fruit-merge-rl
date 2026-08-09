"""辅助动作效果、多策略头与主动学习基础契约测试。"""

import json
from dataclasses import replace
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

import torch

from daxigua.rl.action_effects import build_action_effect_targets
from daxigua.rl.config import (
    AnalysisExportConfig,
    AutoScaleConfig,
    BranchLearningConfig,
    DashboardConfig,
    DqnConfig,
    EvaluationConfig,
    ModelConfig,
    ReplayConfig,
    RewardConfig,
    TrainingConfig,
)
from daxigua.rl.learner import DqnLearner
from daxigua.rl.model import BaselineGnnDqn
from daxigua.rl.observations import TensorState
from daxigua.rl.replay import GpuReplayBuffer
from daxigua.rl.trainer import (
    BaselineTrainer,
    active_learning_probability,
    rank_correlation_from_sums,
    ranked_active_learning_candidates,
)
from daxigua.simulator import SimulatorConfig, TensorVectorSimulator


def _model_config():
    return ModelConfig(
        hidden_dim=32,
        edge_hidden_dim=32,
        message_layers=1,
        queue_hidden_dim=16,
        queue_layers=1,
        level_embedding_dim=8,
        max_neighbors=4,
        nearest_neighbors=2,
        motion_neighbors=1,
        vertical_neighbors_per_direction=1,
        action_key_fruits=2,
        policy_head_count=3,
        action_effect_enabled=True,
    )


def _training_config(directory, device):
    return TrainingConfig(
        run_dir=directory,
        device=device,
        seed=321,
        max_envs=2,
        active_envs=2,
        total_transitions=4,
        log_interval_seconds=0.01,
        max_episode_drops=4,
        stage_pilot_envs=2,
        stage_pilot_max_drops=2,
        model=_model_config(),
        dqn=DqnConfig(
            use_bfloat16=False,
            fused_adam=False,
            target_update_interval=1,
            auxiliary_loss_weight=0.2,
            active_learning_enabled=True,
            epsilon_start=0.0,
            epsilon_end=0.0,
            active_learning_start_epsilon=0.5,
            active_learning_full_epsilon=0.0,
            active_learning_max_probability=1.0,
            active_learning_top_k=2,
        ),
        reward=RewardConfig(kind='score_v1'),
        replay=ReplayConfig(
            capacity=4,
            batch_size=2,
            warmup_transitions=2,
            warmup_stage_ratios=(1.0, 0.0, 0.0, 0.0),
        ),
        evaluation=EvaluationConfig(
            periodic_episodes=1,
            final_episodes=1,
            parallel_envs=1,
            max_episode_drops=1,
        ),
        analysis=AnalysisExportConfig(
            transition_sample_size=0,
            trajectory_episodes=0,
            critical_event_episodes=0,
        ),
        dashboard=DashboardConfig(enabled=False),
        autoscale=AutoScaleConfig(enabled=False),
    )


def _branch_training_config(directory, device):
    config = _training_config(directory, device)
    return replace(
        config,
        total_transitions=8,
        dqn=replace(
            config.dqn,
            active_learning_enabled=False,
        ),
        replay=replace(
            config.replay,
            capacity=8,
            batch_size=2,
            warmup_transitions=2,
        ),
        branch_learning=BranchLearningConfig(
            enabled=True,
            transition_budget=4,
            start_transition=0,
            actions_per_state=2,
            simulator_batch_size=4,
            replay_capacity=4,
            replay_warmup=2,
            learner_batch_size=1,
            loss_weight=0.5,
        ),
    )


class ActionEffectLearningTest(unittest.TestCase):
    def _simulator(self, batch=4):
        return TensorVectorSimulator(
            batch,
            config=SimulatorConfig.training_fast(
                max_fruits=64,
                use_cuda_extension=False,
                track_action_effects=True,
            ),
            device='cpu',
        )

    def _transition(self, batch=4):
        simulator = self._simulator(batch)
        current = TensorState.from_observation(
            simulator.observe(), physics_fps=30, clone=True
        )
        current_count = current.active.sum(dim=1)
        current_danger = current.danger_progress.clone()
        actions = torch.arange(batch).remainder(21)
        result = simulator.step(actions)
        next_state = TensorState.from_observation(
            result.observation, physics_fps=30
        )
        targets = build_action_effect_targets(
            current,
            next_state,
            result,
            max_physics_frames=simulator.config.max_physics_frames,
            current_fruit_count=current_count,
            current_danger_progress=current_danger,
        )
        return current, actions, result, next_state, targets

    def test_first_contact_and_q0_final_state_are_recorded(self):
        _current, _actions, result, _next_state, targets = self._transition(2)
        effects = result.physics.action_effects
        self.assertIsNotNone(effects)
        self.assertTrue(bool((effects.first_contact_age_frames >= 0).all()))
        self.assertTrue(bool(targets.contact_type_bits.any(dim=1).all()))
        self.assertTrue(bool(targets.final_exists.all()))
        self.assertEqual(targets.final_state.shape, (2, 5))

    def test_first_three_generated_fruits_use_global_event_order(self):
        current, _actions, result, next_state, _targets = self._transition(1)
        events = result.physics.merge_events
        events.count[0] = 4
        events.new_levels[0, :4] = torch.tensor((4, 0, 7, 9))
        events.positions[0, :4] = torch.tensor((
            (100.0, 300.0),
            (150.0, 400.0),
            (200.0, 500.0),
            (250.0, 600.0),
        ))
        effects = result.physics.action_effects
        effects.q0_participated.zero_()
        targets = build_action_effect_targets(
            current,
            next_state,
            result,
            current_fruit_count=torch.zeros(1, dtype=torch.int64),
            current_danger_progress=torch.zeros(1),
        )
        self.assertEqual(targets.generation_exists.tolist(), [[True] * 3])
        self.assertEqual(targets.generation_level.tolist(), [[4, 7, 9]])
        self.assertFalse(bool(targets.q0_participated[0]))

    def test_bootstrap_ensemble_and_auxiliary_update_run(self):
        current, actions, result, next_state, targets = self._transition(4)
        model = BaselineGnnDqn(_model_config())
        output = model(current, True, True)
        self.assertEqual(output.q_values.shape, (4, 21))
        self.assertEqual(output.head_q_values.shape, (4, 3, 21))
        self.assertTrue(torch.allclose(
            output.q_values, output.head_q_values.mean(dim=1)
        ))

        replay = GpuReplayBuffer(
            8,
            max_fruits=64,
            device='cpu',
            physics_fps=30,
            policy_head_count=3,
            bootstrap_probability=0.5,
            action_effects_enabled=True,
        )
        replay.append(
            current,
            actions,
            result.physics.score_delta.float() / 66.0,
            next_state,
            result.physics.done,
            action_effects=targets,
        )
        sampled = replay.sample(4)
        self.assertEqual(sampled.bootstrap_mask.shape, (4, 3))
        self.assertTrue(bool(sampled.bootstrap_mask.any(dim=1).all()))
        self.assertIsNotNone(sampled.action_effects)

        learner = DqnLearner(
            model,
            DqnConfig(
                target_update_interval=1,
                use_bfloat16=False,
                fused_adam=False,
                auxiliary_loss_weight=0.2,
            ),
        )
        metrics = learner.update(replay, 4)
        for name in (
                'loss', 'dqn_loss', 'aux_loss_total',
                'aux_loss_first_contact', 'policy_disagreement'):
            self.assertTrue(bool(torch.isfinite(metrics[name])))

        branch_metrics = learner.update(
            replay,
            2,
            branch_replay=replay,
            branch_batch_size=1,
            branch_loss_weight=0.5,
        )
        self.assertAlmostEqual(
            float(branch_metrics['branch_sample_fraction']), 1 / 3
        )
        self.assertTrue(bool(torch.isfinite(
            branch_metrics['branch_dqn_loss']
        )))
        self.assertTrue(bool(torch.isfinite(
            branch_metrics['branch_aux_loss_total']
        )))

    def test_rank_fusion_orders_candidates_without_value_scaling(self):
        q_values = torch.tensor(((1.0, 0.8, 0.7, 0.6),))
        uncertainty = torch.tensor(((0.1, 0.4, 0.3, 0.2),))
        candidates, value_ranks, uncertainty_ranks = (
            ranked_active_learning_candidates(q_values, uncertainty, 3)
        )
        self.assertEqual(value_ranks.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(uncertainty_ranks.tolist(), [[4, 1, 2, 3]])
        self.assertEqual(candidates.tolist(), [[1, 2, 0]])

    def test_active_probability_and_rank_correlation_have_clear_semantics(self):
        config = DqnConfig(
            active_learning_start_epsilon=0.5,
            active_learning_full_epsilon=0.05,
            active_learning_max_probability=0.4,
        )
        self.assertEqual(active_learning_probability(config, 0.5), 0.0)
        self.assertAlmostEqual(active_learning_probability(config, 0.275), 0.2)
        self.assertEqual(active_learning_probability(config, 0.05), 0.4)
        self.assertAlmostEqual(
            rank_correlation_from_sums((3, 6, 12, 14, 56, 28)),
            1.0,
        )

    def test_trainer_writes_auxiliary_transitions_end_to_end(self):
        with TemporaryDirectory() as directory:
            trainer = BaselineTrainer(_training_config(directory, 'cpu'))
            result = trainer.run(final_evaluation=False)
            rows = [
                json.loads(line)
                for line in (Path(directory) / 'metrics.jsonl').read_text(
                    encoding='utf-8'
                ).splitlines()
            ]
            status = json.loads(
                (Path(directory) / 'run_status.json').read_text(
                    encoding='utf-8'
                )
            )
        self.assertEqual(result['transitions'], 4)
        self.assertGreaterEqual(result['updates'], 1)
        self.assertTrue(any(
            'active_selected_rank_correlation' in row for row in rows
        ))
        self.assertFalse(any(
            any(key.startswith('shadow_bonus_') for key in row) for row in rows
        ))
        self.assertTrue(any(
            'active_learning_effective_action_fraction' in row
            for row in rows
        ))
        self.assertEqual(status['phase'], 'completed')

    def test_single_step_branch_learning_preserves_parent_budget(self):
        with TemporaryDirectory() as directory:
            trainer = BaselineTrainer(
                _branch_training_config(directory, 'cpu')
            )
            result = trainer.run(final_evaluation=False)
            rows = [
                json.loads(line)
                for line in (Path(directory) / 'metrics.jsonl').read_text(
                    encoding='utf-8'
                ).splitlines()
            ]
            resumed = BaselineTrainer(
                _branch_training_config(directory, 'cpu')
            )
            resumed.resume(Path(directory) / 'checkpoints' / 'final.pt')
        self.assertEqual(result['transitions'], 8)
        self.assertEqual(result['branch_transitions'], 4)
        self.assertEqual(result['branch_source_states'], 2)
        self.assertEqual(result['branch_progress_fraction'], 1.0)
        self.assertEqual(len(trainer.branch_replay), 4)
        self.assertEqual(resumed.branch_transitions, 4)
        self.assertEqual(resumed.branch_replay_training_threshold, 1)
        self.assertTrue(torch.equal(
            trainer.branch_generator.get_state(),
            resumed.branch_generator.get_state(),
        ))
        self.assertTrue(all(
            row.get('active_learning_action_fraction', 0.0) == 0.0
            for row in rows
        ))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_single_step_branch_learning_runs_end_to_end(self):
        with TemporaryDirectory() as directory:
            trainer = BaselineTrainer(
                _branch_training_config(directory, 'cuda')
            )
            result = trainer.run(final_evaluation=False)
            resumed = BaselineTrainer(
                _branch_training_config(directory, 'cuda')
            )
            resumed.resume(Path(directory) / 'checkpoints' / 'final.pt')
        self.assertEqual(result['transitions'], 8)
        self.assertEqual(result['branch_transitions'], 4)
        self.assertEqual(result['branch_source_states'], 2)
        self.assertTrue(torch.equal(
            trainer.branch_generator.get_state(),
            resumed.branch_generator.get_state(),
        ))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_trainer_runs_auxiliary_pipeline_end_to_end(self):
        with TemporaryDirectory() as directory:
            trainer = BaselineTrainer(_training_config(directory, 'cuda'))
            result = trainer.run(final_evaluation=False)
        self.assertEqual(result['transitions'], 4)
        self.assertGreaterEqual(result['updates'], 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_kernel_records_contact_and_q0_lineage(self):
        simulator = TensorVectorSimulator(
            8,
            config=SimulatorConfig.training_fast(
                max_fruits=64,
                use_cuda_extension=True,
                track_action_effects=True,
            ),
            device='cuda',
        )
        simulator.reset(
            fruit_queue=torch.ones((8, 4), dtype=torch.int64, device='cuda')
        )
        actions = torch.full((8,), 10, dtype=torch.int64, device='cuda')
        first = simulator.step(actions)
        second = simulator.step(actions)
        torch.cuda.synchronize()
        self.assertTrue(bool(
            (first.physics.action_effects.first_contact_age_frames >= 0).all()
        ))
        self.assertTrue(bool(
            second.physics.action_effects.q0_participated.all()
        ))
        self.assertTrue(bool(
            (second.physics.action_effects.q0_lineage_depth >= 1).all()
        ))


if __name__ == '__main__':
    unittest.main()
