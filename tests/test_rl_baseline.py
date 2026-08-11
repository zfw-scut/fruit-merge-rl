"""第一版 GNN-DQN 模型、Replay、Learner 和扩容契约测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import json
import random
import unittest

import torch

from daxigua.rl.autoscale import AdaptiveScaleController
from daxigua.rl.checkpoint import (
    initialize_learner_weights,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
)
from daxigua.rl.config import (
    AutoScaleConfig,
    DashboardConfig,
    DqnConfig,
    EvaluationConfig,
    ModelConfig,
    RewardConfig,
    ReplayConfig,
    TrainingConfig,
)
from daxigua.rl.curves import (
    existing_curve_metadata,
    render_training_curve_snapshot,
)
from daxigua.rl.evaluation import (
    evaluate_policy,
    select_critical_episodes,
)
from daxigua.rl.learner import DqnLearner
from daxigua.rl.model import BaselineGnnDqn
from daxigua.rl.monitoring import (
    _DASHBOARD_HTML,
    _DashboardState,
    _completed_dashboard_snapshot,
)
from daxigua.rl.observations import TensorState
from daxigua.rl.replay import GpuReplayBuffer
from daxigua.rl.trainer import (
    BaselineTrainer,
    _bounded_stage_thresholds,
    _lower_unreachable_prewarm_targets,
    epsilon_at_transition,
    training_simulator_config,
)
from daxigua.rl.viewer import (
    load_viewer_model,
    viewer_simulator_config,
)
from daxigua.simulator import SimulatorConfig, TensorVectorSimulator


def _small_model_config():
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
    )


class ObservationAndModelTest(unittest.TestCase):
    def setUp(self):
        self.simulator = TensorVectorSimulator(
            2,
            config=SimulatorConfig(
                max_fruits=64, use_cuda_extension=False
            ),
            device='cpu',
        )

    def test_observation_exposes_danger_progress_and_line_state(self):
        self.simulator.active[0, 0] = True
        self.simulator.levels[0, 0] = 1
        self.simulator.physics_radii[0, 0] = 20.0
        self.simulator.positions[0, 0] = torch.tensor((100.0, 260.0))
        self.simulator.fail_frames[0] = (
            self.simulator.config.danger_frame_limit // 2
        )
        observation = self.simulator.observe()
        self.assertTrue(bool(observation.over_danger_line[0]))
        self.assertAlmostEqual(
            float(observation.danger_progress[0]), 0.5, places=2
        )
        clone = observation.clone()
        self.assertTrue(torch.equal(
            clone.over_danger_line, observation.over_danger_line
        ))

    def test_model_outputs_21_finite_q_values(self):
        self.simulator.active[:, 0] = True
        self.simulator.levels[:, 0] = 2
        self.simulator.physics_radii[:, 0] = 30.0
        self.simulator.positions[:, 0, 0] = torch.tensor((120.0, 430.0))
        self.simulator.positions[:, 0, 1] = 900.0
        state = TensorState.from_observation(
            self.simulator.observe(), physics_fps=120
        )
        model = BaselineGnnDqn(_small_model_config())
        q_values = model(state)
        self.assertEqual(q_values.shape, (2, 21))
        self.assertTrue(bool(torch.isfinite(q_values).all()))

    def test_evaluation_can_export_complete_decision_trajectories(self):
        model = BaselineGnnDqn(_small_model_config())
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'trajectories.pt'
            episode_index = Path(directory) / 'episode_index.pt'
            summary, details = evaluate_policy(
                model,
                physics_fps=30,
                episodes=2,
                parallel_envs=2,
                device='cpu',
                seed_base=1234,
                max_episode_drops=1,
                trajectory_output_path=output,
                trajectory_episodes=2,
                episode_index_output_path=episode_index,
            )
            payload = torch.load(output, weights_only=False)
            index = torch.load(episode_index, weights_only=False)
        self.assertEqual(summary.episodes, 2)
        self.assertEqual(details['recorded_trajectory_episodes'], 2)
        self.assertEqual(payload['episodes'], 2)
        self.assertEqual(len(payload['frames']), 1)
        self.assertTrue(bool(payload['frames'][0]['terminal'].all()))
        self.assertEqual(index['format_version'], 2)
        self.assertEqual(index['actions'].shape, (2, 1))
        self.assertEqual(index['source_level_merge_counts'].shape, (2, 12))


class EpsilonAndEventSelectionTest(unittest.TestCase):
    def test_piecewise_epsilon_interpolates_and_holds_last_value(self):
        config = DqnConfig(epsilon_schedule=(
            (0, 1.0),
            (100, 0.2),
            (200, 0.05),
        ))
        self.assertAlmostEqual(epsilon_at_transition(config, 50, 200), 0.6)
        self.assertAlmostEqual(epsilon_at_transition(config, 150, 200), 0.125)
        self.assertAlmostEqual(epsilon_at_transition(config, 999, 200), 0.05)

    def test_piecewise_epsilon_rejects_unsorted_points(self):
        with self.assertRaises(ValueError):
            DqnConfig(epsilon_schedule=((0, 1.0), (100, 0.2), (50, 0.1)))

    def test_critical_selection_keeps_event_and_density_strata(self):
        episodes = 12
        index = {
            'scores': torch.tensor(
                (1000, 1500, 2200, 3500, 7000, 7500,
                 8000, 8500, 9500, 10000, 11000, 4500)
            ),
            'created_level_counts': torch.zeros(episodes, 12),
            'source_level_merge_counts': torch.zeros(episodes, 12),
            'final_level_counts': torch.zeros(episodes, 12),
        }
        index['created_level_counts'][8, 11] = 1
        index['created_level_counts'][9, 11] = 1
        index['source_level_merge_counts'][9, 11] = 1
        index['final_level_counts'][7, 9] = 2
        selected, reasons = select_critical_episodes(
            index, 10, score_bin_width=1000, seed=7
        )
        self.assertEqual(selected.numel(), 10)
        self.assertIn(9, selected.tolist())
        self.assertIn(8, selected.tolist())
        self.assertIn(7, selected.tolist())
        self.assertEqual(len(reasons), 10)


class ReplayAndLearnerTest(unittest.TestCase):
    def _state(self, batch=2):
        simulator = TensorVectorSimulator(
            batch,
            config=SimulatorConfig(
                max_fruits=64, use_cuda_extension=False
            ),
            device='cpu',
        )
        return TensorState.from_observation(
            simulator.observe(), physics_fps=30
        )

    def test_replay_wraps_and_samples_raw_tensor_states(self):
        state = self._state()
        replay = GpuReplayBuffer(
            4, max_fruits=64, device='cpu', physics_fps=30
        )
        for _ in range(3):
            replay.append(
                state,
                torch.tensor((0, 1)),
                torch.tensor((0.0, 1.0)),
                state,
                torch.tensor((False, True)),
                stages=torch.tensor((1, 3)),
            )
        self.assertEqual(len(replay), 4)
        self.assertEqual(replay.cursor, 2)
        batch = replay.sample(3)
        self.assertEqual(batch.current.positions.shape, (3, 64, 2))
        self.assertEqual(batch.action.shape, (3,))
        self.assertTrue(set(batch.stage.tolist()).issubset({1, 3}))

    def test_double_dqn_update_and_target_sync_run(self):
        state = self._state(batch=4)
        replay = GpuReplayBuffer(
            8, max_fruits=64, device='cpu', physics_fps=30
        )
        replay.append(
            state,
            torch.tensor((0, 1, 2, 3)),
            torch.tensor((0.0, 0.5, 0.0, 1.0)),
            state,
            torch.tensor((False, False, True, True)),
        )
        model = BaselineGnnDqn(_small_model_config())
        learner = DqnLearner(
            model,
            DqnConfig(
                target_update_interval=1,
                use_bfloat16=False,
                fused_adam=False,
            ),
        )
        metrics = learner.update(replay, 2)
        self.assertEqual(metrics['update_count'], 1)
        self.assertTrue(metrics['target_synced'])
        self.assertTrue(bool(torch.isfinite(metrics['loss'])))


class RewardTrainingConfigTest(unittest.TestCase):
    def test_reward_modes_are_explicit_and_validate_weights(self):
        self.assertEqual(RewardConfig(kind='score_v1').kind, 'score_v1')
        self.assertEqual(RewardConfig().kind, 'spatial_v2_1')
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            RewardConfig(queue_weights=(0.5, 0.5, 0.5))

    def test_baseline_and_reward_v2_toml_keep_separate_reward_modes(self):
        project_root = Path(__file__).resolve().parents[1]
        baseline = TrainingConfig.from_toml(
            project_root / 'configs' / 'gnn_dqn_baseline.toml'
        )
        reward_v2 = TrainingConfig.from_toml(
            project_root / 'configs' / 'gnn_dqn_reward_v2.toml'
        )
        reward_v2_1 = TrainingConfig.from_toml(
            project_root / 'configs' / 'gnn_dqn_reward_v2_1.toml'
        )
        baseline_scale_v1 = TrainingConfig.from_toml(
            project_root / 'configs' / 'gnn_dqn_baseline_scale_v1.toml'
        )

        self.assertEqual(baseline.reward.kind, 'score_v1')
        self.assertEqual(reward_v2.reward.kind, 'spatial_v2')
        self.assertEqual(reward_v2_1.reward.kind, 'spatial_v2_1')
        self.assertEqual(reward_v2.reward.queue_weights, (0.5, 0.3, 0.2))
        self.assertEqual(reward_v2_1.replay.batch_size, 1792)
        self.assertEqual(baseline_scale_v1.reward.kind, 'score_v1')
        self.assertEqual(baseline_scale_v1.model.message_layers, 5)
        self.assertEqual(baseline_scale_v1.model.hidden_dim, 128)
        self.assertEqual(baseline_scale_v1.total_transitions, 16_000_000)
        self.assertEqual(baseline_scale_v1.max_wall_seconds, 7200.0)

        baseline_scale_v1_l4 = TrainingConfig.from_toml(
            project_root / 'configs' / 'gnn_dqn_baseline_scale_v1_l4.toml'
        )
        self.assertEqual(baseline_scale_v1_l4.reward.kind, 'score_v1')
        self.assertEqual(baseline_scale_v1_l4.model.message_layers, 4)
        self.assertEqual(baseline_scale_v1_l4.model.hidden_dim, 128)
        self.assertEqual(baseline_scale_v1_l4.total_transitions, 16_000_000)
        self.assertEqual(baseline_scale_v1_l4.replay.batch_size, 768)

        fast_l5 = TrainingConfig.from_toml(
            project_root / 'configs' / 'gnn_dqn_baseline_l5_fast_24m.toml'
        )
        slow_l5 = TrainingConfig.from_toml(
            project_root / 'configs' / 'gnn_dqn_baseline_l5_slow_24m.toml'
        )
        self.assertEqual(fast_l5.model.message_layers, 5)
        self.assertEqual(slow_l5.total_transitions, 24_000_000)
        self.assertEqual(fast_l5.dqn.epsilon_schedule[1], (6_400_000, 0.05))
        self.assertEqual(slow_l5.dqn.epsilon_schedule[1], (6_400_000, 0.20))
        self.assertEqual(
            fast_l5.evaluation.seed_base, slow_l5.evaluation.seed_base
        )

        long_fast_l5 = TrainingConfig.from_toml(
            project_root
            / 'configs'
            / 'gnn_dqn_baseline_l5_fast_128m_resume.toml'
        )
        self.assertEqual(long_fast_l5.total_transitions, 128_000_000)
        self.assertEqual(long_fast_l5.max_envs, 1536)
        self.assertEqual(long_fast_l5.replay.batch_size, 384)
        self.assertEqual(
            long_fast_l5.dqn.epsilon_schedule,
            fast_l5.dqn.epsilon_schedule,
        )
        self.assertEqual(long_fast_l5.evaluation.fast_interval_transitions,
                         8_000_000)
        self.assertEqual(
            long_fast_l5.evaluation.accurate_milestones,
            [48_000_000, 64_000_000, 96_000_000],
        )
        self.assertEqual(long_fast_l5.evaluation.seed_base, 42_000_000)
        self.assertEqual(long_fast_l5.model.policy_head_count, 1)
        self.assertFalse(long_fast_l5.model.action_effect_enabled)
        self.assertFalse(long_fast_l5.dqn.active_learning_enabled)
        self.assertFalse(long_fast_l5.branch_learning.enabled)

        auxiliary_l5 = TrainingConfig.from_toml(
            project_root / 'configs' / 'gnn_dqn_auxiliary_action_l5_24m.toml'
        )
        self.assertEqual(auxiliary_l5.dqn.epsilon_decay_fraction, 0.40)
        self.assertEqual(
            auxiliary_l5.dqn.epsilon_schedule,
            ((0, 1.0), (6_400_000, 0.05), (24_000_000, 0.05)),
        )
        self.assertFalse(auxiliary_l5.dqn.active_learning_enabled)
        self.assertTrue(auxiliary_l5.model.structured_contact_enabled)
        self.assertEqual(auxiliary_l5.replay.batch_size, 256)
        self.assertTrue(auxiliary_l5.branch_learning.enabled)
        self.assertEqual(
            auxiliary_l5.branch_learning.transition_budget, 4_000_000
        )
        self.assertEqual(auxiliary_l5.branch_learning.actions_per_state, 4)
        self.assertEqual(auxiliary_l5.branch_learning.loss_weight, 0.25)
        self.assertAlmostEqual(
            auxiliary_l5.branch_learning.transition_budget
            / auxiliary_l5.branch_learning.actions_per_state
            / auxiliary_l5.total_transitions,
            1 / 24,
        )
        self.assertAlmostEqual(
            auxiliary_l5.branch_learning.learner_batch_size
            / (
                auxiliary_l5.replay.batch_size
                + auxiliary_l5.branch_learning.learner_batch_size
            ),
            0.20,
        )
        self.assertAlmostEqual(
            auxiliary_l5.branch_learning.loss_weight
            / (1.0 + auxiliary_l5.branch_learning.loss_weight),
            0.20,
        )

        transfer_120fps = TrainingConfig.from_toml(
            project_root
            / 'configs'
            / 'gnn_dqn_auxiliary_action_structured_120fps_transfer_16m.toml'
        )
        self.assertEqual(transfer_120fps.training_physics_fps, 120)
        self.assertEqual(transfer_120fps.total_transitions, 16_000_000)
        self.assertEqual(transfer_120fps.dqn.learning_rate, 3e-5)
        self.assertEqual(
            transfer_120fps.dqn.epsilon_schedule,
            ((0, 0.10), (4_000_000, 0.05), (16_000_000, 0.05)),
        )
        self.assertTrue(transfer_120fps.model.structured_contact_enabled)
        self.assertFalse(transfer_120fps.branch_learning.enabled)


class AutoScaleAndCheckpointTest(unittest.TestCase):
    def test_training_physics_profile_is_explicit_and_validated(self):
        with self.assertRaisesRegex(ValueError, '30 or 120'):
            TrainingConfig(training_physics_fps=60)
        fast = training_simulator_config(TrainingConfig(), 'cpu')
        accurate = training_simulator_config(
            TrainingConfig(training_physics_fps=120), 'cpu'
        )
        self.assertEqual(fast.physics_fps, 30)
        self.assertEqual(accurate.physics_fps, 120)
        self.assertTrue(fast.drop_fast_forward)
        self.assertTrue(accurate.drop_fast_forward)

    def test_censored_stage_quantiles_use_the_pilot_window_quartiles(self):
        self.assertEqual(
            _bounded_stage_thresholds((128, 128, 128), 128),
            (32, 64, 96),
        )
        self.assertEqual(
            _bounded_stage_thresholds((8, 20, 80), 128),
            (8, 20, 80),
        )

    def test_unreachable_prewarm_targets_are_lowered_before_retry(self):
        targets = torch.tensor((12, 20, 7, 4), dtype=torch.int64)
        survived = torch.tensor((8, 30, 1, 0), dtype=torch.int64)
        failed = torch.tensor((True, False, True, False))

        adjusted = _lower_unreachable_prewarm_targets(
            targets, survived, failed
        )

        self.assertEqual(adjusted.tolist(), [7, 20, 0, 4])

    def test_stage_pilot_environment_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'stage_pilot_envs'):
            TrainingConfig(stage_pilot_envs=0)

    def test_stage_pilot_drop_horizon_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'stage_pilot_max_drops'):
            TrainingConfig(stage_pilot_max_drops=0)

    def test_autoscale_trials_and_commits_only_after_throughput_gain(self):
        config = AutoScaleConfig(
            candidate_envs=(2, 4),
            observation_seconds=0.0,
            trial_seconds=0.0,
            cooldown_seconds=0.0,
            minimum_throughput_gain=0.08,
        )
        controller = AdaptiveScaleController(
            config, initial_envs=2, maximum_envs=4
        )
        resources = {
            'gpu_utilization': 10.0,
            'cpu_utilization': 10.0,
            'gpu_memory_used_mb': 100.0,
            'gpu_memory_total_mb': 1000.0,
        }
        self.assertIsNone(controller.observe(resources, 100.0, now=1.0))
        trial = controller.observe(resources, 100.0, now=2.0)
        self.assertEqual((trial.action, trial.target_envs), ('trial', 4))
        commit = controller.observe(resources, 120.0, now=3.0)
        self.assertEqual((commit.action, commit.target_envs), ('commit', 4))

    def test_checkpoint_round_trip_keeps_progress_and_omits_replay(self):
        model = BaselineGnnDqn(_small_model_config())
        learner = DqnLearner(
            model,
            DqnConfig(use_bfloat16=False, fused_adam=False),
        )
        config = TrainingConfig(
            device='cpu',
            max_envs=2,
            active_envs=2,
            total_transitions=8,
            model=_small_model_config(),
            replay=ReplayConfig(
                capacity=8,
                batch_size=2,
                warmup_transitions=2,
                warmup_stage_ratios=(1.0, 0.0, 0.0, 0.0),
            ),
            evaluation=EvaluationConfig(
                periodic_episodes=2,
                final_episodes=2,
                parallel_envs=2,
            ),
            dashboard=DashboardConfig(enabled=False),
            autoscale=AutoScaleConfig(enabled=False),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint_atomic(
                path,
                learner=learner,
                training_config=config,
                progress={'transitions': 8},
                replay_metadata={'replay_saved_in_checkpoint': False},
            )
            loaded = load_checkpoint(path)
            viewer_model = load_viewer_model(path, device='cpu')
        self.assertEqual(loaded['progress']['transitions'], 8)
        self.assertEqual(viewer_model.progress['transitions'], 8)
        self.assertEqual(viewer_model.model_config, _small_model_config())
        self.assertEqual(viewer_model.device, torch.device('cpu'))
        self.assertFalse(
            loaded['replay_metadata']['replay_saved_in_checkpoint']
        )

    def test_weights_only_initialization_resets_training_state(self):
        model_config = _small_model_config()
        source = DqnLearner(
            BaselineGnnDqn(model_config),
            DqnConfig(use_bfloat16=False, fused_adam=False),
        )
        with torch.no_grad():
            for parameter in source.online_module.parameters():
                parameter.fill_(0.25)
            for parameter in source.target_module.parameters():
                parameter.fill_(0.75)
        source.update_count = 17
        config = TrainingConfig(
            device='cpu',
            max_envs=2,
            active_envs=2,
            total_transitions=8,
            model=model_config,
            replay=ReplayConfig(
                capacity=8,
                batch_size=2,
                warmup_transitions=2,
                warmup_stage_ratios=(1.0, 0.0, 0.0, 0.0),
            ),
            dashboard=DashboardConfig(enabled=False),
            autoscale=AutoScaleConfig(enabled=False),
        )
        target = DqnLearner(
            BaselineGnnDqn(model_config),
            DqnConfig(use_bfloat16=False, fused_adam=False),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pt'
            save_checkpoint_atomic(
                path,
                learner=source,
                training_config=config,
                progress={'transitions': 128_000_000, 'updates': 17},
                replay_metadata={'replay_saved_in_checkpoint': False},
            )
            metadata = initialize_learner_weights(
                target,
                path,
                expected_model_config=config.to_dict()['model'],
            )
        self.assertEqual(target.update_count, 0)
        self.assertFalse(target.optimizer.state)
        self.assertEqual(metadata['kind'], 'weights_only')
        self.assertEqual(
            metadata['source_progress']['transitions'], 128_000_000
        )
        for online, source_online, target_parameter in zip(
                target.online_module.parameters(),
                source.online_module.parameters(),
                target.target_module.parameters()):
            self.assertTrue(torch.equal(online, source_online))
            self.assertTrue(torch.equal(target_parameter, source_online))

    def test_120fps_training_uses_120fps_as_periodic_primary_eval(self):
        trainer = object.__new__(BaselineTrainer)
        trainer.config = TrainingConfig(
            training_physics_fps=120,
            evaluation=EvaluationConfig(
                fast_interval_transitions=4_000_000,
                accurate_milestones=(8_000_000,),
            ),
        )
        trainer.simulator_config = SimulatorConfig.high_fidelity_fast()
        trainer.transitions = 4_000_000
        trainer.completed_accurate_milestones = set()
        calls = []
        trainer._evaluate = lambda fps, episodes, transition: calls.append(
            (fps, episodes, transition)
        )
        trainer._maybe_evaluate(0)
        self.assertEqual(calls, [(120, 512, 4_000_000)])

    def test_trainer_records_weights_only_initialization_identity(self):
        model_config = _small_model_config()
        source = DqnLearner(
            BaselineGnnDqn(model_config),
            DqnConfig(use_bfloat16=False, fused_adam=False),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_config = TrainingConfig(
                run_dir=str(root / 'source'),
                device='cpu',
                max_envs=2,
                active_envs=2,
                total_transitions=8,
                model=model_config,
                replay=ReplayConfig(
                    capacity=8,
                    batch_size=2,
                    warmup_transitions=2,
                    warmup_stage_ratios=(1.0, 0.0, 0.0, 0.0),
                ),
                dashboard=DashboardConfig(enabled=False),
                autoscale=AutoScaleConfig(enabled=False),
            )
            source_path = root / 'source.pt'
            save_checkpoint_atomic(
                source_path,
                learner=source,
                training_config=source_config,
                progress={'transitions': 128_000_000},
                replay_metadata={'replay_saved_in_checkpoint': False},
            )
            target_config = TrainingConfig(
                run_dir=str(root / 'target'),
                device='cpu',
                training_physics_fps=120,
                max_envs=2,
                active_envs=2,
                total_transitions=8,
                model=model_config,
                replay=source_config.replay,
                dashboard=DashboardConfig(enabled=False),
                autoscale=AutoScaleConfig(enabled=False),
            )
            trainer = BaselineTrainer(target_config)
            try:
                metadata = trainer.initialize_from_checkpoint(source_path)
                initialization = json.loads(
                    (root / 'target' / 'initialization.json').read_text(
                        encoding='utf-8'
                    )
                )
                identity = json.loads(
                    (root / 'target' / 'run_identity.json').read_text(
                        encoding='utf-8'
                    )
                )
            finally:
                trainer.resource_sampler.close()
                trainer.dashboard.close()
        self.assertEqual(trainer.transitions, 0)
        self.assertEqual(trainer.learner.update_count, 0)
        self.assertEqual(metadata['target_training_physics_fps'], 120)
        self.assertEqual(initialization['source_training_physics_fps'], 30)
        self.assertEqual(
            identity['initialization']['source_checkpoint_sha256'],
            metadata['source_checkpoint_sha256'],
        )
        self.assertEqual(
            identity['training_simulator_config']['physics_fps'], 120
        )

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_restore_rng_state_accepts_cuda_mapped_sidecar(self):
        expected_cpu = torch.get_rng_state().clone()
        expected_cuda = [item.clone() for item in torch.cuda.get_rng_state_all()]
        state = {
            'python': random.getstate(),
            'torch_cpu': expected_cpu.to('cuda'),
            'torch_cuda': [item.to('cuda') for item in expected_cuda],
        }

        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(5678)
        restore_rng_state(state)

        self.assertTrue(torch.equal(torch.get_rng_state(), expected_cpu))
        restored_cuda = torch.cuda.get_rng_state_all()
        self.assertEqual(len(restored_cuda), len(expected_cuda))
        for actual, expected in zip(restored_cuda, expected_cuda):
            self.assertTrue(torch.equal(actual, expected))

    def test_viewer_physics_defaults_to_accurate_120_fps(self):
        model_config = _small_model_config()
        accurate = viewer_simulator_config(120, model_config, 'cuda')
        training = viewer_simulator_config(30, model_config, 'cuda')

        self.assertEqual(accurate.physics_fps, 120)
        self.assertEqual(training.physics_fps, 30)
        self.assertTrue(accurate.use_cuda_extension)
        self.assertTrue(training.use_cuda_extension)
        with self.assertRaisesRegex(ValueError, '30 or 120'):
            viewer_simulator_config(60, model_config, 'cuda')


class DashboardTest(unittest.TestCase):
    def test_dashboard_keeps_score_curves_and_uses_chinese_labels(self):
        state = _DashboardState(history_size=4)
        state.update_training({
            'transitions': 1000,
            'env_steps_per_second': 200.0,
            'updates_per_second': 0.5,
            'branch_steps_per_second': 40.0,
            'branch_aux_loss_total': 0.75,
            'branch_sample_fraction': 0.20,
            'training_window_mean_score': 123.0,
            'training_window_max_score': 456.0,
            'training_rolling_mean_score': 120.0,
        })
        history = state.snapshot()['history']
        self.assertEqual(history[0]['training_window_mean_score'], 123.0)
        self.assertEqual(history[0]['training_window_max_score'], 456.0)
        self.assertEqual(history[0]['branch_steps_per_second'], 40.0)
        self.assertEqual(history[0]['branch_aux_loss_total'], 0.75)
        self.assertEqual(history[0]['branch_sample_fraction'], 0.20)
        self.assertIn('训练效果曲线', _DASHBOARD_HTML)
        self.assertIn('窗口局均分', _DASHBOARD_HTML)
        self.assertIn('score-chart', _DASHBOARD_HTML)
        self.assertIn('Windows 11 Fluent', _DASHBOARD_HTML)
        self.assertIn('概览', _DASHBOARD_HTML)
        self.assertIn('定期保存曲线', _DASHBOARD_HTML)
        self.assertIn('curve-snapshot', _DASHBOARD_HTML)
        self.assertIn('active-rank-chart', _DASHBOARD_HTML)
        self.assertIn('gpu-resource-chart', _DASHBOARD_HTML)
        self.assertIn('训练已正常完成，面板保持在线', _DASHBOARD_HTML)
        self.assertNotIn('文件(F)', _DASHBOARD_HTML)

    def test_dashboard_keeps_lightweight_gpu_resource_history(self):
        state = _DashboardState(history_size=2)
        state.update_resources({
            'timestamp': 10.0,
            'gpu_utilization': 87.0,
            'gpu_memory_used_mb': 12.0,
            'gpu_memory_total_mb': 24.0,
        })
        resource = state.snapshot()['resource_history'][0]
        self.assertEqual(resource['gpu_utilization'], 87.0)
        self.assertEqual(resource['gpu_memory_utilization'], 50.0)

    def test_dashboard_state_exposes_curve_snapshot_metadata(self):
        state = _DashboardState(history_size=4)
        state.update_plot('training_curves', {
            'url': '/plots/training_curves.png',
            'source_last_transition': 1234,
        })
        plot = state.snapshot()['plots']['training_curves']
        self.assertEqual(plot['source_last_transition'], 1234)

    def test_completed_dashboard_reads_persistent_normal_status(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run_status.json').write_text(json.dumps({
                'phase': 'completed',
                'completion_message': '训练已正常完成',
                'transitions': 24_000_000,
                'total_transitions': 24_000_000,
            }), encoding='utf-8')
            state = _completed_dashboard_snapshot(root).snapshot()
        self.assertEqual(state['training']['phase'], 'completed')
        self.assertEqual(
            state['training']['completion_message'], '训练已正常完成'
        )

    def test_curve_snapshot_interval_must_be_positive(self):
        with self.assertRaisesRegex(
                ValueError, 'curve_snapshot_interval_seconds'):
            DashboardConfig(curve_snapshot_interval_seconds=0.0)


@unittest.skipUnless(
    importlib.util.find_spec('matplotlib') is not None,
    'matplotlib is required for curve snapshot rendering',
)
class CurveSnapshotTest(unittest.TestCase):
    def test_jsonl_metrics_generate_atomic_png_and_metadata(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            rows = [
                {
                    'transitions': transition,
                    'training_rolling_mean_score': 1000 + index * 100,
                    'training_window_mean_score': 1050 + index * 110,
                    'training_window_max_score': 2000 + index * 300,
                    'loss': 0.04 - index * 0.005,
                    'mean_abs_td_error': 0.16 - index * 0.01,
                    'env_steps_per_second': 4000 + index * 500,
                    'learner_samples_per_second': 3900 + index * 500,
                }
                for index, transition in enumerate(
                    (1_000_000, 2_000_000, 3_000_000)
                )
            ]
            metrics_path = run_dir / 'metrics.jsonl'
            metrics_path.write_text(
                ''.join(json.dumps(row) + '\n' for row in rows)
                + '{incomplete',
                encoding='utf-8',
            )
            evaluation_dir = run_dir / 'evaluations'
            evaluation_dir.mkdir()
            (evaluation_dir / 'metrics.jsonl').write_text(
                json.dumps({
                    'transition': 3_000_000,
                    'physics_fps': 120,
                    'mean_score': 1325.0,
                }) + '\n',
                encoding='utf-8',
            )

            metadata = render_training_curve_snapshot(run_dir)
            output = run_dir / 'plots' / 'training_curves.png'
            stored = existing_curve_metadata(run_dir)

            self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(metadata['source_metric_rows'], 3)
            self.assertEqual(metadata['source_evaluation_rows'], 1)
            self.assertEqual(metadata['source_last_transition'], 3_000_000)
            self.assertEqual(stored['url'], '/plots/training_curves.png')
            self.assertFalse(list((run_dir / 'plots').glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
