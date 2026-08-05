"""第一版 GNN-DQN 模型、Replay、Learner 和扩容契约测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from daxigua.rl.autoscale import AdaptiveScaleController
from daxigua.rl.checkpoint import load_checkpoint, save_checkpoint_atomic
from daxigua.rl.config import (
    AutoScaleConfig,
    DashboardConfig,
    DqnConfig,
    EvaluationConfig,
    ModelConfig,
    ReplayConfig,
    TrainingConfig,
)
from daxigua.rl.evaluation import evaluate_policy
from daxigua.rl.learner import DqnLearner
from daxigua.rl.model import BaselineGnnDqn
from daxigua.rl.observations import TensorState
from daxigua.rl.replay import GpuReplayBuffer
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
            )
            payload = torch.load(output, weights_only=False)
        self.assertEqual(summary.episodes, 2)
        self.assertEqual(details['recorded_trajectory_episodes'], 2)
        self.assertEqual(payload['episodes'], 2)
        self.assertEqual(len(payload['frames']), 1)
        self.assertTrue(bool(payload['frames'][0]['terminal'].all()))


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


class AutoScaleAndCheckpointTest(unittest.TestCase):
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
        self.assertEqual(loaded['progress']['transitions'], 8)
        self.assertFalse(
            loaded['replay_metadata']['replay_saved_in_checkpoint']
        )


if __name__ == '__main__':
    unittest.main()
