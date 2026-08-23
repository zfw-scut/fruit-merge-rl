"""从零128M基线和真实短跑吞吐标定的契约测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import torch

from daxigua.rl.checkpoint import save_checkpoint_atomic
from daxigua.rl.config import (
    AnalysisExportConfig,
    AutoScaleConfig,
    DashboardConfig,
    DqnConfig,
    ModelConfig,
    ReplayConfig,
    TrainingConfig,
)
from daxigua.rl.learner import DqnLearner
from daxigua.rl.model import BaselineGnnDqn
from daxigua.rl.trainer import BaselineTrainer
from tools.calibrate_real_training import (
    build_trial_config,
    environment_bracket,
    select_recommendation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = (
    PROJECT_ROOT
    / 'configs'
    / 'sab-full-fall-edge18-128m-b16m-r1.toml'
)


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


def _result(envs, batch, speed, memory):
    return {
        'status': 'ok',
        'envs': envs,
        'batch_size': batch,
        'compile_model': True,
        'use_bfloat16': True,
        'peak_memory_reserved_mb': memory,
        'total_memory_mb': 100.0,
        'phases': {
            'parent_only': {'parent_transitions_per_second': speed * 1.2},
            'branch_active': {'parent_transitions_per_second': speed},
        },
    }


class RealTrainingCalibrationTest(unittest.TestCase):
    def test_formal_config_is_scratch_128m_with_scaled_branch(self):
        config = TrainingConfig.from_toml(FORMAL_CONFIG)
        self.assertEqual(config.total_transitions, 128_000_000)
        self.assertEqual(config.training_physics_fps, 30)
        self.assertEqual(config.stage_pilot_max_drops, 300)
        self.assertEqual(config.stage_pilot_policy_epsilon, 0.05)
        self.assertEqual(config.dqn.epsilon_schedule[0], (0, 1.0))
        self.assertEqual(config.dqn.epsilon_schedule[1], (6_400_000, 0.05))
        self.assertTrue(config.branch_learning.enabled)
        self.assertEqual(
            config.branch_learning.transition_budget, 16_777_216
        )
        self.assertEqual(config.branch_learning.start_transition, 6_400_000)

    def test_trial_keeps_formal_capacity_and_forces_heavy_phase(self):
        base = TrainingConfig.from_toml(FORMAL_CONFIG)
        trial = build_trial_config(
            base,
            run_dir='runs/test-calibration',
            envs=1024,
            batch_size=384,
            parent_steps=4,
            branch_steps=8,
        )
        self.assertEqual(trial.replay.capacity, base.replay.capacity)
        self.assertEqual(
            trial.branch_learning.replay_capacity,
            base.branch_learning.replay_capacity,
        )
        self.assertEqual(trial.replay.batch_size, 384)
        self.assertEqual(trial.dqn.epsilon_schedule[0], (0, 0.05))
        self.assertEqual(
            trial.branch_learning.start_transition,
            trial.replay.warmup_transitions + 1024 * 4,
        )
        self.assertGreater(trial.branch_learning.transition_budget, 0)
        self.assertEqual(
            trial.branch_learning.transition_budget
            % trial.branch_learning.simulator_batch_size,
            0,
        )

    def test_lightweight_bracket_and_twenty_percent_selection(self):
        self.assertEqual(
            environment_bracket(1792, 896, 3584),
            (1792, 1280, 2304),
        )
        results = (
            _result(1280, 256, 900.0, 55.0),
            _result(1792, 256, 1000.0, 70.0),
            _result(2304, 256, 1080.0, 84.0),
        )
        selected, fastest = select_recommendation(results)
        self.assertEqual(fastest, 1080.0)
        self.assertEqual(selected['envs'], 1280)

    def test_stage_pilot_epsilon_is_validated(self):
        with self.assertRaisesRegex(ValueError, 'stage_pilot_policy_epsilon'):
            TrainingConfig(stage_pilot_policy_epsilon=1.1)

    def test_prewarm_teacher_does_not_initialize_training_model(self):
        model_config = _small_model_config()
        source = DqnLearner(
            BaselineGnnDqn(model_config),
            DqnConfig(use_bfloat16=False, fused_adam=False),
        )
        with torch.no_grad():
            for parameter in source.online_module.parameters():
                parameter.fill_(0.25)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            common = dict(
                device='cpu',
                max_envs=2,
                active_envs=2,
                total_transitions=4,
                model=model_config,
                dqn=DqnConfig(use_bfloat16=False, fused_adam=False),
                replay=ReplayConfig(
                    capacity=4,
                    batch_size=2,
                    warmup_transitions=2,
                    warmup_stage_ratios=(1.0, 0.0, 0.0, 0.0),
                ),
                analysis=AnalysisExportConfig(
                    transition_sample_size=0,
                    trajectory_episodes=0,
                    critical_event_episodes=0,
                ),
                dashboard=DashboardConfig(enabled=False),
                autoscale=AutoScaleConfig(enabled=False),
            )
            source_config = TrainingConfig(
                run_dir=str(root / 'source'), **common
            )
            checkpoint = root / 'teacher.pt'
            save_checkpoint_atomic(
                checkpoint,
                learner=source,
                training_config=source_config,
                progress={'transitions': 128_000_000},
                replay_metadata={'replay_saved_in_checkpoint': False},
            )
            target_config = TrainingConfig(
                run_dir=str(root / 'target'),
                stage_pilot_policy_epsilon=0.05,
                **common,
            )
            trainer = BaselineTrainer(target_config)
            before = {
                name: value.detach().clone()
                for name, value in trainer.learner.online_module.state_dict().items()
            }
            try:
                metadata = trainer.load_stage_pilot_checkpoint(checkpoint)
                after = trainer.learner.online_module.state_dict()
                actions = trainer._prewarm_actions(
                    torch.ones(2, dtype=torch.bool)
                )
                identity = json.loads(
                    (root / 'target' / 'run_identity.json').read_text(
                        encoding='utf-8'
                    )
                )
                trainer.release_stage_pilot_model()
            finally:
                trainer.resource_sampler.close()
                trainer.dashboard.close()
            self.assertTrue(all(
                torch.equal(before[name], after[name]) for name in before
            ))
            self.assertEqual(actions.shape, (2,))
            self.assertEqual(metadata['kind'], 'prewarm_teacher_only')
            self.assertIn('online_model', metadata[
                'excluded_from_training_initialization'
            ])
            self.assertEqual(
                identity['stage_pilot_teacher']['kind'],
                'prewarm_teacher_only',
            )

    def test_short_run_can_skip_formal_artifacts(self):
        model_config = _small_model_config()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = TrainingConfig(
                run_dir=str(root),
                device='cpu',
                max_envs=2,
                active_envs=2,
                total_transitions=4,
                log_interval_seconds=0.001,
                model=model_config,
                dqn=DqnConfig(use_bfloat16=False, fused_adam=False),
                replay=ReplayConfig(
                    capacity=4,
                    batch_size=2,
                    warmup_transitions=2,
                    warmup_stage_ratios=(1.0, 0.0, 0.0, 0.0),
                ),
                analysis=AnalysisExportConfig(
                    transition_sample_size=0,
                    trajectory_episodes=0,
                    critical_event_episodes=0,
                ),
                dashboard=DashboardConfig(enabled=False),
                autoscale=AutoScaleConfig(enabled=False),
            )
            trainer = BaselineTrainer(config)
            result = trainer.run(
                final_evaluation=False,
                finalize_artifacts=False,
            )
            status = json.loads(
                (root / 'run_status.json').read_text(encoding='utf-8')
            )
        self.assertEqual(result['transitions'], 4)
        self.assertFalse((root / 'checkpoints' / 'final.pt').exists())
        self.assertFalse((root / 'artifact_manifest.json').exists())
        self.assertFalse(status['artifact_finalization'])


if __name__ == '__main__':
    unittest.main()
