from pathlib import Path
import unittest

from daxigua.rl.config import TrainingConfig
from daxigua.rl.trainer import training_simulator_config
from daxigua.simulator import PHYSICS_IDENTITY


class FullFallTransferConfigTest(unittest.TestCase):
    def test_transfer_config_starts_a_fresh_120fps_physics_run(self):
        project_root = Path(__file__).resolve().parents[1]
        config = TrainingConfig.from_toml(
            project_root / 'configs' / 'sab-full-fall-t120-16m-r1.toml'
        )
        simulator = training_simulator_config(config, 'cpu')

        self.assertEqual(PHYSICS_IDENTITY, 'tensor_cuda_v3_full_fall')
        self.assertEqual(config.training_physics_fps, 120)
        self.assertEqual(config.total_transitions, 16_000_000)
        self.assertEqual(config.seed, 20260813)
        self.assertEqual(config.finalization_reserve_seconds, 14_400.0)
        self.assertEqual(config.reward.kind, 'score_v1')
        self.assertFalse(config.branch_learning.enabled)
        self.assertEqual(simulator.physics_fps, 120)
        self.assertFalse(simulator.drop_fast_forward)


if __name__ == '__main__':
    unittest.main()
