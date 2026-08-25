import tempfile
import unittest
from pathlib import Path

import torch

from daxigua.simulator import (
    SimulatorConfig,
    TensorVectorSimulator,
    load_static_scene_dataset,
    save_static_scene_dataset,
)


class StaticSceneDatasetTest(unittest.TestCase):
    def test_round_trip_preserves_observation_and_row_selection(self):
        simulator = TensorVectorSimulator(
            3,
            config=SimulatorConfig.training_fast(
                max_fruits=8, action_count=21, queue_length=4,
                use_cuda_extension=False,
            ),
            device='cpu',
        )
        observation = simulator.reset(seeds=torch.tensor([11, 22, 33]))
        metadata = {
            'seed': torch.tensor([11, 22, 33], dtype=torch.int64),
            'policy_mode': torch.tensor([1, 0, 1], dtype=torch.int8),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scenes.pt'
            save_static_scene_dataset(
                path,
                observation,
                metadata=metadata,
                manifest={'purpose': 'unit_test'},
            )
            loaded = load_static_scene_dataset(path, rows=[2, 0])

        self.assertEqual(loaded.batch_size, 2)
        self.assertEqual(loaded.manifest['purpose'], 'unit_test')
        self.assertTrue(torch.equal(
            loaded.metadata['seed'], torch.tensor([33, 11])
        ))
        self.assertTrue(torch.equal(
            loaded.observation.fruit_queue,
            observation.fruit_queue.index_select(0, torch.tensor([2, 0])),
        ))

    def test_metadata_must_share_observation_batch(self):
        simulator = TensorVectorSimulator(
            2,
            config=SimulatorConfig.training_fast(
                max_fruits=8, action_count=21, queue_length=4,
                use_cuda_extension=False,
            ),
            device='cpu',
        )
        observation = simulator.reset(seeds=5)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'batch dimension'):
                save_static_scene_dataset(
                    Path(temporary) / 'invalid.pt',
                    observation,
                    metadata={'seed': torch.tensor([5])},
                )


if __name__ == '__main__':
    unittest.main()
