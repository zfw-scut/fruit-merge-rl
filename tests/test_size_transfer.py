"""新场地 weights-only 模型迁移入口测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from daxigua_rl.models import GNNQNetwork
from daxigua_rl.scripts.train_dqn import load_initial_model_weights
from daxigua_rl.training.checkpointing import (
    atomic_torch_save,
    build_training_checkpoint,
)


class SizeTransferInitializationTest(unittest.TestCase):
    """迁移只能继承网络参数，且必须保留新 run 语义。"""

    @staticmethod
    def _args(**overrides):
        values = {
            'hidden_dim': 8,
            'message_layers': 1,
            'activation': 'silu',
            'dropout': 0.0,
            'board_width': 560,
            'board_height': 1120,
            'spawn_y': 252,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _source_config():
        return {
            'hidden_dim': 8,
            'message_layers': 1,
            'activation': 'silu',
            'dropout': 0.0,
            'action_count': 15,
            # 模拟首次 250K：当时 manifest 还没有显式几何字段。
        }

    def _write_checkpoint(self, path):
        source_model = GNNQNetwork(
            hidden_dim=8,
            message_layers=1,
            activation='silu',
            dropout=0.0,
        )
        with torch.no_grad():
            for parameter in source_model.parameters():
                parameter.fill_(0.125)
        payload = build_training_checkpoint(
            training_state={
                'online_model': source_model.state_dict(),
                'update_step': 120_000,
                'env_steps': 1_234_567,
                # 即使源 checkpoint 带有这些状态，初始化入口也不会读取。
                'optimizer': {'source_only': True},
                'target_model': {'source_only': True},
            },
            config=self._source_config(),
        )
        atomic_torch_save(payload, path)
        return source_model

    def test_loads_online_into_online_and_target_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'source.pt'
            source_model = self._write_checkpoint(path)
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

            report = load_initial_model_weights(
                path,
                args=self._args(),
                online_model=online_model,
                target_model=target_model,
            )

        for name, source_tensor in source_model.state_dict().items():
            self.assertTrue(torch.equal(
                online_model.state_dict()[name],
                source_tensor,
            ))
            self.assertTrue(torch.equal(
                target_model.state_dict()[name],
                source_tensor,
            ))
        self.assertEqual(report['mode'], 'weights_only')
        self.assertEqual(report['source_update_step'], 120_000)
        self.assertEqual(
            report['source_geometry'],
            {
                'board_width': 400,
                'board_height': 800,
                'spawn_y': 180,
            },
        )
        self.assertEqual(
            report['target_geometry'],
            {
                'board_width': 560,
                'board_height': 1120,
                'spawn_y': 252,
            },
        )
        self.assertIn('optimizer', report['reset_state'])
        self.assertIn('td_replay', report['reset_state'])
        self.assertIn('causal_replay', report['reset_state'])
        self.assertIn('rng', report['reset_state'])

    def test_rejects_semantic_architecture_mismatch_even_if_shapes_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'source.pt'
            self._write_checkpoint(path)
            online_model = GNNQNetwork(
                hidden_dim=8,
                message_layers=1,
                activation='relu',
                dropout=0.0,
            )
            target_model = GNNQNetwork(
                hidden_dim=8,
                message_layers=1,
                activation='relu',
                dropout=0.0,
            )

            with self.assertRaisesRegex(
                    ValueError,
                    'architecture mismatch'):
                load_initial_model_weights(
                    path,
                    args=self._args(activation='relu'),
                    online_model=online_model,
                    target_model=target_model,
                )


if __name__ == '__main__':
    unittest.main()
