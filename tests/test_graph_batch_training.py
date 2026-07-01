"""GraphBatch 和张量化 DQN 训练链路测试。"""

from __future__ import annotations

import random
import unittest

import torch

from daxigua_rl import DaxiguaEnv, DaxiguaEnvConfig, GraphBuilder, ReplayBuffer
from daxigua_rl.graph.tensor import collate_graph_tensors, graph_to_tensor
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.training import (
    DQNTrainer,
    DQNTrainerConfig,
    RolloutCollector,
    TensorTransition,
)


class GraphBatchTrainingTest(unittest.TestCase):
    """验证批量图前向和张量化训练更新。"""

    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)

    def _make_graph_tensors(self, count=3):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(action_count=7, max_physics_frames=240, stable_frames=6))
        builder = GraphBuilder()
        obs, info = env.reset(seed=0)
        graphs = []

        for action_offset in range(count):
            candidates = tuple(info['action_candidates'])
            graph_data = builder.build(obs, candidates)
            graphs.append(graph_to_tensor(graph_data))

            obs, _reward, terminated, truncated, info = env.step(action_offset % len(candidates))
            if terminated or truncated:
                obs, info = env.reset(seed=100 + action_offset)

        return graphs

    def test_graph_batch_matches_individual_forward(self):
        """同一模型上，单图前向和 GraphBatch 对应切片应完全一致。"""

        graphs = self._make_graph_tensors(count=3)
        model = GNNQNetwork(hidden_dim=32, message_layers=2)
        model.eval()

        with torch.no_grad():
            individual_outputs = [model(graph) for graph in graphs]
            graph_batch = collate_graph_tensors(graphs)
            batch_output = model(graph_batch)

        self.assertEqual(graph_batch.num_graphs, len(graphs))
        self.assertEqual(int(batch_output.shape[0]), graph_batch.action_count)

        for expected, (start, end) in zip(individual_outputs, graph_batch.action_slices):
            actual = batch_output[start:end]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_collector_writes_tensor_transition_and_trainer_updates(self):
        """RolloutCollector 应写入 TensorTransition，DQNTrainer 应完成一次批量更新。"""

        env = DaxiguaEnv(config=DaxiguaEnvConfig(action_count=7, max_physics_frames=240, stable_frames=6))
        replay_buffer = ReplayBuffer(capacity=128, seed=1)
        model = GNNQNetwork(hidden_dim=32, message_layers=2)
        target_model = GNNQNetwork(hidden_dim=32, message_layers=2)

        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay_buffer,
            model=model,
            seed=2,
        )
        collector.collect_steps(16, epsilon=1.0)

        sampled = replay_buffer.sample(1)[0]
        self.assertIsInstance(sampled, TensorTransition)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trainer = DQNTrainer(
            online_model=model,
            target_model=target_model,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            config=DQNTrainerConfig(batch_size=8, target_update_interval=10, grad_clip_norm=10.0),
        )
        stats = trainer.train_step()

        self.assertEqual(stats.batch_size, 8)
        self.assertEqual(stats.update_step, 1)
        self.assertTrue(torch.isfinite(torch.tensor(stats.loss)))


if __name__ == '__main__':
    unittest.main()
