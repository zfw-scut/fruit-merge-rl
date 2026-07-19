"""GraphBatch 和张量化 DQN 训练链路测试。"""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import torch

from daxigua_rl import DaxiguaEnv, DaxiguaEnvConfig, GraphBuilder, ReplayBuffer
from daxigua_rl.graph.tensor import collate_graph_tensors, graph_to_tensor
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.training import (
    DQNTrainer,
    DQNTrainerConfig,
    ParallelRolloutCollector,
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
        self.assertEqual(sampled.graph.node_features.dtype, torch.float16)
        self.assertEqual(sampled.graph.edge_features.dtype, torch.float16)

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

    def test_collector_reuses_next_graph_as_next_current_graph(self):
        """非终止 step 的 next_graph 应被下一步直接复用，减少重复构图。"""

        env = DaxiguaEnv(config=DaxiguaEnvConfig(action_count=7, max_physics_frames=120, stable_frames=4))
        replay_buffer = ReplayBuffer(capacity=32, seed=5)
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay_buffer,
            model=GNNQNetwork(hidden_dim=32, message_layers=2),
            seed=6,
        )

        stats = collector.collect_steps(6, epsilon=1.0)
        transitions = replay_buffer.to_tuple()
        reusable_pairs = [
            (left, right)
            for left, right in zip(transitions, transitions[1:])
            if not left.done
        ]

        self.assertGreater(stats.graph_cache_hits, 0)
        self.assertTrue(reusable_pairs)
        for left, right in reusable_pairs:
            self.assertIs(left.next_graph, right.graph)

    def test_float16_replay_storage_trains_with_float32_model(self):
        """ReplayBuffer 固定用 float16 存图，模型前向时应自动转回 float32。"""

        env = DaxiguaEnv(config=DaxiguaEnvConfig(action_count=7, max_physics_frames=240, stable_frames=6))
        replay_buffer = ReplayBuffer(capacity=128, seed=3)
        model = GNNQNetwork(hidden_dim=32, message_layers=2)
        target_model = GNNQNetwork(hidden_dim=32, message_layers=2)

        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay_buffer,
            model=model,
            seed=4,
        )
        collector.collect_steps(16, epsilon=1.0)

        sampled = replay_buffer.sample(1)[0]
        self.assertIsInstance(sampled, TensorTransition)
        self.assertEqual(sampled.graph.node_features.dtype, torch.float16)
        self.assertEqual(sampled.graph.edge_features.dtype, torch.float16)

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
        self.assertTrue(torch.isfinite(torch.tensor(stats.loss)))

    def test_hybrid_replay_samples_from_hot_and_cold_storage(self):
        """热内存 + 冷磁盘 replay 应能降低常驻热数据并继续支持 DQN 训练。"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            replay_buffer = ReplayBuffer(
                capacity=16,
                seed=7,
                hot_capacity=5,
                cold_dir=Path(tmp_dir) / 'cold',
                segment_size=4,
                cold_cache_size=8,
                cold_sample_ratio=0.5,
                cold_cache_refresh_interval=1,
            )
            env = DaxiguaEnv(config=DaxiguaEnvConfig(action_count=7, max_physics_frames=120, stable_frames=4))
            model = GNNQNetwork(hidden_dim=32, message_layers=2)
            target_model = GNNQNetwork(hidden_dim=32, message_layers=2)
            collector = RolloutCollector(
                env=env,
                graph_builder=GraphBuilder(),
                replay_buffer=replay_buffer,
                model=model,
                seed=8,
            )
            collector.collect_steps(16, epsilon=1.0)

            storage_stats = replay_buffer.storage_stats
            self.assertEqual(storage_stats['mode'], 'hybrid')
            self.assertEqual(len(replay_buffer), 16)
            self.assertLessEqual(storage_stats['hot_count'], 5)
            self.assertGreater(storage_stats['cold_count'], 0)

            sampled = replay_buffer.sample(8)
            self.assertEqual(len(sampled), 8)
            self.assertTrue(all(isinstance(item, TensorTransition) for item in sampled))

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            trainer = DQNTrainer(
                online_model=model,
                target_model=target_model,
                replay_buffer=replay_buffer,
                optimizer=optimizer,
                config=DQNTrainerConfig(batch_size=8, target_update_interval=10, grad_clip_norm=10.0),
            )
            stats = trainer.train_step()
            self.assertTrue(torch.isfinite(torch.tensor(stats.loss)))

    def test_parallel_collector_async_handle_collects_transitions(self):
        """ParallelRolloutCollector 应能通过异步 handle 从多个 worker 回收经验。"""

        replay_buffer = ReplayBuffer(capacity=32, seed=9)
        model = GNNQNetwork(hidden_dim=32, message_layers=2)
        collector = ParallelRolloutCollector(
            worker_count=2,
            env_config=DaxiguaEnvConfig(
                action_count=5,
                physics_fps=30,
                max_physics_frames=80,
                stable_frames=3,
                space_iterations=8,
            ),
            replay_buffer=replay_buffer,
            model_config={
                'hidden_dim': 32,
                'message_layers': 2,
                'activation': 'silu',
                'dropout': 0.0,
            },
            model=model,
            seed=10,
        )
        try:
            collector.sync_model(model)
            handle = collector.start_collect_steps(4, epsilon=1.0)
            stats = collector.finish_collect_steps(handle)
        finally:
            collector.close()

        self.assertEqual(stats.steps, 4)
        self.assertEqual(len(replay_buffer), 4)
        self.assertGreater(stats.collect_seconds, 0.0)
        self.assertIsInstance(replay_buffer.sample(1)[0], TensorTransition)


if __name__ == '__main__':
    unittest.main()
