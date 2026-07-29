"""ParallelRolloutCollector 集中式 actor 推理集成测试。"""

from __future__ import annotations

import time
import unittest

import torch

from daxigua_rl import DaxiguaEnvConfig, ReplayBuffer
from daxigua_rl.attribution import ANALYSIS_ACTION_COUNT
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.training import ParallelRolloutCollector


class CentralizedActorInferenceTest(unittest.TestCase):
    """验证 worker 请求、主进程微批推理和资源关闭形成完整闭环。"""

    def setUp(self):
        torch.manual_seed(29)

    def test_cpu_centralized_actor_collects_batches_and_closes(self):
        replay_buffer = ReplayBuffer(capacity=32, seed=31)
        model_config = {
            'hidden_dim': 16,
            'message_layers': 1,
            'activation': 'silu',
            'dropout': 0.0,
        }
        model = GNNQNetwork(**model_config)
        collector = ParallelRolloutCollector(
            worker_count=2,
            env_config=DaxiguaEnvConfig(
                # ParallelRolloutCollector 的正式 worker 默认启用完整归因，
                # 因而必须使用当前规范动作列。
                action_count=ANALYSIS_ACTION_COUNT,
                physics_fps=30,
                max_physics_frames=40,
                stable_frames=2,
                space_iterations=8,
            ),
            replay_buffer=replay_buffer,
            model_config=model_config,
            model=model,
            seed=37,
            centralized_actor_inference=True,
            actor_batch_size=2,
            actor_batch_wait_ms=50.0,
            actor_request_timeout_seconds=15.0,
        )

        close_result = None
        try:
            self.assertTrue(collector.centralized_actor_inference)
            self.assertFalse(collector.model_synced)
            self.assertEqual(collector.actor_inference_requests, 0)
            self.assertEqual(collector.actor_inference_batches, 0)

            # 使用明确的哨兵参数验证 sync_model 更新的是主进程 actor 副本，
            # 而不只是把 model_synced 标志置为真。
            with torch.no_grad():
                model.q_head[-1].bias.fill_(1.25)
            collector.sync_model(
                model,
                policy_version='centralized-test-v1',
            )

            self.assertTrue(collector.model_synced)
            self.assertEqual(
                collector.policy_version,
                'centralized-test-v1',
            )
            self.assertTrue(torch.equal(
                collector._actor_model.q_head[-1].bias,
                model.q_head[-1].bias,
            ))
            self.assertFalse(collector._actor_model.training)

            # epsilon=0 保证每个真实环境 step 都必须经集中 actor 请求 Q 值。
            stats = collector.collect_steps(4, epsilon=0.0)

            self.assertEqual(stats.steps, 4)
            self.assertEqual(stats.greedy_actions, 4)
            self.assertEqual(stats.random_actions, 0)
            self.assertEqual(len(replay_buffer), 4)
            self.assertEqual(stats.replay_transitions_emitted, 4)

            self.assertIsNone(collector._actor_failure)
            self.assertEqual(
                collector.actor_inference_requests,
                stats.greedy_actions,
            )
            self.assertGreater(
                collector.actor_inference_batches,
                0,
            )
            self.assertLessEqual(
                collector.actor_inference_batches,
                collector.actor_inference_requests,
            )
            self.assertGreaterEqual(
                collector.actor_inference_max_batch,
                1,
            )
            self.assertLessEqual(
                collector.actor_inference_max_batch,
                collector.actor_batch_size,
            )
            self.assertAlmostEqual(
                collector.actor_mean_batch_size,
                (
                    collector.actor_inference_requests
                    / collector.actor_inference_batches
                ),
            )
            self.assertGreater(
                collector.actor_inference_seconds,
                0.0,
            )
            snapshot = collector.actor_stats_snapshot()
            self.assertEqual(
                snapshot['requests'],
                collector.actor_inference_requests,
            )
            self.assertEqual(
                snapshot['batches'],
                collector.actor_inference_batches,
            )
            self.assertAlmostEqual(
                snapshot['mean_batch_size'],
                collector.actor_mean_batch_size,
            )
            self.assertEqual(
                snapshot['max_batch'],
                collector.actor_inference_max_batch,
            )
        finally:
            close_started_at = time.perf_counter()
            close_result = collector.close()
            close_seconds = time.perf_counter() - close_started_at

        # close() 必须完成 worker finalization、executor shutdown、actor thread
        # join 和队列关闭；第二次调用还应幂等返回，不重新等待任何进程。
        self.assertTrue(collector._closed)
        self.assertIsNone(collector._actor_thread)
        self.assertEqual(len(close_result), 2)
        self.assertLess(close_seconds, 15.0)
        repeated_started_at = time.perf_counter()
        self.assertIs(collector.close(), close_result)
        self.assertLess(
            time.perf_counter() - repeated_started_at,
            1.0,
        )


if __name__ == '__main__':
    unittest.main()
