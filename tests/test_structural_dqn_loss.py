"""DQN 六维结构辅助监督的兼容性与掩码测试。"""

from __future__ import annotations

import unittest

import torch

from daxigua_rl import (
    DaxiguaEnv,
    DaxiguaEnvConfig,
    GraphBuilder,
    ReplayBuffer,
)
from daxigua_rl.graph.tensor import (
    collate_graph_tensors,
    graph_to_tensor,
)
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.training import (
    DQNTrainer,
    DQNTrainerConfig,
    TensorTransition,
)
from daxigua_rl.training.structural_targets import StructuralTarget


def _graph(action_count=7):
    env = DaxiguaEnv(config=DaxiguaEnvConfig(
        action_count=action_count,
    ))
    state, info = env.reset(seed=43)
    graph_data = GraphBuilder().build(
        state,
        tuple(info['action_candidates']),
    )
    return graph_to_tensor(graph_data)


def _terminal_transition(
        graph,
        *,
        action_offset,
        target=None,
        reward=0.0):
    return TensorTransition(
        graph=graph,
        action_offset=action_offset,
        reward=reward,
        next_graph=None,
        terminated=True,
        truncated=False,
        structural_target=target,
    )


class StructuralDQNMaskTest(unittest.TestCase):
    """直接验证实际动作索引和逐维 valid_mask。"""

    def setUp(self):
        torch.manual_seed(47)
        self.graph = _graph()
        self.online_model = GNNQNetwork(
            hidden_dim=16,
            message_layers=1,
        )
        self.target_model = GNNQNetwork(
            hidden_dim=16,
            message_layers=1,
        )
        self.trainer = DQNTrainer(
            online_model=self.online_model,
            target_model=self.target_model,
            replay_buffer=ReplayBuffer(capacity=4, seed=53),
            optimizer=torch.optim.Adam(
                self.online_model.parameters(),
                lr=1e-4,
            ),
            config=DQNTrainerConfig(
                batch_size=1,
                sync_target_on_init=False,
                lambda_structural=1.0,
            ),
        )

    def test_only_selected_action_and_valid_dimensions_contribute(self):
        first_target = StructuralTarget(
            values=(0.5, 0.0, -0.5, 0.0, 0.0, 0.0),
            valid_mask=(1 << 0) | (1 << 2),
        )
        second_target = StructuralTarget(
            values=(0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            valid_mask=1 << 5,
        )
        transitions = (
            _terminal_transition(
                self.graph,
                action_offset=1,
                target=first_target,
            ),
            _terminal_transition(
                self.graph,
                action_offset=2,
                target=second_target,
            ),
            _terminal_transition(
                self.graph,
                action_offset=3,
                target=None,
            ),
        )
        graph_batch = collate_graph_tensors(
            transition.graph
            for transition in transitions
        )

        # 所有非监督位置故意设成非零；如果实现误用了其它动作或无效维度，
        # loss 和梯度稀疏性都会立刻变化。
        predictions = torch.full(
            (graph_batch.action_count, 6),
            0.9,
        )
        first_action = graph_batch.action_slices[0][0] + 1
        second_action = graph_batch.action_slices[1][0] + 2
        predictions[first_action, 0] = 1.0
        predictions[first_action, 2] = 0.5
        predictions[second_action, 5] = 0.0
        predictions.requires_grad_(True)

        result = self.trainer._compute_structural_loss(
            transitions=transitions,
            graph_batch=graph_batch,
            predictions=predictions,
            reference_tensor=torch.zeros(
                len(transitions),
                requires_grad=True,
            ),
        )

        # SmoothL1(beta=1):
        # error 0.5 -> 0.125；两个 error 1.0 -> 0.5 + 0.5。
        self.assertAlmostEqual(
            float(result['loss'].detach()),
            0.375,
            places=6,
        )
        self.assertEqual(result['valid_count'], 3)
        self.assertEqual(result['sample_count'], 2)
        self.assertAlmostEqual(
            result['mean_abs_error'],
            (0.5 + 1.0 + 1.0) / 3.0,
            places=6,
        )

        result['loss'].backward()
        expected_nonzero = torch.zeros_like(
            predictions,
            dtype=torch.bool,
        )
        expected_nonzero[first_action, 0] = True
        expected_nonzero[first_action, 2] = True
        expected_nonzero[second_action, 5] = True
        actual_nonzero = predictions.grad != 0.0
        self.assertTrue(torch.equal(
            actual_nonzero,
            expected_nonzero,
        ))


class StructuralDQNOptimizerIntegrationTest(unittest.TestCase):
    """验证真实 GNN 的结构头进入同一次 DQN optimizer step。"""

    def test_train_step_reports_and_updates_structural_head(self):
        torch.manual_seed(59)
        graph = _graph()
        replay = ReplayBuffer(capacity=4, seed=61)
        replay.extend((
            _terminal_transition(
                graph,
                action_offset=0,
                reward=1.0,
                target=StructuralTarget(
                    values=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    valid_mask=1 << 0,
                ),
            ),
            _terminal_transition(
                graph,
                action_offset=1,
                reward=2.0,
                target=StructuralTarget(
                    values=(0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
                    valid_mask=1 << 5,
                ),
            ),
            _terminal_transition(
                graph,
                action_offset=2,
                reward=3.0,
                target=None,
            ),
        ))
        online_model = GNNQNetwork(
            hidden_dim=16,
            message_layers=1,
        )
        target_model = GNNQNetwork(
            hidden_dim=16,
            message_layers=1,
        )
        optimizer = torch.optim.Adam(
            online_model.parameters(),
            lr=1e-3,
        )
        trainer = DQNTrainer(
            online_model=online_model,
            target_model=target_model,
            replay_buffer=replay,
            optimizer=optimizer,
            config=DQNTrainerConfig(
                batch_size=3,
                sync_target_on_init=True,
                lambda_rule=0.0,
                lambda_counterfactual=0.0,
                lambda_structural=0.5,
            ),
        )
        before = tuple(
            parameter.detach().clone()
            for parameter in online_model.structure_head.parameters()
        )

        stats = trainer.train_step()

        after = tuple(
            parameter.detach()
            for parameter in online_model.structure_head.parameters()
        )
        self.assertEqual(stats.structural_valid_count, 2)
        self.assertEqual(stats.structural_sample_count, 2)
        self.assertGreater(stats.structural_loss, 0.0)
        self.assertAlmostEqual(
            stats.weighted_structural_loss,
            0.5 * stats.structural_loss,
            places=6,
        )
        self.assertAlmostEqual(
            stats.loss,
            stats.td_loss + stats.weighted_structural_loss,
            places=5,
        )
        self.assertGreater(
            stats.structural_mean_abs_error,
            0.0,
        )
        self.assertTrue(any(
            not torch.equal(left, right)
            for left, right in zip(before, after)
        ))


if __name__ == '__main__':
    unittest.main()
