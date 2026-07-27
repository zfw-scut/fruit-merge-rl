"""DQN 主 TD、规则排序和反事实差值联合训练测试。"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from daxigua_rl.attribution.causal_replay import (
    CausalReplayBuffer,
    CausalSample,
)
from daxigua_rl.graph.tensor import GraphTensor
from daxigua_rl.training.dqn import DQNTrainer, DQNTrainerConfig
from daxigua_rl.training.identity import TransitionKey
from daxigua_rl.training.replay_buffer import ReplayBuffer
from daxigua_rl.training.tensor_transition import TensorTransition


def _graph():
    action_count = 15
    return GraphTensor(
        node_features=torch.zeros((action_count, 1), dtype=torch.float16),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_features=torch.empty((0, 1), dtype=torch.float16),
        action_node_indices=torch.arange(action_count, dtype=torch.long),
        action_indices=torch.arange(action_count, dtype=torch.long),
        node_feature_names=('value',),
        edge_feature_names=('edge',),
    )


class _OffsetQModel(nn.Module):
    """只按动作 offset 输出参数，便于验证 pair 梯度方向。"""

    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(15))

    def forward(self, graph):
        action_slices = getattr(graph, 'action_slices', None)
        if action_slices is None:
            return self.q[:graph.action_count]
        return torch.cat([
            self.q[:action_end - action_start]
            for action_start, action_end in action_slices
        ])


def _td_replay(*, reward=0.0):
    replay = ReplayBuffer(capacity=8, seed=0)
    for _ in range(2):
        replay.push(TensorTransition(
            graph=_graph(),
            action_offset=7,
            reward=reward,
            next_graph=None,
            terminated=True,
            truncated=False,
        ))
    return replay


def _causal_sample(
        *,
        step,
        kind='rule',
        direction=1,
        target_delta=None):
    return CausalSample(
        graph=_graph(),
        actual_action_offset=14,
        comparison_action_offset=0,
        direction=direction,
        target_margin=2.0 if kind == 'rule' else 0.0,
        confidence=0.9,
        cause_type=(
            'DIRECT_TRIGGER'
            if kind == 'rule'
            else 'COUNTERFACTUAL_RETURN'
        ),
        delay=2,
        transition_key=TransitionKey(0, 0, step),
        attribution_version='causal_attribution_v1',
        supervision_kind=kind,
        stratum=(
            'positive_setup'
            if kind == 'rule'
            else 'counterfactual'
        ),
        event_key=f'event:{step}',
        budget_key=f'budget:{step}',
        target_delta=target_delta,
    )


def _trainer(causal_replay, *, interval=1):
    online = _OffsetQModel()
    target = _OffsetQModel()
    return DQNTrainer(
        online_model=online,
        target_model=target,
        replay_buffer=_td_replay(),
        optimizer=torch.optim.SGD(online.parameters(), lr=0.1),
        causal_replay_buffer=causal_replay,
        config=DQNTrainerConfig(
            batch_size=2,
            target_update_interval=100,
            grad_clip_norm=None,
            causal_batch_size=8,
            causal_update_interval=interval,
            lambda_rule=0.15,
            lambda_counterfactual=0.10,
        ),
    )


class CausalDQNTrainingTests(unittest.TestCase):

    def test_rule_ranking_contributes_to_total_loss_and_gradient(self):
        causal = CausalReplayBuffer(capacity=8, seed=0)
        causal.push(_causal_sample(step=0))
        trainer = _trainer(causal)

        stats = trainer.train_step()

        self.assertTrue(stats.causal_update_applied)
        self.assertEqual(stats.causal_batch_size, 1)
        self.assertEqual(stats.rule_batch_size, 1)
        self.assertEqual(stats.counterfactual_batch_size, 0)
        self.assertGreater(stats.rule_rank_loss, 0.0)
        self.assertGreater(stats.loss, stats.td_loss)
        self.assertEqual(stats.rule_pair_accuracy, 0.0)
        self.assertEqual(stats.rule_margin_satisfaction_rate, 0.0)
        self.assertGreater(
            trainer.online_model.q[14].item(),
            trainer.online_model.q[0].item(),
        )

    def test_counterfactual_delta_uses_huber_and_reports_sign(self):
        causal = CausalReplayBuffer(capacity=8, seed=0)
        causal.push(_causal_sample(
            step=1,
            kind='counterfactual',
            target_delta=4.0,
        ))
        trainer = _trainer(causal)

        stats = trainer.train_step()

        self.assertEqual(stats.rule_batch_size, 0)
        self.assertEqual(stats.counterfactual_batch_size, 1)
        self.assertGreater(stats.counterfactual_loss, 0.0)
        self.assertGreater(stats.counterfactual_mean_abs_error, 0.0)
        self.assertEqual(stats.counterfactual_sign_accuracy, 0.0)
        self.assertGreater(stats.loss, stats.td_loss)
        self.assertGreater(
            trainer.online_model.q[14].item(),
            trainer.online_model.q[0].item(),
        )

    def test_counterfactual_prediction_outside_target_clip_keeps_gradient(self):
        causal = CausalReplayBuffer(capacity=8, seed=0)
        causal.push(_causal_sample(
            step=2,
            kind='counterfactual',
            target_delta=1.0,
        ))
        trainer = _trainer(causal)
        scale = trainer.config.counterfactual_return_scale
        with torch.no_grad():
            trainer.online_model.q[14] = 6.0 * scale
        previous = trainer.online_model.q[14].item()

        stats = trainer.train_step()

        self.assertGreater(stats.counterfactual_loss, 0.0)
        self.assertLess(trainer.online_model.q[14].item(), previous)

    def test_non_finite_target_fails_before_optimizer_step(self):
        online = _OffsetQModel()
        target = _OffsetQModel()
        trainer = DQNTrainer(
            online_model=online,
            target_model=target,
            replay_buffer=_td_replay(reward=float('nan')),
            optimizer=torch.optim.SGD(online.parameters(), lr=0.1),
            config=DQNTrainerConfig(
                batch_size=2,
                target_update_interval=100,
            ),
        )
        previous = online.q.detach().clone()

        with self.assertRaisesRegex(
                FloatingPointError,
                'non-finite target'):
            trainer.train_step()

        torch.testing.assert_close(online.q.detach(), previous)
        self.assertEqual(trainer.update_step, 0)

    def test_causal_interval_degrades_other_updates_to_pure_td(self):
        causal = CausalReplayBuffer(capacity=8, seed=0)
        causal.push(_causal_sample(step=0))
        trainer = _trainer(causal, interval=2)

        first = trainer.train_step()
        second = trainer.train_step()

        self.assertFalse(first.causal_update_applied)
        self.assertEqual(first.loss, first.td_loss)
        self.assertTrue(second.causal_update_applied)
        self.assertEqual(second.rule_batch_size, 1)

    def test_invalid_causal_config_and_buffer_are_rejected(self):
        online = _OffsetQModel()
        target = _OffsetQModel()
        with self.assertRaises(TypeError):
            DQNTrainer(
                online_model=online,
                target_model=target,
                replay_buffer=_td_replay(),
                optimizer=torch.optim.SGD(online.parameters(), lr=0.1),
                causal_replay_buffer=object(),
            )
        with self.assertRaises(ValueError):
            DQNTrainer(
                online_model=online,
                target_model=target,
                replay_buffer=_td_replay(),
                optimizer=torch.optim.SGD(online.parameters(), lr=0.1),
                config=DQNTrainerConfig(causal_update_interval=0),
            )


if __name__ == '__main__':
    unittest.main()
