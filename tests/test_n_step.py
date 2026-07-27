"""3-step return、Double DQN target 与冷 replay 兼容测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from daxigua_rl.graph.tensor import GraphTensor
from daxigua_rl.training import (
    DQNTrainer,
    DQNTrainerConfig,
    NStepTransitionAccumulator,
    ReplayBuffer,
    TensorTransition,
)


def _graph(action_count=3):
    """构造无需物理环境的最小合法动作图。"""

    return GraphTensor(
        node_features=torch.zeros(
            (action_count, 1),
            dtype=torch.float16,
        ),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_features=torch.empty((0, 1), dtype=torch.float16),
        action_node_indices=torch.arange(
            action_count,
            dtype=torch.long,
        ),
        action_indices=torch.arange(
            action_count,
            dtype=torch.long,
        ),
        node_feature_names=('feature',),
        edge_feature_names=('edge_feature',),
    )


def _transition(
        graph,
        next_graph,
        reward,
        *,
        terminated=False,
        truncated=False,
        bootstrap_steps=1):
    return TensorTransition(
        graph=graph,
        action_offset=0,
        reward=reward,
        next_graph=next_graph,
        terminated=terminated,
        truncated=truncated,
        bootstrap_steps=bootstrap_steps,
    )


class _FixedActionModel(nn.Module):
    """按 action_index 返回可控 Q，并记录 forward 时的 train/eval mode。"""

    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(
            torch.tensor(values, dtype=torch.float32)
        )
        self.forward_modes = []

    def forward(self, graph):
        self.forward_modes.append(self.training)
        indices = graph.action_indices.to(device=self.values.device)
        return self.values.index_select(0, indices)


class TensorTransitionHorizonTest(unittest.TestCase):
    def test_bootstrap_steps_is_strictly_positive_integer(self):
        graph = _graph()

        with self.assertRaises(TypeError):
            _transition(
                graph,
                graph,
                1.0,
                bootstrap_steps=True,
            )
        with self.assertRaises(TypeError):
            _transition(
                graph,
                graph,
                1.0,
                bootstrap_steps=1.5,
            )
        with self.assertRaises(ValueError):
            _transition(
                graph,
                graph,
                1.0,
                bootstrap_steps=0,
            )

    def test_transition_cannot_be_terminal_and_truncated(self):
        graph = _graph()
        with self.assertRaises(ValueError):
            _transition(
                graph,
                None,
                1.0,
                terminated=True,
                truncated=True,
            )


class NStepTransitionAccumulatorTest(unittest.TestCase):
    def test_normal_three_step_window_spans_append_calls(self):
        graphs = tuple(_graph() for _ in range(4))
        accumulator = NStepTransitionAccumulator(
            n_step=3,
            gamma=0.5,
        )

        first = accumulator.append(_transition(
            graphs[0],
            graphs[1],
            1.0,
        ))
        second = accumulator.append(_transition(
            graphs[1],
            graphs[2],
            2.0,
        ))
        third = accumulator.append(_transition(
            graphs[2],
            graphs[3],
            4.0,
        ))

        self.assertEqual(first, ())
        self.assertEqual(second, ())
        self.assertEqual(len(third), 1)
        emitted = third[0]
        self.assertIs(emitted.graph, graphs[0])
        self.assertIs(emitted.next_graph, graphs[3])
        self.assertAlmostEqual(emitted.reward, 1.0 + 0.5 * 2.0 + 0.25 * 4.0)
        self.assertEqual(emitted.bootstrap_steps, 3)
        self.assertFalse(emitted.done)
        # 滑动窗口仍保留最近两条，证明 collect 调用边界不能隐式 flush。
        self.assertEqual(accumulator.pending_count, 2)

    def test_terminal_flushes_three_two_one_without_bootstrap(self):
        graphs = tuple(_graph() for _ in range(3))
        accumulator = NStepTransitionAccumulator(
            n_step=3,
            gamma=0.5,
        )
        accumulator.append(_transition(
            graphs[0],
            graphs[1],
            1.0,
        ))
        accumulator.append(_transition(
            graphs[1],
            graphs[2],
            2.0,
        ))

        emitted = accumulator.append(_transition(
            graphs[2],
            None,
            4.0,
            terminated=True,
        ))

        self.assertEqual(
            tuple(item.bootstrap_steps for item in emitted),
            (3, 2, 1),
        )
        self.assertEqual(
            tuple(item.reward for item in emitted),
            (3.0, 4.0, 4.0),
        )
        self.assertTrue(all(item.terminated for item in emitted))
        self.assertTrue(all(item.next_graph is None for item in emitted))
        self.assertTrue(all(not item.can_bootstrap for item in emitted))
        self.assertEqual(accumulator.pending_count, 0)

    def test_truncated_flush_preserves_final_graph_and_short_horizons(self):
        graphs = tuple(_graph() for _ in range(4))
        accumulator = NStepTransitionAccumulator(
            n_step=3,
            gamma=0.5,
        )
        accumulator.append(_transition(
            graphs[0],
            graphs[1],
            1.0,
        ))
        accumulator.append(_transition(
            graphs[1],
            graphs[2],
            2.0,
        ))

        emitted = accumulator.append(_transition(
            graphs[2],
            graphs[3],
            4.0,
            truncated=True,
        ))

        self.assertEqual(
            tuple(item.bootstrap_steps for item in emitted),
            (3, 2, 1),
        )
        self.assertTrue(all(item.truncated for item in emitted))
        self.assertTrue(all(not item.terminated for item in emitted))
        self.assertTrue(all(item.next_graph is graphs[3] for item in emitted))
        self.assertTrue(all(item.can_bootstrap for item in emitted))
        self.assertEqual(accumulator.pending_count, 0)

    def test_accumulator_rejects_already_aggregated_input(self):
        graph = _graph()
        accumulator = NStepTransitionAccumulator(n_step=3, gamma=0.99)
        with self.assertRaises(ValueError):
            accumulator.append(_transition(
                graph,
                graph,
                1.0,
                bootstrap_steps=2,
            ))


class ReplaySegmentNstepCompatibilityTest(unittest.TestCase):
    def test_version_two_round_trip_preserves_bootstrap_steps(self):
        graph = _graph()
        transition = _transition(
            graph,
            graph,
            1.5,
            bootstrap_steps=3,
        )
        replay = ReplayBuffer(capacity=4)

        payload = replay._pack_segment((transition,))
        self.assertEqual(payload['version'], 2)
        self.assertEqual(payload['records'][0][-1], 3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'segment.pt'
            torch.save(payload, path)
            restored = replay._load_segment({
                'path': str(path),
                'count': 1,
            })[0]

        self.assertEqual(restored.bootstrap_steps, 3)
        self.assertAlmostEqual(restored.reward, 1.5)

    def test_version_one_segment_defaults_to_single_step(self):
        graph = _graph()
        legacy_payload = {
            'version': 1,
            'graphs': (graph,),
            'records': ((0, 0, 2.0, 0, False, False),),
        }
        replay = ReplayBuffer(capacity=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'legacy_segment.pt'
            torch.save(legacy_payload, path)
            restored = replay._load_segment({
                'path': str(path),
                'count': 1,
            })[0]

        self.assertEqual(restored.bootstrap_steps, 1)
        self.assertAlmostEqual(restored.reward, 2.0)


class DoubleDQNTargetTest(unittest.TestCase):
    def _trainer(self, online_values, target_values, *, gamma=0.5, n_step=3):
        online = _FixedActionModel(online_values)
        target = _FixedActionModel(target_values)
        trainer = DQNTrainer(
            online_model=online,
            target_model=target,
            replay_buffer=ReplayBuffer(capacity=4),
            optimizer=torch.optim.Adam(online.parameters(), lr=1e-3),
            config=DQNTrainerConfig(
                gamma=gamma,
                n_step=n_step,
                batch_size=1,
                sync_target_on_init=False,
            ),
        )
        return trainer, online, target

    def test_online_selects_action_and_target_network_only_estimates_it(self):
        graph = _graph()
        # online 选择 action 1；target 自己若取 max 会选择 action 0。
        trainer, online, _target = self._trainer(
            online_values=(0.0, 5.0, 1.0),
            target_values=(10.0, 2.0, 3.0),
        )
        online.train()
        transition = _transition(
            graph,
            graph,
            1.0,
            bootstrap_steps=2,
        )

        targets, count = trainer._compute_target_values(
            (transition,),
            selected_q=torch.zeros(1),
        )

        self.assertEqual(count, 1)
        # 1 + gamma**2 * target(action selected by online)
        self.assertAlmostEqual(float(targets[0]), 1.0 + 0.25 * 2.0)
        self.assertTrue(online.training)
        self.assertEqual(online.forward_modes, [False])

    def test_terminal_does_not_bootstrap_but_truncated_does(self):
        graph = _graph()
        trainer, _online, _target = self._trainer(
            online_values=(0.0, 5.0, 1.0),
            target_values=(10.0, 2.0, 3.0),
        )
        truncated = _transition(
            graph,
            graph,
            1.0,
            truncated=True,
            bootstrap_steps=2,
        )
        terminal = _transition(
            graph,
            None,
            3.0,
            terminated=True,
            bootstrap_steps=2,
        )

        targets, count = trainer._compute_target_values(
            (truncated, terminal),
            selected_q=torch.zeros(2),
        )

        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(targets[0]), 1.0 + 0.25 * 2.0)
        self.assertAlmostEqual(float(targets[1]), 3.0)

    def test_trainer_rejects_sample_beyond_configured_horizon(self):
        graph = _graph()
        trainer, _online, _target = self._trainer(
            online_values=(0.0, 5.0, 1.0),
            target_values=(10.0, 2.0, 3.0),
            n_step=3,
        )
        transition = _transition(
            graph,
            graph,
            1.0,
            bootstrap_steps=4,
        )

        with self.assertRaisesRegex(ValueError, 'exceeds trainer n_step'):
            trainer._compute_target_values(
                (transition,),
                selected_q=torch.zeros(1),
            )


if __name__ == '__main__':
    unittest.main()
