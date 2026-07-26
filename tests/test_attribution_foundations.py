"""完整状态归因依赖的终止、半径和轨迹身份基础测试。"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from daxigua.core.engine import HeadlessGame
from daxigua.core.rules import (
    dropped_fruit_physics_radius,
    fruit_radius,
    merged_fruit_physics_radius,
)
from daxigua.core.state import ActionCandidate, BoardGeometry, FruitState, GameState
from daxigua_rl.env import DaxiguaEnv, DaxiguaEnvConfig
from daxigua_rl.graph import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, GraphBuilder
from daxigua_rl.graph.tensor import graph_to_tensor
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.playable_adapter import board_action_candidates, board_game_state
from daxigua_rl.training import (
    DQNTrainer,
    DQNTrainerConfig,
    ReplayBuffer,
    RolloutCollector,
    TensorTransition,
    TransitionKey,
)


class TerminationSemanticsTest(unittest.TestCase):
    """验证物理稳定边界和 DQN bootstrap 边界不会再混淆。"""

    def _initial_graph(self):
        game = HeadlessGame(seed=0)
        state = game.get_state()
        graph = GraphBuilder().build(state, game.get_action_candidates(3))
        return graph_to_tensor(graph)

    def test_stable_requires_the_full_consecutive_frame_window(self):
        """最后一帧瞬时稳定不能冒充连续稳定。"""

        game = HeadlessGame(gravity=(0, 0), seed=0)

        truncated = game.advance_physics(max_frames=1, stable_frames=2)
        self.assertFalse(truncated.stable)
        self.assertTrue(truncated.truncated)

        settled = game.advance_physics(max_frames=2, stable_frames=2)
        self.assertTrue(settled.stable)
        self.assertFalse(settled.truncated)

    def test_invalid_physics_limits_are_rejected(self):
        game = HeadlessGame(seed=0)

        with self.assertRaises(ValueError):
            game.advance_physics(max_frames=0)
        with self.assertRaises(ValueError):
            game.advance_physics(max_frames=1, stable_frames=0)

        fixed_step = game.advance_physics(
            max_frames=1,
            until_stable=False,
            stable_frames=0,
        )
        self.assertFalse(fixed_step.truncated)

    def test_stability_streak_resets_and_terminal_wins_over_frame_limit(self):
        game = HeadlessGame(gravity=(0, 0), seed=0)
        stable_values = iter((True, False, True, True))
        game._is_stable = lambda: next(stable_values)

        settled = game.advance_physics(max_frames=4, stable_frames=2)
        self.assertEqual(settled.frames_simulated, 4)
        self.assertTrue(settled.stable)

        terminal_game = HeadlessGame(gravity=(0, 0), seed=0)

        def terminate():
            terminal_game.alive = False
            return True

        terminal_game.check_fail = terminate
        terminal = terminal_game.advance_physics(max_frames=1, stable_frames=2)
        self.assertTrue(terminal.done)
        self.assertFalse(terminal.truncated)

    def test_truncated_transition_keeps_bootstrap_graph(self):
        graph = self._initial_graph()

        truncated = TensorTransition(
            graph=graph,
            action_offset=0,
            reward=1.0,
            next_graph=graph,
            terminated=False,
            truncated=True,
        )
        terminal = TensorTransition(
            graph=graph,
            action_offset=0,
            reward=1.0,
            next_graph=None,
            terminated=True,
            truncated=False,
        )

        self.assertTrue(truncated.done)
        self.assertTrue(truncated.can_bootstrap)
        self.assertTrue(terminal.done)
        self.assertFalse(terminal.can_bootstrap)

        with self.assertRaises(ValueError):
            TensorTransition(
                graph=graph,
                action_offset=0,
                reward=1.0,
                next_graph=None,
                terminated=False,
                truncated=True,
            )

    def test_dqn_target_bootstraps_truncated_but_not_terminal(self):
        graph = self._initial_graph()
        truncated = TensorTransition(
            graph=graph,
            action_offset=0,
            reward=1.0,
            next_graph=graph,
            terminated=False,
            truncated=True,
        )
        terminal = TensorTransition(
            graph=graph,
            action_offset=0,
            reward=3.0,
            next_graph=None,
            terminated=True,
            truncated=False,
        )
        online_model = GNNQNetwork(hidden_dim=16, message_layers=1)
        target_model = GNNQNetwork(hidden_dim=16, message_layers=1)
        with torch.no_grad():
            for parameter in target_model.parameters():
                parameter.zero_()
            target_model.q_head[-1].bias.fill_(2.0)

        trainer = DQNTrainer(
            online_model=online_model,
            target_model=target_model,
            replay_buffer=ReplayBuffer(capacity=2, seed=0),
            optimizer=torch.optim.Adam(online_model.parameters(), lr=1e-4),
            config=DQNTrainerConfig(
                batch_size=2,
                gamma=0.9,
                sync_target_on_init=False,
            ),
        )
        targets, bootstrap_count = trainer._compute_target_values(
            (truncated, terminal),
            selected_q=torch.zeros(2),
        )

        self.assertEqual(bootstrap_count, 1)
        self.assertAlmostEqual(float(targets[0]), 1.0 + 0.9 * 2.0)
        self.assertAlmostEqual(float(targets[1]), 3.0)

    def test_cold_replay_preserves_truncated_bootstrap_graph(self):
        graph = self._initial_graph()
        truncated = TensorTransition(
            graph=graph,
            action_offset=0,
            reward=1.0,
            next_graph=graph,
            terminated=False,
            truncated=True,
        )
        terminal = TensorTransition(
            graph=graph,
            action_offset=0,
            reward=0.0,
            next_graph=None,
            terminated=True,
            truncated=False,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            replay_buffer = ReplayBuffer(
                capacity=2,
                hot_capacity=1,
                cold_dir=Path(tmp_dir) / 'cold',
                segment_size=1,
                cold_cache_size=1,
            )
            replay_buffer.push(truncated)
            replay_buffer.push(terminal)
            restored = replay_buffer.to_tuple()[0]

        self.assertTrue(restored.truncated)
        self.assertIsNotNone(restored.next_graph)
        self.assertTrue(restored.can_bootstrap)

    def test_collector_builds_next_graph_before_resetting_truncation(self):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(
            action_count=3,
            max_physics_frames=1,
            stable_frames=2,
        ))
        replay_buffer = ReplayBuffer(capacity=4, seed=0)
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay_buffer,
            seed=0,
            worker_id=7,
        )

        stats = collector.collect_steps(2, epsilon=1.0)
        transitions = replay_buffer.to_tuple()

        self.assertEqual(stats.truncated_episodes, 2)
        self.assertEqual(
            tuple(key.as_tuple() for key in stats.transition_keys),
            ((7, 0, 0), (7, 1, 0)),
        )
        self.assertTrue(all(item.truncated for item in transitions))
        self.assertTrue(all(item.next_graph is not None for item in transitions))
        self.assertTrue(all(item.can_bootstrap for item in transitions))
        self.assertGreater(
            transitions[0].next_graph.num_nodes,
            transitions[1].graph.num_nodes,
        )


class PhysicsRadiusTest(unittest.TestCase):
    """验证显示半径和真实碰撞半径同时且正确地公开。"""

    def test_radius_rules_keep_source_dependent_physics_values(self):
        self.assertEqual(fruit_radius(3), 42)
        self.assertEqual(dropped_fruit_physics_radius(3), 40)
        self.assertEqual(merged_fruit_physics_radius(3), 41)

    def test_legacy_state_construction_falls_back_to_display_radius(self):
        fruit = FruitState(
            fruit_id=1,
            level=3,
            radius=42,
            x=100,
            y=200,
            vx=0,
            vy=0,
            angle=0,
            angular_velocity=0,
            age_frames=0,
            stable=True,
            distance_to_left_wall=38,
            distance_to_right_wall=318,
            distance_to_floor=438,
            distance_to_danger_line=58,
        )
        action = ActionCandidate(
            action_index=0,
            drop_x=100,
            normalized_drop_x=0,
            current_level=3,
            current_radius=42,
        )

        self.assertEqual(fruit.physics_radius, 42.0)
        self.assertEqual(action.current_physics_radius, 42.0)

    def test_headless_state_preserves_same_level_source_difference(self):
        game = HeadlessGame(seed=0)
        game.reset(fruit_queue=(3, 1, 1, 1))

        candidate = game.get_action_candidates(1)[0]
        self.assertEqual(candidate.current_radius, 42.0)
        self.assertEqual(candidate.current_physics_radius, 40.0)

        game.drop_at(candidate.drop_x)
        game._create_ball(
            candidate.drop_x + 100,
            game.spawn_y + 150,
            3,
            physics_radius=merged_fruit_physics_radius(3),
        )
        level_three = tuple(
            fruit
            for fruit in game.get_state().board_fruits
            if fruit.level == 3
        )

        self.assertEqual({fruit.radius for fruit in level_three}, {42.0})
        self.assertEqual(
            {fruit.physics_radius for fruit in level_three},
            {40.0, 41.0},
        )
        direct_fruit = next(
            fruit
            for fruit in level_three
            if fruit.physics_radius == 40.0
        )
        self.assertEqual(direct_fruit.distance_to_danger_line, -40.0)
        self.assertEqual(
            game.get_state().max_height,
            game.height - (game.spawn_y - 40),
        )
        expected_area = torch.pi * (40 ** 2 + 41 ** 2)
        playable_area = game.width * (game.height - game.spawn_y)
        self.assertAlmostEqual(
            game.get_state().empty_space_ratio,
            1 - float(expected_area) / playable_area,
        )

    def test_playable_adapter_reads_shape_radius(self):
        ball = SimpleNamespace(
            collision_type=3,
            radius=41,
            body=SimpleNamespace(
                position=(120, 240),
                velocity=(0, 0),
                angular_velocity=0,
                angle=0,
            ),
        )
        board = SimpleNamespace(
            balls=(ball,),
            WIDTH=500,
            HEIGHT=700,
            init_y=100,
            wall_width=20,
            fruit_queue=(3, 1, 1, 1),
            score=0,
            last_score=0,
            alive=True,
            waiting=True,
            i=3,
        )

        state = board_game_state(board)
        candidate = board_action_candidates(board, action_count=1)[0]

        self.assertEqual(state.board_fruits[0].radius, 42.0)
        self.assertEqual(state.board_fruits[0].physics_radius, 41.0)
        self.assertEqual(candidate.current_physics_radius, 40.0)

    def test_graph_geometry_uses_physics_radius_without_dimension_change(self):
        geometry = BoardGeometry(
            width=500,
            height=700,
            spawn_y=100,
            wall_width=20,
            floor_y=680,
        )
        fruit = FruitState(
            fruit_id=1,
            level=3,
            radius=42,
            physics_radius=41,
            x=182,
            y=240,
            vx=0,
            vy=0,
            angle=0,
            angular_velocity=0,
            age_frames=0,
            stable=True,
            distance_to_left_wall=121,
            distance_to_right_wall=257,
            distance_to_floor=399,
            distance_to_danger_line=99,
        )
        action = ActionCandidate(
            action_index=0,
            drop_x=100,
            normalized_drop_x=0,
            current_level=3,
            current_radius=42,
            current_physics_radius=40,
        )
        state = GameState(
            board_fruits=(fruit,),
            fruit_queue=(3,),
            score=0,
            last_score=0,
            step_count=0,
            physics_frame=0,
            done=False,
            geometry=geometry,
            max_height=501,
            fruit_count=1,
            max_level=3,
            empty_space_ratio=0.9,
        )
        builder = GraphBuilder()

        path_features = builder._action_board_edge_features(action, fruit, geometry)
        graph = builder.build(state, (action,))
        radius_index = NODE_FEATURE_NAMES.index('radius')
        board_node = next(
            index
            for index, ref in enumerate(graph.node_refs)
            if ref.node_type == 'board_fruit'
        )
        action_node = graph.action_node_indices[0]
        queue_node = next(
            index
            for index, ref in enumerate(graph.node_refs)
            if ref.node_type == 'queue_fruit'
        )

        self.assertEqual(path_features['is_under_drop_path'], 0.0)
        self.assertLess(path_features['path_overlap_margin'], 0.0)
        self.assertAlmostEqual(
            graph.node_features[board_node][radius_index],
            41 / builder.max_radius,
        )
        self.assertAlmostEqual(
            graph.node_features[action_node][radius_index],
            40 / builder.max_radius,
        )
        self.assertAlmostEqual(
            graph.node_features[queue_node][radius_index],
            40 / builder.max_radius,
        )
        self.assertEqual(graph.node_feature_dim, len(NODE_FEATURE_NAMES))
        self.assertEqual(graph.edge_feature_dim, len(EDGE_FEATURE_NAMES))


class TransitionIdentityTest(unittest.TestCase):
    """验证键在 collect 调用间连续，并在真正 reset 后换 episode。"""

    def test_transition_key_is_hashable_ordered_and_pickle_safe(self):
        first = TransitionKey(0, 1, 2)
        second = TransitionKey(0, 1, 3)

        self.assertLess(first, second)
        self.assertEqual(len({first, pickle.loads(pickle.dumps(first))}), 1)
        self.assertEqual(first.as_tuple(), (0, 1, 2))

        with self.assertRaises(ValueError):
            TransitionKey(-1, 0, 0)
        with self.assertRaises(TypeError):
            TransitionKey(1.5, 0, 0)

    def test_collector_identity_survives_collect_call_boundaries(self):
        game = HeadlessGame(gravity=(0, 0), seed=0)
        game.stable_velocity_epsilon = 100.0
        env = DaxiguaEnv(
            config=DaxiguaEnvConfig(
                action_count=3,
                max_physics_frames=1,
                stable_frames=1,
            ),
            game=game,
        )
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=ReplayBuffer(capacity=4, seed=0),
            seed=0,
            worker_id=5,
        )

        first = collector.collect_steps(1, epsilon=1.0)
        second = collector.collect_steps(1, epsilon=1.0)
        collector.reset()
        after_reset = collector.collect_steps(1, epsilon=1.0)

        self.assertEqual(first.transition_keys[0].as_tuple(), (5, 0, 0))
        self.assertEqual(second.transition_keys[0].as_tuple(), (5, 0, 1))
        self.assertEqual(after_reset.transition_keys[0].as_tuple(), (5, 1, 0))


if __name__ == '__main__':
    unittest.main()
