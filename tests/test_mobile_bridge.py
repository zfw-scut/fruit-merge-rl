"""移动端纯 Python 状态分析桥的契约测试。"""

import json
from pathlib import Path
import subprocess
import sys
import unittest

from daxigua_mobile import (
    ACTION_COUNT,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    MobileSceneError,
    build_action_candidates,
    build_mobile_graph,
    build_mobile_graph_json,
    scene_to_game_state,
    scene_to_transition_key,
)


ROOT = Path(__file__).resolve().parents[1]


def _scene():
    """返回带有可合成小结构的当前 560x1120 稳定局面。"""

    return {
        'width': 560,
        'height': 1120,
        'spawn_y': 252,
        'q0': {'level': 1},
        'q1': 2,
        'q2': 3,
        'q3': 4,
        'score': 12,
        'last_score': 10,
        'step_count': 7,
        'physics_frame': 960,
        'episode_id': 3,
        'stable_boundary': True,
        'fruits': [
            {
                'fruit_id': 10,
                'level': 1,
                'x': 110.0,
                'y': 1079.0,
                'physics_radius': 20.0,
                'age_frames': 180,
                'stable': True,
            },
            {
                'id': 11,
                'level': 1,
                'x': 151.0,
                'y': 1079.0,
                'physics_radius': 20.0,
                'stable': True,
            },
            {
                'fruit_id': 12,
                'level': 4,
                'x': 360.0,
                'y': 1033.0,
                'collision_radius': 45.0,
                'stable': True,
            },
        ],
    }


class MobileStateConversionTest(unittest.TestCase):
    """验证 Android 普通字段会还原训练时的数据语义。"""

    def test_scene_builds_game_state_actions_and_transition_key(self):
        scene = _scene()
        state = scene_to_game_state(scene)
        actions = build_action_candidates(state)
        transition_key = scene_to_transition_key(scene, state)

        self.assertEqual((state.geometry.width, state.geometry.height), (560, 1120))
        self.assertEqual(state.geometry.spawn_y, 252)
        self.assertEqual(state.geometry.floor_y, 1100)
        self.assertEqual(state.fruit_queue, (1, 2, 3, 4))
        self.assertEqual(state.fruit_count, 3)
        self.assertEqual(state.max_level, 4)
        self.assertEqual(transition_key.as_tuple(), (0, 3, 7))

        self.assertEqual(len(actions), ACTION_COUNT)
        self.assertEqual(
            tuple(action.action_index for action in actions),
            tuple(range(ACTION_COUNT)),
        )
        self.assertTrue(
            all(
                left.drop_x < right.drop_x
                for left, right in zip(actions, actions[1:])
            )
        )
        self.assertEqual(actions[0].normalized_drop_x, 0.0)
        self.assertEqual(actions[-1].normalized_drop_x, 1.0)

        first = state.board_fruits[0]
        self.assertEqual(first.distance_to_left_wall, 70.0)
        self.assertEqual(first.distance_to_floor, 1.0)
        self.assertEqual(first.distance_to_danger_line, 807.0)

    def test_queue_array_and_json_input_are_supported(self):
        scene = _scene()
        scene.pop('q0')
        scene.pop('q1')
        scene.pop('q2')
        scene.pop('q3')
        scene['fruit_queue'] = [1, 2, 3, 4]

        state = scene_to_game_state(json.dumps(scene))

        self.assertEqual(state.fruit_queue, (1, 2, 3, 4))

    def test_invalid_scene_is_rejected_before_analysis(self):
        wrong_geometry = _scene()
        wrong_geometry['width'] = 400
        with self.assertRaisesRegex(MobileSceneError, 'exactly 560x1120'):
            scene_to_game_state(wrong_geometry)

        duplicate = _scene()
        duplicate['fruits'][1]['id'] = 10
        with self.assertRaisesRegex(MobileSceneError, 'must be unique'):
            scene_to_game_state(duplicate)

        wrong_key = _scene()
        wrong_key['transition_key'] = {
            'worker_id': 0,
            'episode_id': 0,
            'step_index': 6,
        }
        state = scene_to_game_state(wrong_key)
        with self.assertRaisesRegex(MobileSceneError, 'must equal'):
            scene_to_transition_key(wrong_key, state)


class MobileGraphContractTest(unittest.TestCase):
    """冻结 ONNX 输入使用的 21 动作、62/47 特征和扁平内存布局。"""

    def test_graph_is_flat_serializable_and_matches_model_abi(self):
        result = build_mobile_graph(_scene())
        node_count, node_dim = result['node_features_shape']
        edge_count = result['edge_index_shape'][1]
        edge_feature_count, edge_dim = result['edge_features_shape']

        self.assertEqual(node_dim, NODE_FEATURE_DIM)
        self.assertEqual(edge_dim, EDGE_FEATURE_DIM)
        self.assertEqual(edge_feature_count, edge_count)
        self.assertEqual(
            len(result['node_features']),
            node_count * NODE_FEATURE_DIM,
        )
        self.assertEqual(len(result['edge_index']), 2 * edge_count)
        self.assertEqual(
            len(result['edge_features']),
            edge_count * EDGE_FEATURE_DIM,
        )
        self.assertEqual(result['action_indices'], list(range(ACTION_COUNT)))
        self.assertEqual(len(result['action_node_indices']), ACTION_COUNT)
        self.assertEqual(len(result['global_node_index']), 1)
        self.assertEqual(len(result['node_feature_names']), NODE_FEATURE_DIM)
        self.assertEqual(len(result['edge_feature_names']), EDGE_FEATURE_DIM)

        # edge_index 的扁平布局是 [sources..., targets...]，所有下标都必须落在图内。
        sources = result['edge_index'][:edge_count]
        targets = result['edge_index'][edge_count:]
        self.assertTrue(all(0 <= value < node_count for value in sources))
        self.assertTrue(all(0 <= value < node_count for value in targets))

        global_index = result['global_node_index'][0]
        global_flag_column = result['node_feature_names'].index(
            'is_global_node'
        )
        self.assertEqual(
            result['node_features'][
                global_index * NODE_FEATURE_DIM + global_flag_column
            ],
            1.0,
        )
        self.assertTrue(result['analysis']['valid_for_attribution'])
        self.assertFalse(result['analysis']['degraded'])

        # 结果必须是标准 JSON；禁止 NaN、dataclass 和 tuple-only 自定义对象。
        encoded = json.dumps(result, allow_nan=False)
        self.assertEqual(json.loads(encoded)['edge_index_shape'], [2, edge_count])

    def test_compact_json_entrypoint_matches_dictionary_entrypoint(self):
        scene_json = json.dumps(_scene())

        dictionary = build_mobile_graph(scene_json)
        encoded = build_mobile_graph_json(scene_json)

        self.assertEqual(json.loads(encoded), dictionary)

    def test_unstable_boundary_is_reported_without_changing_shape_contract(self):
        scene = _scene()
        scene['stable_boundary'] = False

        result = build_mobile_graph(scene)

        self.assertFalse(result['analysis']['valid_for_attribution'])
        self.assertTrue(result['analysis']['degraded'])
        self.assertIn(
            'unstable_boundary',
            result['analysis']['warning_codes'],
        )
        self.assertEqual(len(result['action_node_indices']), ACTION_COUNT)

    def test_import_and_graph_build_do_not_require_desktop_or_torch_modules(self):
        """在隔离解释器中主动阻止重依赖，避免本机已安装依赖掩盖回归。"""

        script = f"""
import importlib.abc
import json
import sys

class ForbiddenImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {{'pygame', 'pymunk', 'torch'}}:
            raise RuntimeError('forbidden mobile dependency: ' + fullname)
        return None

sys.meta_path.insert(0, ForbiddenImport())
sys.path.insert(0, {str(ROOT / 'src')!r})

from daxigua_mobile import build_mobile_graph
result = build_mobile_graph({{
    'fruit_queue': [1, 2, 3, 4],
    'fruits': [],
    'step_count': 0,
}})
assert result['node_features_shape'][1] == 62
assert result['edge_features_shape'][1] == 47
assert len(result['action_node_indices']) == 21
print(json.dumps(result['edge_index_shape']))
"""
        completed = subprocess.run(
            [sys.executable, '-c', script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertTrue(completed.stdout.strip().startswith('[2, '))


if __name__ == '__main__':
    unittest.main()
