"""自定义场景实验室的真实物理与Reward V2评估服务。"""

from __future__ import annotations

import math
from numbers import Integral, Real
from threading import Lock

import torch

from daxigua.core.rules import (
    MAX_FRUIT_LEVEL,
    SPAWN_FRUIT_MAX_LEVEL,
    SPAWN_FRUIT_MIN_LEVEL,
    merged_fruit_physics_radius,
)

from .config import SimulatorConfig
from .spatial_reward import (
    AccessibleSpaceCalculator,
    SpatialRewardConfig,
    diagnose_spatial_reward,
)
from .vector import TensorVectorSimulator


def _finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _integer(name, value, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return value


def validate_scenario(scene):
    """校验浏览器场景并返回后端可直接装载的规范数据。"""

    if not isinstance(scene, dict):
        raise TypeError('scene must be an object')
    fps = _integer('fps', scene.get('fps', 120))
    if fps not in (30, 120):
        raise ValueError('fps must be 30 or 120')
    queue = scene.get('queue')
    if not isinstance(queue, list) or len(queue) != 4:
        raise ValueError('queue must contain q0 through q3')
    queue = tuple(
        _integer(
            f'queue[{index}]',
            value,
            minimum=SPAWN_FRUIT_MIN_LEVEL,
            maximum=SPAWN_FRUIT_MAX_LEVEL,
        )
        for index, value in enumerate(queue)
    )
    raw_fruits = scene.get('fruits')
    if not isinstance(raw_fruits, list):
        raise ValueError('fruits must be an array')
    if len(raw_fruits) > 64:
        raise ValueError('scenario cannot contain more than 64 fruits')

    fruits = []
    fruit_ids = set()
    for index, raw in enumerate(raw_fruits):
        if not isinstance(raw, dict):
            raise TypeError(f'fruits[{index}] must be an object')
        level = _integer(
            f'fruits[{index}].level',
            raw.get('level'),
            minimum=1,
            maximum=MAX_FRUIT_LEVEL,
        )
        fruit_id = _integer(
            f'fruits[{index}].id', raw.get('id', index + 1), minimum=1
        )
        if fruit_id in fruit_ids:
            raise ValueError('fruit ids must be unique')
        fruit_ids.add(fruit_id)
        radius = _finite_number(
            f'fruits[{index}].physics_radius',
            raw.get('physics_radius', merged_fruit_physics_radius(level)),
        )
        if radius <= 0.0:
            raise ValueError('fruit physics radius must be positive')
        fruits.append({
            'id': fruit_id,
            'level': level,
            'physics_radius': radius,
            'x': _finite_number(f'fruits[{index}].x', raw.get('x')),
            'y': _finite_number(f'fruits[{index}].y', raw.get('y')),
            'vx': _finite_number(f'fruits[{index}].vx', raw.get('vx', 0.0)),
            'vy': _finite_number(f'fruits[{index}].vy', raw.get('vy', 0.0)),
            'angle': _finite_number(
                f'fruits[{index}].angle', raw.get('angle', 0.0)
            ),
            'angular_velocity': _finite_number(
                f'fruits[{index}].angular_velocity',
                raw.get('angular_velocity', 0.0),
            ),
            'age_frames': _integer(
                f'fruits[{index}].age_frames',
                raw.get('age_frames', 0),
                minimum=0,
            ),
        })
    return {
        'name': str(scene.get('name', '未命名场景'))[:80],
        'fps': fps,
        'queue': queue,
        'fruits': tuple(fruits),
        'probe_action': _integer(
            'probe_action', scene.get('probe_action', 10), minimum=0, maximum=20
        ),
        'score': _integer('score', scene.get('score', 0), minimum=0),
        'step_count': _integer(
            'step_count', scene.get('step_count', 0), minimum=0
        ),
    }


def _simulator_config(fps, *, device):
    common = {
        'max_fruits': 64,
        'action_count': 21,
        'queue_length': 4,
        'use_cuda_extension': torch.device(device).type == 'cuda',
    }
    if fps == 30:
        return SimulatorConfig.training_fast(**common)
    return SimulatorConfig.high_fidelity_fast(**common)


def _load_scenario(simulator, scene):
    """把同一规范场景复制到21个动作环境。"""

    device = simulator.device
    batch_size = simulator.num_envs
    seeds = torch.full(
        (batch_size,), 20260806, dtype=torch.int64, device=device
    )
    simulator.reset(seeds=seeds, fruit_queue=scene['queue'])
    fruit_count = len(scene['fruits'])
    if fruit_count:
        positions = torch.tensor(
            [(fruit['x'], fruit['y']) for fruit in scene['fruits']],
            dtype=torch.float32,
            device=device,
        )
        velocities = torch.tensor(
            [(fruit['vx'], fruit['vy']) for fruit in scene['fruits']],
            dtype=torch.float32,
            device=device,
        )
        levels = torch.tensor(
            [fruit['level'] for fruit in scene['fruits']],
            dtype=torch.int64,
            device=device,
        )
        radii = torch.tensor(
            [fruit['physics_radius'] for fruit in scene['fruits']],
            dtype=torch.float32,
            device=device,
        )
        ids = torch.tensor(
            [fruit['id'] for fruit in scene['fruits']],
            dtype=torch.int64,
            device=device,
        )
        angles = torch.tensor(
            [fruit['angle'] for fruit in scene['fruits']],
            dtype=torch.float32,
            device=device,
        )
        angular_velocities = torch.tensor(
            [fruit['angular_velocity'] for fruit in scene['fruits']],
            dtype=torch.float32,
            device=device,
        )
        ages = torch.tensor(
            [fruit['age_frames'] for fruit in scene['fruits']],
            dtype=torch.int64,
            device=device,
        )
        slots = slice(0, fruit_count)
        simulator.positions[:, slots] = positions[None, ...]
        simulator.velocities[:, slots] = velocities[None, ...]
        simulator.levels[:, slots] = levels[None, ...]
        simulator.physics_radii[:, slots] = radii[None, ...]
        simulator.fruit_ids[:, slots] = ids[None, ...]
        simulator.angles[:, slots] = angles[None, ...]
        simulator.angular_velocities[:, slots] = angular_velocities[None, ...]
        simulator.age_frames[:, slots] = ages[None, ...]
        simulator.active[:, slots] = True
        masses = simulator._mass_table[levels]
        simulator.masses[:, slots] = masses[None, ...]
        simulator.inverse_masses[:, slots] = masses.reciprocal()[None, ...]
        inertias = 0.5 * masses * radii.square()
        simulator.inverse_inertias[:, slots] = inertias.reciprocal()[None, ...]
        simulator.next_fruit_id.fill_(int(ids.max().item()) + 1)
    simulator.score.fill_(scene['score'])
    simulator.last_score.fill_(scene['score'])
    simulator.step_count.fill_(scene['step_count'])


def _float_list(values, digits=5):
    return [round(float(value), digits) for value in values]


def _result_fruits(observation, env_index):
    rows = torch.nonzero(
        observation.active[env_index], as_tuple=False
    ).flatten()
    fruits = []
    for slot in rows.tolist():
        fruits.append({
            'id': int(observation.fruit_ids[env_index, slot].item()),
            'level': int(observation.levels[env_index, slot].item()),
            'x': round(float(observation.positions[env_index, slot, 0]), 3),
            'y': round(float(observation.positions[env_index, slot, 1]), 3),
            'vx': round(float(observation.velocities[env_index, slot, 0]), 3),
            'vy': round(float(observation.velocities[env_index, slot, 1]), 3),
            'physics_radius': round(
                float(observation.physics_radii[env_index, slot]), 3
            ),
        })
    return fruits


class ScenarioLabEvaluator:
    """复用21环境模拟器，串行处理浏览器场景评估请求。"""

    def __init__(self, *, device='cpu', reward_scale=1.0):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        self.reward_config = SpatialRewardConfig(reward_scale=reward_scale)
        self._runtimes = {}
        self._lock = Lock()

    def _runtime(self, fps):
        runtime = self._runtimes.get(fps)
        if runtime is None:
            config = _simulator_config(fps, device=self.device)
            simulator = TensorVectorSimulator(
                21, config=config, device=self.device
            )
            calculator = AccessibleSpaceCalculator(
                config,
                device=self.device,
                reward_config=self.reward_config,
            )
            runtime = (simulator, calculator)
            self._runtimes[fps] = runtime
        return runtime

    @torch.inference_mode()
    def evaluate(self, scene, *, mode='all'):
        scene = validate_scenario(scene)
        if mode not in ('all', 'probe'):
            raise ValueError('mode must be all or probe')
        with self._lock:
            simulator, calculator = self._runtime(scene['fps'])
            _load_scenario(simulator, scene)
            previous = simulator.observe().clone()
            result = simulator.step(
                torch.arange(21, dtype=torch.int64, device=self.device)
            )
            diagnostic = diagnose_spatial_reward(
                calculator, previous, result
            )
            return self._payload(scene, result, diagnostic, mode=mode)

    def _payload(self, scene, result, diagnostic, *, mode):
        # 统一复制到CPU后组装JSON，避免逐字段触发大量CUDA同步。
        before = diagnostic.before
        after = diagnostic.after
        before_levels = before.candidate_levels.detach().cpu()
        before_x = before.drop_x.detach().cpu()
        before_depths = before.depths.detach().cpu()
        before_areas = before.normalized_areas.detach().cpu()
        after_x = after.drop_x.detach().cpu()
        after_depths = after.depths.detach().cpu()
        after_areas = after.normalized_areas.detach().cpu()
        per_slot_raw_delta = diagnostic.per_slot_raw_delta.detach().cpu()
        per_slot_compensation = (
            diagnostic.per_slot_compensation.detach().cpu()
        )
        per_slot_unscaled = (
            diagnostic.per_slot_unscaled_reward.detach().cpu()
        )
        previous_potential = diagnostic.previous_potential.detach().cpu()
        next_potential = diagnostic.next_potential.detach().cpu()
        raw_space_delta = diagnostic.raw_space_delta.detach().cpu()
        compensation = diagnostic.compensation.detach().cpu()
        reward = diagnostic.reward.detach().cpu()
        terminal = diagnostic.terminal.detach().cpu()
        drop_x = result.drop.drop_x.detach().cpu()
        score_delta = result.physics.score_delta.detach().cpu()
        stable = result.physics.stable.detach().cpu()
        settle_timeout = result.physics.settle_timeout.detach().cpu()
        frames_simulated = result.physics.frames_simulated.detach().cpu()
        observation = result.observation
        observation_cpu = type(observation)(**{
            field_name: getattr(observation, field_name).detach().cpu()
            for field_name in observation.__dataclass_fields__
        })
        weights = self.reward_config.queue_weights
        actions = []
        for action_index in range(21):
            slots = []
            for slot_index in range(3):
                slots.append({
                    'queue_slot': slot_index + 1,
                    'level': int(before_levels[action_index, slot_index]),
                    'weight': weights[slot_index],
                    'before': {
                        'drop_x': _float_list(
                            before_x[action_index, slot_index].tolist(), 3
                        ),
                        'depths': _float_list(
                            before_depths[action_index, slot_index].tolist(), 3
                        ),
                        'normalized_area': round(float(
                            before_areas[action_index, slot_index]
                        ), 6),
                    },
                    'after': {
                        'drop_x': _float_list(
                            after_x[action_index, slot_index].tolist(), 3
                        ),
                        'depths': _float_list(
                            after_depths[action_index, slot_index].tolist(), 3
                        ),
                        'normalized_area': round(float(
                            after_areas[action_index, slot_index]
                        ), 6),
                        'effective_normalized_area': round(
                            0.0 if terminal[action_index]
                            else float(after_areas[action_index, slot_index]),
                            6,
                        ),
                    },
                    'raw_delta': round(float(
                        per_slot_raw_delta[action_index, slot_index]
                    ), 6),
                    'compensation': round(float(
                        per_slot_compensation[action_index, slot_index]
                    ), 6),
                    'unscaled_reward': round(float(
                        per_slot_unscaled[action_index, slot_index]
                    ), 6),
                })
            actions.append({
                'action': action_index,
                'drop_x': round(float(drop_x[action_index]), 3),
                'reward': round(float(reward[action_index]), 6),
                'previous_potential': round(float(
                    previous_potential[action_index]
                ), 6),
                'next_potential': round(float(
                    next_potential[action_index]
                ), 6),
                'raw_space_delta': round(float(
                    raw_space_delta[action_index]
                ), 6),
                'compensation': round(float(
                    compensation[action_index]
                ), 6),
                'score_delta': int(score_delta[action_index]),
                'done': bool(terminal[action_index]),
                'stable': bool(stable[action_index]),
                'settle_timeout': bool(settle_timeout[action_index]),
                'frames_simulated': int(frames_simulated[action_index]),
                'space_slots': slots,
                'result_fruits': _result_fruits(observation_cpu, action_index),
            })
        best_action = max(
            range(21), key=lambda index: actions[index]['reward']
        )
        selected_action = (
            scene['probe_action'] if mode == 'probe' else best_action
        )
        return {
            'format_version': 2,
            'reward_version': 'spatial_v2',
            'reward_scale': self.reward_config.reward_scale,
            'physics_fps': scene['fps'],
            'mode': mode,
            'best_action': best_action,
            'selected_action': selected_action,
            'queue_before': list(scene['queue']),
            'aligned_future_levels': list(scene['queue'][1:4]),
            'queue_weights': list(weights),
            'actions': actions,
            'message': (
                '结果来自真实物理与Reward V2后端；新随机q3未参与当前奖励。'
            ),
        }


__all__ = ['ScenarioLabEvaluator', 'validate_scenario']
