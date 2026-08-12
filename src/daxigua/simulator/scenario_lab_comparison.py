"""场景实验室的双环境实时物理对照会话。"""

from __future__ import annotations

from copy import deepcopy
import math
from queue import Empty, Queue
from threading import Condition, Event, Thread
import time

import torch

from .config import SimulatorConfig
from .scenario_lab_live import ScenarioLabLiveSession


COMPARISON_PRESETS = ('backend_parity', 'play_vs_training')


def _profile_payload(lane, *, role, execution):
    return {
        'role': role,
        'backend': lane.backend,
        'device': str(lane.device),
        'physics_fps': lane.config.physics_fps,
        'drop_fast_forward': lane.config.drop_fast_forward,
        'adaptive_collision_substeps': (
            lane.config.adaptive_collision_substeps
        ),
        'max_collision_substeps': lane.config.max_collision_substeps,
        'position_correction': lane.config.position_correction,
        'execution': execution,
    }


def _difference_payload(left, right, first_divergence):
    left_by_id = {fruit['id']: fruit for fruit in left['fruits']}
    right_by_id = {fruit['id']: fruit for fruit in right['fruits']}
    left_ids = set(left_by_id)
    right_ids = set(right_by_id)
    shared_ids = sorted(left_ids & right_ids)
    level_mismatches = []
    max_position = 0.0
    max_velocity = 0.0
    max_angle = 0.0
    max_angular_velocity = 0.0
    worst_position_fruit_id = None
    for fruit_id in shared_ids:
        left_fruit = left_by_id[fruit_id]
        right_fruit = right_by_id[fruit_id]
        if left_fruit['level'] != right_fruit['level']:
            level_mismatches.append(fruit_id)
        position_delta = math.hypot(
            left_fruit['x'] - right_fruit['x'],
            left_fruit['y'] - right_fruit['y'],
        )
        velocity_delta = math.hypot(
            left_fruit['vx'] - right_fruit['vx'],
            left_fruit['vy'] - right_fruit['vy'],
        )
        if position_delta > max_position:
            max_position = position_delta
            worst_position_fruit_id = fruit_id
        max_velocity = max(max_velocity, velocity_delta)
        max_angle = max(
            max_angle,
            abs(left_fruit['angle'] - right_fruit['angle']),
        )
        max_angular_velocity = max(
            max_angular_velocity,
            abs(
                left_fruit['angular_velocity']
                - right_fruit['angular_velocity']
            ),
        )

    left_only = sorted(left_ids - right_ids)
    right_only = sorted(right_ids - left_ids)
    discrete_diverged = bool(
        left_only
        or right_only
        or level_mismatches
        or left['queue'] != right['queue']
        or left['score'] != right['score']
        or left['step_count'] != right['step_count']
        or left['done'] != right['done']
    )
    continuous_diverged = bool(
        max_position > 0.05
        or max_velocity > 0.1
        or max_angle > 1e-4
        or max_angular_velocity > 1e-3
    )
    return {
        'shared_fruit_count': len(shared_ids),
        'left_only_fruit_ids': left_only,
        'right_only_fruit_ids': right_only,
        'level_mismatch_fruit_ids': level_mismatches,
        'queue_equal': left['queue'] == right['queue'],
        'score_delta_right_minus_left': right['score'] - left['score'],
        'fruit_count_delta_right_minus_left': (
            len(right['fruits']) - len(left['fruits'])
        ),
        'max_position_delta': round(max_position, 6),
        'max_velocity_delta': round(max_velocity, 6),
        'max_angle_delta': round(max_angle, 8),
        'max_angular_velocity_delta': round(
            max_angular_velocity, 8
        ),
        'worst_position_fruit_id': worst_position_fruit_id,
        'discrete_diverged': discrete_diverged,
        'continuous_diverged': continuous_diverged,
        'diverged': discrete_diverged or continuous_diverged,
        'first_divergence': deepcopy(first_divergence),
    }


class ScenarioLabComparisonSession:
    """同步驱动两个世界并发布按水果 ID 对齐的实时差异。"""

    def __init__(
            self,
            *,
            preset='play_vs_training',
            play_device='cuda',
            accelerated_device='cuda',
            publish_fps=60,
            seed=20260806):
        if preset not in COMPARISON_PRESETS:
            raise ValueError(f'preset must be one of {COMPARISON_PRESETS}')
        self.play_device = torch.device(play_device)
        self.accelerated_device = torch.device(accelerated_device)
        for device in (self.play_device, self.accelerated_device):
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA is not available')
        self.seed = int(seed)
        self.publish_fps = int(publish_fps)
        if not 1 <= self.publish_fps <= 120:
            raise ValueError('publish_fps must be in [1, 120]')
        self.preset = preset
        self.left = None
        self.right = None
        self._training_trace = None
        self._training_trace_meta = None
        self._training_elapsed_ticks = 0
        self._training_record_index = 0
        self._training_current_record_is_exact = True
        self._training_target_ticks = None
        self._action_in_flight = False
        self._first_divergence = None
        self._commands = Queue(maxsize=256)
        self._condition = Condition()
        self._stop = Event()
        self._thread = None
        self._sequence = 0
        self._tick = 0
        self.paused = True
        self._build_lanes(preset)
        self._latest = self._build_payload()

    def _config(self, *, fps, device):
        factory = (
            SimulatorConfig.training_fast
            if int(fps) == 30
            else SimulatorConfig.high_fidelity_fast
        )
        return factory(
            max_fruits=64,
            action_count=21,
            queue_length=4,
            use_cuda_extension=torch.device(device).type == 'cuda',
            track_action_effects=False,
        )

    def _lane(self, *, fps, device):
        return ScenarioLabLiveSession(
            publish_fps=min(self.publish_fps, int(fps)),
            seed=self.seed,
            device=device,
            config=self._config(fps=fps, device=device),
        )

    def _build_lanes(self, preset):
        if preset == 'backend_parity':
            self.left = self._lane(fps=120, device='cpu')
            self.right = self._lane(
                fps=120,
                device=self.accelerated_device,
            )
        else:
            self.left = self._lane(
                fps=120,
                device=self.play_device,
            )
            self.right = self._lane(
                fps=30,
                device=self.accelerated_device,
            )
        self.preset = preset
        self._training_trace = None
        self._training_trace_meta = None
        self._training_elapsed_ticks = 0
        self._training_record_index = 0
        self._training_current_record_is_exact = True
        self._training_target_ticks = None
        self._action_in_flight = False
        self._first_divergence = None
        self._tick = 0
        self.paused = True

    @property
    def profiles(self):
        if self.preset == 'backend_parity':
            return {
                'left': _profile_payload(
                    self.left,
                    role='Tensor / CPU',
                    execution='incremental_frame',
                ),
                'right': _profile_payload(
                    self.right,
                    role=(
                        'CUDA Kernel'
                        if self.right.device.type == 'cuda'
                        else 'Tensor / CPU fallback'
                    ),
                    execution='incremental_frame',
                ),
            }
        return {
            'left': _profile_payload(
                self.left,
                role='场景实验室实时游玩',
                execution='incremental_frame',
            ),
            'right': _profile_payload(
                self.right,
                role='CUDA大规模训练',
                execution='full_step_trace',
            ),
        }

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                name='scenario-lab-comparison',
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._condition:
            self._condition.notify_all()

    def execute(self, command, *, timeout=5.0):
        if not isinstance(command, dict):
            raise TypeError('command must be an object')
        if self._thread is None:
            raise RuntimeError('comparison session is not running')
        completed = Event()
        outcome = {}
        self._commands.put_nowait((deepcopy(command), completed, outcome))
        if not completed.wait(float(timeout)):
            raise TimeoutError('comparison command timed out')
        if 'error' in outcome:
            raise outcome['error']
        return outcome['result']

    def snapshot(self):
        with self._condition:
            return deepcopy(self._latest)

    def wait_for_snapshot(self, after_sequence, *, timeout=15.0):
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while (
                    self._latest['sequence'] <= int(after_sequence)
                    and not self._stop.is_set()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return deepcopy(self._latest)

    def _run(self):
        interval = 1.0 / 120.0
        publish_every = max(1, round(120 / self.publish_fps))
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            command_changed = self._drain_commands()
            now = time.perf_counter()
            changed = command_changed
            if not self.paused and now >= next_tick:
                due = min(2, int((now - next_tick) / interval) + 1)
                advanced = 0
                for _ in range(due):
                    self._advance_tick()
                    self._tick += 1
                    advanced += 1
                    if self.paused:
                        break
                next_tick += advanced * interval
                changed = True
            elif self.paused:
                next_tick = now + interval
            if changed and (
                    command_changed
                    or self.paused
                    or self._tick % publish_every == 0):
                self._publish()
            remaining = next_tick - time.perf_counter()
            time.sleep(min(0.005, remaining) if remaining > 0 else 0.0)

    def _drain_commands(self):
        changed = False
        completions = []
        while True:
            try:
                command, completed, outcome = self._commands.get_nowait()
            except Empty:
                break
            try:
                outcome['result'] = self._apply_command(command)
                changed = True
            except Exception as error:
                outcome['error'] = error
            finally:
                completions.append(completed)
        if changed:
            self._publish()
        for completed in completions:
            completed.set()
        return changed

    def _advance_tick(self):
        if not bool(self.left.simulator.needs_reset[0].item()):
            physics = self.left.simulator.advance_incremental_frame()
            self.left._pending_events.extend(
                physics.merge_events.to_python(0)
            )
        if self.preset == 'backend_parity':
            if not bool(self.right.simulator.needs_reset[0].item()):
                physics = self.right.simulator.advance_incremental_frame()
                self.right._pending_events.extend(
                    physics.merge_events.to_python(0)
                )
            if (
                    self._action_in_flight
                    and bool(self.left.simulator.incremental_stable()[0])
                    and bool(self.right.simulator.incremental_stable()[0])):
                self._action_in_flight = False
                self.paused = True
        elif self._training_trace is not None:
            self._training_elapsed_ticks += 1
            target_frame = self._training_elapsed_ticks * 30 // 120
            frame_numbers = self._training_trace.frame_numbers[0]
            record_count = int(self._training_trace.record_counts[0])
            while (
                    self._training_record_index + 1 < record_count
                    and int(frame_numbers[self._training_record_index + 1])
                    <= target_frame):
                self._training_record_index += 1
            self._training_current_record_is_exact = (
                self._training_elapsed_ticks % 4 == 0
                and int(frame_numbers[self._training_record_index])
                == target_frame
            )
            if (
                    self._action_in_flight
                    and self._training_current_record_is_exact
                    and self._training_record_index + 1 >= record_count):
                # 停在两条轨迹完全相同的物理时刻，便于直接观察端点差异。
                self._action_in_flight = False
                self.paused = True
        elif self._action_in_flight and self._training_target_ticks is not None:
            # 无 CUDA 时没有逐帧 trace；仍在等价物理时刻冻结 CPU 回退结果。
            self._training_elapsed_ticks += 1
            if self._training_elapsed_ticks >= self._training_target_ticks:
                self._action_in_flight = False
                self.paused = True
        elif not bool(self.right.simulator.needs_reset[0].item()):
            # 手工编辑场景时，两侧都退回到逐帧物理，便于同步检查。
            physics = self.right.simulator.advance_incremental_frame()
            self.right._pending_events.extend(
                physics.merge_events.to_python(0)
            )

    def _apply_command(self, command):
        kind = command.get('type')
        if kind == 'refresh_state':
            return {'accepted': True}
        if kind == 'set_preset':
            preset = str(command.get('preset'))
            if preset not in COMPARISON_PRESETS:
                raise ValueError(
                    f'preset must be one of {COMPARISON_PRESETS}'
                )
            self._build_lanes(preset)
            return {'accepted': True, 'preset': preset, 'reset': True}
        if kind == 'pause':
            self.paused = True
            return {'accepted': True, 'paused': True}
        if kind == 'resume':
            self.paused = False
            return {'accepted': True, 'paused': False}
        if kind in ('clear', 'load_scene'):
            for lane in (self.left, self.right):
                lane._apply_command(command)
            self._reset_comparison_history()
            self.paused = bool(command.get('paused', True))
            return {'accepted': True, 'paused': self.paused}
        if kind in ('drop', 'remove', 'refresh'):
            outcomes = [
                lane._apply_command(command) for lane in (self.left, self.right)
            ]
            self._reset_comparison_history()
            self.paused = True
            return {
                'accepted': all(
                    outcome.get('accepted', True) for outcome in outcomes
                ),
                'left': outcomes[0],
                'right': outcomes[1],
                'note': (
                    'manual_edit_uses_incremental_physics'
                    if self.preset == 'play_vs_training' else None
                ),
            }
        if kind == 'drop_action':
            return self._drop_action(int(command['action']))
        raise ValueError('unsupported comparison command type')

    def _drop_action(self, action):
        if not 0 <= action < self.left.config.action_count:
            raise IndexError('action index out of range')
        if self.preset == 'backend_parity':
            outcomes = [
                lane._apply_command({'type': 'drop_action', 'action': action})
                for lane in (self.left, self.right)
            ]
            self._action_in_flight = True
            self.paused = False
            return {
                'accepted': True,
                'action': action,
                'left': outcomes[0],
                'right': outcomes[1],
            }
        if self._training_trace is not None:
            record_count = int(self._training_trace.record_counts[0])
            if self._training_record_index + 1 < record_count:
                raise RuntimeError(
                    'training trace playback must finish before next action'
                )
        self.paused = False
        self._action_in_flight = True
        left_drop = self.left._apply_command({
            'type': 'drop_action', 'action': action,
        })
        actions = torch.tensor(
            [action], dtype=torch.int64, device=self.right.device
        )
        initial_physics_frame = int(
            self.right.simulator.physics_frame[0].item()
        )
        if self.right.device.type != 'cuda':
            result = self.right.simulator.step(actions)
            self._training_trace = None
            self._training_trace_meta = None
            self._training_elapsed_ticks = 0
            self._training_target_ticks = max(
                1, int(result.physics.frames_simulated[0].item()) * 4
            )
            self._training_current_record_is_exact = False
            return {
                'accepted': True,
                'action': action,
                'left': left_drop,
                'right': {
                    'frames': int(result.physics.frames_simulated[0]),
                    'fast_forwarded_frames': int(
                        result.physics.fast_forwarded_frames[0]
                    ),
                    'trace_records': 0,
                    'note': 'CPU fallback publishes final training state',
                },
            }
        result, trace = self.right.simulator.step_with_trace(
            actions,
            torch.tensor([0], dtype=torch.int64, device=self.right.device),
            frame_stride=1,
        )
        trace = trace.cpu()
        final_payload = self.right._build_payload()
        self._training_trace = trace
        self._training_trace_meta = {
            'initial_physics_frame': initial_physics_frame,
            'final_payload': final_payload,
            'queue': [
                int(value) for value in result.drop.queue_after[0].tolist()
            ],
            'step_count': int(self.right.simulator.step_count[0].item()),
        }
        self._training_elapsed_ticks = 0
        self._training_record_index = 0
        self._training_current_record_is_exact = True
        self._training_target_ticks = None
        return {
            'accepted': True,
            'action': action,
            'left': left_drop,
            'right': {
                'frames': int(result.physics.frames_simulated[0].item()),
                'fast_forwarded_frames': int(
                    result.physics.fast_forwarded_frames[0].item()
                ),
                'trace_records': int(trace.record_counts[0]),
            },
        }

    def _reset_comparison_history(self):
        self._training_trace = None
        self._training_trace_meta = None
        self._training_elapsed_ticks = 0
        self._training_record_index = 0
        self._training_current_record_is_exact = True
        self._training_target_ticks = None
        self._action_in_flight = False
        self._first_divergence = None
        self._tick = 0

    def _trace_payload(self):
        trace = self._training_trace
        meta = self._training_trace_meta
        record_index = self._training_record_index
        count = int(trace.record_counts[0])
        final_record = record_index + 1 >= count
        active = trace.active[0, record_index]
        fruit_ids = trace.fruit_ids[0, record_index]
        slots = torch.nonzero(active, as_tuple=False).flatten()
        if slots.numel():
            slots = slots[torch.argsort(fruit_ids[slots])]
        fruits = []
        for slot in slots.tolist():
            position = trace.positions[0, record_index, slot]
            velocity = trace.velocities[0, record_index, slot]
            fruits.append({
                'id': int(fruit_ids[slot]),
                'level': int(trace.levels[0, record_index, slot]),
                'x': round(float(position[0]), 3),
                'y': round(float(position[1]), 3),
                'vx': round(float(velocity[0]), 3),
                'vy': round(float(velocity[1]), 3),
                'angle': round(
                    float(trace.angles[0, record_index, slot]), 6
                ),
                'angular_velocity': round(float(
                    trace.angular_velocities[0, record_index, slot]
                ), 6),
                'age_frames': 0,
                'physics_radius': round(float(
                    trace.physics_radii[0, record_index, slot]
                ), 3),
            })
        relative_frame = int(trace.frame_numbers[0, record_index])
        final_payload = meta['final_payload']
        return {
            'format_version': 2,
            'sequence': self._sequence,
            'physics_backend': self.right.backend,
            'physics_device': str(self.right.device),
            'training_physics_equivalent': True,
            'physics_fps': 30,
            'publish_fps': self.publish_fps,
            'paused': self.paused,
            'stable': bool(trace.stable[0]) if final_record else False,
            'done': bool(trace.done[0]) if final_record else False,
            'danger_progress': (
                final_payload['danger_progress'] if final_record else 0.0
            ),
            'over_danger_line': any(
                fruit['y'] - fruit['physics_radius']
                < self.right.config.spawn_y
                for fruit in fruits
            ),
            'score': int(trace.scores[0, record_index]),
            'step_count': meta['step_count'],
            'physics_frame': (
                meta['initial_physics_frame'] + relative_frame
            ),
            'queue': list(meta['queue']),
            'fruits': fruits,
            'merge_events': [],
            'timestamp': time.time(),
            'trace': {
                'record_index': record_index,
                'record_count': count,
                'semantic_frame': relative_frame,
                'playback_complete': final_record,
                'contains_fast_forward_gap': bool(
                    count > 1
                    and int(trace.frame_numbers[0, 1]) > 1
                ),
            },
        }

    def _lane_payloads(self):
        left = self.left._build_payload()
        if self.preset == 'play_vs_training' and self._training_trace is not None:
            right = self._trace_payload()
        else:
            right = self.right._build_payload()
        left['profile'] = self.profiles['left']
        right['profile'] = self.profiles['right']
        return left, right

    def _build_payload(self):
        left, right = self._lane_payloads()
        difference_comparable = (
            self.preset == 'backend_parity'
            or (
                self._training_trace is None
                and not self._action_in_flight
            )
            or self._training_current_record_is_exact
        )
        provisional = _difference_payload(
            left, right, self._first_divergence
        )
        if (
                difference_comparable
                and provisional['diverged']
                and self._first_divergence is None):
            self._first_divergence = {
                'sequence': self._sequence + 1,
                'comparison_tick': self._tick,
                'left_physics_frame': left['physics_frame'],
                'right_physics_frame': right['physics_frame'],
                'discrete': provisional['discrete_diverged'],
                'max_position_delta': provisional['max_position_delta'],
                'max_velocity_delta': provisional['max_velocity_delta'],
            }
        difference = _difference_payload(
            left, right, self._first_divergence
        )
        return {
            'format_version': 1,
            'sequence': self._sequence,
            'preset': self.preset,
            'paused': self.paused,
            'comparison_tick': self._tick,
            'profiles': self.profiles,
            'left': left,
            'right': right,
            'difference': difference,
            'difference_comparable': difference_comparable,
            'action_in_progress': self._action_in_flight,
            'timestamp': time.time(),
        }

    def _publish(self):
        payload = self._build_payload()
        self.left._pending_events.clear()
        self.right._pending_events.clear()
        with self._condition:
            self._sequence += 1
            payload['sequence'] = self._sequence
            self._latest = payload
            self._condition.notify_all()


__all__ = ['COMPARISON_PRESETS', 'ScenarioLabComparisonSession']
