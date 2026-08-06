"""场景实验室的持久化单环境实时物理会话。"""

from __future__ import annotations

from copy import deepcopy
import math
from queue import Empty, Queue
from threading import Condition, Event, Thread
import time

from daxigua.core import FruitState, fruit_radius

from .config import SimulatorConfig
from .reference import PymunkReferenceGame


class ScenarioLabLiveSession:
    """在独立线程中按固定时间步持续推进一个 Pymunk 世界。"""

    def __init__(self, *, physics_fps=120, publish_fps=60, seed=20260806):
        if physics_fps not in (30, 120):
            raise ValueError('physics_fps must be 30 or 120')
        if not 1 <= int(publish_fps) <= int(physics_fps):
            raise ValueError('publish_fps must be in [1, physics_fps]')
        self.config = SimulatorConfig(
            physics_fps=int(physics_fps),
            max_physics_frames=int(physics_fps) * 6,
            stable_frames=max(1, round(int(physics_fps) * 0.125)),
            max_fruits=64,
            action_count=21,
            queue_length=4,
            use_cuda_extension=False,
        )
        self.publish_fps = int(publish_fps)
        self.game = PymunkReferenceGame(self.config, seed=seed)
        self.paused = False
        self._commands = Queue(maxsize=2048)
        self._condition = Condition()
        self._stop = Event()
        self._thread = None
        self._sequence = 0
        self._pending_events = []
        self._latest = self._build_payload()

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                name='scenario-lab-live-physics',
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

    def execute(self, command, *, timeout=2.0):
        """把命令交给物理线程，并等待命令在帧边界生效。"""

        if not isinstance(command, dict):
            raise TypeError('command must be an object')
        if self._thread is None:
            raise RuntimeError('live session is not running')
        completed = Event()
        outcome = {}
        self._commands.put_nowait((deepcopy(command), completed, outcome))
        if not completed.wait(float(timeout)):
            raise TimeoutError('live command timed out')
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
        physics_interval = 1.0 / self.config.physics_fps
        publish_interval = 1.0 / self.publish_fps
        next_physics = time.perf_counter()
        next_publish = next_physics
        while not self._stop.is_set():
            command_changed = self._drain_commands()
            now = time.perf_counter()
            if command_changed:
                next_publish = now + publish_interval
            changed = False
            if not self.paused and self.game.alive and now >= next_physics:
                due = min(4, int((now - next_physics) / physics_interval) + 1)
                for _ in range(due):
                    result = self.game.advance_frame()
                    self._pending_events.extend(result.merge_events)
                next_physics += due * physics_interval
                changed = True
            elif self.paused or not self.game.alive:
                next_physics = now + physics_interval

            if changed and now >= next_publish:
                self._publish()
                next_publish = now + publish_interval
            elif not changed and next_publish <= now:
                next_publish = now + publish_interval

            wake_at = min(next_physics, next_publish)
            self._stop.wait(max(0.001, min(0.01, wake_at - time.perf_counter())))

    def _drain_commands(self):
        changed = False
        completions = []
        while True:
            try:
                command, completed, outcome = self._commands.get_nowait()
            except Empty:
                break
            try:
                result = self._apply_command(command)
                outcome['result'] = result
                changed = True
            except Exception as error:  # 命令错误要返回HTTP调用方，物理线程继续运行。
                outcome['error'] = error
            finally:
                completions.append(completed)
        if changed:
            self._publish()
        for completed in completions:
            completed.set()
        return changed

    def _apply_command(self, command):
        kind = command.get('type')
        if kind == 'drop':
            level = int(command['level'])
            x = float(command['x'])
            if not math.isfinite(x):
                raise ValueError('drop x must be finite')
            fruit_id = self.game.spawn_fruit(
                level,
                x,
            )
            return {'accepted': True, 'fruit_id': fruit_id}
        if kind == 'remove':
            fruit_id = int(command['fruit_id'])
            return {
                'accepted': self.game.remove_fruit(fruit_id),
                'fruit_id': fruit_id,
            }
        if kind == 'pause':
            self.paused = True
            return {'accepted': True, 'paused': True}
        if kind == 'resume':
            self.paused = False
            return {'accepted': True, 'paused': False}
        if kind == 'clear':
            queue = command.get('queue') or tuple(self.game.fruit_queue)
            self.game.reset(fruit_queue=queue)
            return {'accepted': True}
        if kind == 'load_scene':
            self._replace_from_scene(command['scene'])
            self.paused = bool(command.get('paused', False))
            return {'accepted': True, 'paused': self.paused}
        raise ValueError('unsupported live command type')

    def _replace_from_scene(self, scene):
        geometry = self.config
        fruits = []
        for raw in scene['fruits']:
            radius = float(fruit_radius(raw['level']))
            physics_radius = float(raw['physics_radius'])
            fruits.append(FruitState(
                fruit_id=int(raw['id']),
                level=int(raw['level']),
                radius=radius,
                physics_radius=physics_radius,
                x=float(raw['x']),
                y=float(raw['y']),
                vx=float(raw['vx']),
                vy=float(raw['vy']),
                angle=float(raw['angle']),
                angular_velocity=float(raw['angular_velocity']),
                age_frames=int(raw['age_frames']),
                stable=False,
                distance_to_left_wall=float(raw['x']) - geometry.wall_width,
                distance_to_right_wall=(
                    geometry.board_width - geometry.wall_width - float(raw['x'])
                ),
                distance_to_floor=(
                    geometry.board_height - geometry.wall_width - float(raw['y'])
                ),
                distance_to_danger_line=float(raw['y']) - geometry.spawn_y,
            ))
        self.game.replace_state(
            fruits,
            fruit_queue=scene['queue'],
            score=scene['score'],
            step_count=scene['step_count'],
        )
        self._pending_events.clear()

    def _publish(self):
        payload = self._build_payload()
        self._pending_events.clear()
        with self._condition:
            self._sequence += 1
            payload['sequence'] = self._sequence
            self._latest = payload
            self._condition.notify_all()

    def _build_payload(self):
        state = self.game.get_state()
        return {
            'format_version': 1,
            'sequence': self._sequence,
            'physics_fps': self.config.physics_fps,
            'publish_fps': self.publish_fps,
            'paused': self.paused,
            'stable': self.game.is_stable(),
            'done': state.done,
            'score': state.score,
            'step_count': state.step_count,
            'physics_frame': state.physics_frame,
            'queue': list(state.fruit_queue),
            'fruits': [
                {
                    'id': fruit.fruit_id,
                    'level': fruit.level,
                    'x': round(fruit.x, 3),
                    'y': round(fruit.y, 3),
                    'vx': round(fruit.vx, 3),
                    'vy': round(fruit.vy, 3),
                    'angle': round(fruit.angle, 6),
                    'angular_velocity': round(fruit.angular_velocity, 6),
                    'age_frames': fruit.age_frames,
                    'physics_radius': round(fruit.physics_radius, 3),
                }
                for fruit in state.board_fruits
            ],
            'merge_events': [
                {
                    'new_level': event.new_level,
                    'x': round(event.x, 3),
                    'y': round(event.y, 3),
                    'score_delta': event.score_delta,
                    'source_ids': list(event.source_ids),
                    'new_fruit_id': event.new_fruit_id,
                }
                for event in self._pending_events
            ],
            'timestamp': time.time(),
        }


__all__ = ['ScenarioLabLiveSession']
