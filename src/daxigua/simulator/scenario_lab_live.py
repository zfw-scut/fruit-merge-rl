"""场景实验室的持久化训练同源 Tensor/CUDA 实时物理会话。"""

from __future__ import annotations

from copy import deepcopy
import math
from queue import Empty, Queue
from threading import Condition, Event, Thread
import time

import torch

from daxigua.core import fruit_radius

from .config import SimulatorConfig
from .vector import TensorVectorSimulator


class ScenarioLabLiveSession:
    """在独立线程中逐帧推进一个训练同源 Tensor 环境。"""

    def __init__(
            self,
            *,
            physics_fps=120,
            publish_fps=120,
            seed=20260806,
            device='cpu'):
        if physics_fps not in (30, 120):
            raise ValueError('physics_fps must be 30 or 120')
        if not 1 <= int(publish_fps) <= int(physics_fps):
            raise ValueError('publish_fps must be in [1, physics_fps]')
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')
        factory = (
            SimulatorConfig.training_fast
            if int(physics_fps) == 30
            else SimulatorConfig.high_fidelity_fast
        )
        self.config = factory(
            max_fruits=64,
            action_count=21,
            queue_length=4,
            use_cuda_extension=self.device.type == 'cuda',
            track_action_effects=False,
            # 实时画面不能跳过自由下落；该优化不改变碰撞后的训练物理。
            drop_fast_forward=False,
        )
        self.publish_fps = int(publish_fps)
        self.seed = int(seed)
        self.simulator = TensorVectorSimulator(
            1, config=self.config, device=self.device
        )
        self.simulator.reset(seeds=self.seed)
        if self.device.type == 'cuda':
            # 在物理时钟启动前完成扩展加载、CUDA context 建立和首个 Kernel
            # 调度；否则首帧 JIT/加载延迟会形成永久追帧积压，发布线程只能
            # 每批四帧发布一次，看起来远低于 120 FPS。
            self.simulator.advance_incremental_frame()
            self.simulator.reset(seeds=self.seed)
        self.simulator.reset_incremental_progress(stable=True)
        self.paused = False
        self._commands = Queue(maxsize=2048)
        self._condition = Condition()
        self._stop = Event()
        self._thread = None
        self._sequence = 0
        self._pending_events = []
        self._latest = self._build_payload()

    @property
    def backend(self):
        return 'tensor_cuda' if self.device.type == 'cuda' else 'tensor_cpu'

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
            alive = not bool(self.simulator.needs_reset[0].item())
            if not self.paused and alive and now >= next_physics:
                due = min(4, int((now - next_physics) / physics_interval) + 1)
                for _ in range(due):
                    physics = self.simulator.advance_incremental_frame()
                    self._pending_events.extend(
                        physics.merge_events.to_python(0)
                    )
                    if bool(physics.done[0].item()):
                        break
                next_physics += due * physics_interval
                changed = True
            elif self.paused or not alive:
                next_physics = now + physics_interval

            if changed and (
                    self.publish_fps == self.config.physics_fps
                    or now >= next_publish):
                self._publish()
                next_publish = now + publish_interval

            # 只有物理状态真的推进后才发布；发布时钟已经到点、但物理时钟
            # 尚未到点时，保留这个发布 deadline，等待下一帧后立即发送。
            # 不能把 deadline 向后挪，否则两个时钟的细小漂移会反复吞掉
            # 发布帧。
            wake_at = next_physics
            remaining = wake_at - time.perf_counter()
            # Windows 上 ``threading.Event.wait(<15ms)`` 实际通常会等待约
            # 15.6ms，只能形成约 64Hz 的发布节奏。``time.sleep`` 使用高精度
            # waitable timer；停止信号仍会在至多 10ms 后由循环顶部处理。
            time.sleep(min(0.01, remaining) if remaining > 0.0 else 0.0)

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

    def _apply_command(self, command):
        kind = command.get('type')
        if kind == 'drop':
            level = int(command['level'])
            x = float(command['x'])
            if not math.isfinite(x):
                raise ValueError('drop x must be finite')
            fruit_id, drop_x = self._spawn_manual(level, x)
            return {
                'accepted': True,
                'fruit_id': fruit_id,
                'drop_x': round(drop_x, 3),
            }
        if kind == 'drop_action':
            action = int(command['action'])
            actions = torch.tensor(
                [action], dtype=torch.int64, device=self.device
            )
            drop = self.simulator.begin_incremental_action(actions)
            return {
                'accepted': True,
                'action': action,
                'fruit_id': int(drop.fruit_ids[0].item()),
                'dropped_level': int(drop.dropped_levels[0].item()),
                'drop_x': round(float(drop.drop_x[0].item()), 3),
                'queue_before': [
                    int(value) for value in drop.queue_before[0].tolist()
                ],
                'queue_after': [
                    int(value) for value in drop.queue_after[0].tolist()
                ],
            }
        if kind == 'remove':
            fruit_id = int(command['fruit_id'])
            return {
                'accepted': self._remove_fruit(fruit_id),
                'fruit_id': fruit_id,
            }
        if kind == 'pause':
            self.paused = True
            return {'accepted': True, 'paused': True}
        if kind == 'resume':
            self.paused = False
            return {'accepted': True, 'paused': False}
        if kind == 'clear':
            queue = command.get('queue') or self.simulator.fruit_queue[0].tolist()
            self.simulator.reset(seeds=self.seed, fruit_queue=queue)
            self.simulator.reset_incremental_progress(stable=True)
            self._pending_events.clear()
            return {'accepted': True}
        if kind == 'load_scene':
            self._replace_from_scene(command['scene'])
            self.paused = bool(command.get('paused', False))
            return {'accepted': True, 'paused': self.paused}
        if kind == 'refresh':
            return {'accepted': True}
        raise ValueError('unsupported live command type')

    def _spawn_manual(self, level, x):
        if not 1 <= int(level) <= 11:
            raise ValueError('fruit level must be in [1, 11]')
        simulator = self.simulator
        if bool(simulator.needs_reset[0].item()):
            raise RuntimeError('finished scene must be cleared before dropping')
        slots = torch.nonzero(~simulator.active[0], as_tuple=False).flatten()
        if not int(slots.numel()):
            raise RuntimeError('max_fruits capacity exhausted')
        slot = int(slots[0].item())
        radius = float(simulator._dropped_radii[level].item())
        display_radius = float(fruit_radius(level))
        drop_x = min(
            self.config.board_width - self.config.wall_width - display_radius - 2.0,
            max(self.config.wall_width + display_radius + 2.0, float(x)),
        )
        fruit_id = int(simulator.next_fruit_id[0].item())
        simulator._clear_event_buffers()
        simulator.positions[0, slot] = torch.tensor(
            (drop_x, float(self.config.spawn_y)),
            dtype=torch.float32,
            device=self.device,
        )
        simulator.velocities[0, slot] = torch.tensor(
            (0.0, 80.0), dtype=torch.float32, device=self.device
        )
        simulator.angles[0, slot] = 0.0
        simulator.angular_velocities[0, slot] = 0.0
        simulator.levels[0, slot] = level
        simulator.physics_radii[0, slot] = radius
        simulator.fruit_ids[0, slot] = fruit_id
        simulator.age_frames[0, slot] = 0
        simulator.active[0, slot] = True
        simulator._set_mass_properties(
            torch.tensor([0], dtype=torch.int64, device=self.device),
            torch.tensor([slot], dtype=torch.int64, device=self.device),
            torch.tensor([level], dtype=torch.int64, device=self.device),
            torch.tensor([radius], dtype=torch.float32, device=self.device),
        )
        simulator._last_queue_before[0].copy_(simulator.fruit_queue[0])
        simulator._last_queue_after[0].copy_(simulator.fruit_queue[0])
        simulator._last_drop_level[0] = level
        simulator._last_drop_x[0] = drop_x
        simulator._last_drop_id[0] = fruit_id
        simulator.next_fruit_id[0] += 1
        simulator.step_count[0] += 1
        simulator.reset_incremental_progress(stable=False)
        simulator._last_batch_result = None
        return fruit_id, drop_x

    def _remove_fruit(self, fruit_id):
        simulator = self.simulator
        matches = simulator.active[0] & (simulator.fruit_ids[0] == fruit_id)
        slots = torch.nonzero(matches, as_tuple=False).flatten()
        if not int(slots.numel()):
            return False
        slot = int(slots[0].item())
        simulator.active[0, slot] = False
        simulator.positions[0, slot] = 0.0
        simulator.velocities[0, slot] = 0.0
        simulator.angles[0, slot] = 0.0
        simulator.angular_velocities[0, slot] = 0.0
        simulator.levels[0, slot] = 0
        simulator.physics_radii[0, slot] = 0.0
        simulator.masses[0, slot] = 0.0
        simulator.inverse_masses[0, slot] = 0.0
        simulator.inverse_inertias[0, slot] = 0.0
        simulator.fruit_ids[0, slot] = 0
        simulator.age_frames[0, slot] = 0
        simulator.reset_incremental_progress(stable=False)
        simulator._clear_event_buffers()
        self._pending_events.clear()
        return True

    def _replace_from_scene(self, scene):
        simulator = self.simulator
        simulator.reset(seeds=self.seed, fruit_queue=scene['queue'])
        fruits = tuple(scene['fruits'])
        count = len(fruits)
        if count:
            slots = slice(0, count)
            positions = torch.tensor(
                [(fruit['x'], fruit['y']) for fruit in fruits],
                dtype=torch.float32,
                device=self.device,
            )
            velocities = torch.tensor(
                [(fruit['vx'], fruit['vy']) for fruit in fruits],
                dtype=torch.float32,
                device=self.device,
            )
            levels = torch.tensor(
                [fruit['level'] for fruit in fruits],
                dtype=torch.int64,
                device=self.device,
            )
            radii = torch.tensor(
                [fruit['physics_radius'] for fruit in fruits],
                dtype=torch.float32,
                device=self.device,
            )
            ids = torch.tensor(
                [fruit['id'] for fruit in fruits],
                dtype=torch.int64,
                device=self.device,
            )
            simulator.positions[0, slots] = positions
            simulator.velocities[0, slots] = velocities
            simulator.angles[0, slots] = torch.tensor(
                [fruit['angle'] for fruit in fruits],
                dtype=torch.float32,
                device=self.device,
            )
            simulator.angular_velocities[0, slots] = torch.tensor(
                [fruit['angular_velocity'] for fruit in fruits],
                dtype=torch.float32,
                device=self.device,
            )
            simulator.levels[0, slots] = levels
            simulator.physics_radii[0, slots] = radii
            simulator.fruit_ids[0, slots] = ids
            simulator.age_frames[0, slots] = torch.tensor(
                [fruit['age_frames'] for fruit in fruits],
                dtype=torch.int64,
                device=self.device,
            )
            simulator.active[0, slots] = True
            masses = simulator._mass_table[levels]
            simulator.masses[0, slots] = masses
            simulator.inverse_masses[0, slots] = masses.reciprocal()
            simulator.inverse_inertias[0, slots] = (
                0.5 * masses * radii.square()
            ).reciprocal()
            simulator.next_fruit_id[0] = int(ids.max().item()) + 1
        simulator.score[0] = int(scene['score'])
        simulator.last_score[0] = int(scene['score'])
        simulator.step_count[0] = int(scene['step_count'])
        quiet = bool(simulator._stable_environments()[0].item())
        simulator.reset_incremental_progress(stable=quiet)
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
        simulator = self.simulator
        # 场景实验室每秒最多发布 120 次状态。逐水果调用 ``item()`` 会让
        # CUDA 流产生数十次同步，即使空场景也会把发布频率压到约 65 FPS。
        # 这里按 dtype 合并成两次 GPU -> CPU 传输，再在 CPU 上组装 JSON；
        # 物理状态本身仍完全由训练同源 Tensor/CUDA 内核维护。
        float_state = torch.cat((
            simulator.positions[0].reshape(-1),
            simulator.velocities[0].reshape(-1),
            simulator.angles[0],
            simulator.angular_velocities[0],
            simulator.physics_radii[0],
        )).detach().cpu()
        int_state = torch.cat((
            simulator.levels[0],
            simulator.fruit_ids[0],
            simulator.age_frames[0],
            simulator.active[0].to(torch.int64),
            simulator.fruit_queue[0],
            simulator.score[0:1],
            simulator.step_count[0:1],
            simulator.physics_frame[0:1],
            simulator.terminated[0:1].to(torch.int64),
            simulator.fail_frames[0:1],
            simulator._incremental_stable_count[0:1],
        )).detach().cpu()

        capacity = self.config.max_fruits
        float_offset = 0
        positions = float_state[float_offset:float_offset + 2 * capacity]
        positions = positions.reshape(capacity, 2)
        float_offset += 2 * capacity
        velocities = float_state[float_offset:float_offset + 2 * capacity]
        velocities = velocities.reshape(capacity, 2)
        float_offset += 2 * capacity
        angles = float_state[float_offset:float_offset + capacity]
        float_offset += capacity
        angular_velocities = float_state[
            float_offset:float_offset + capacity
        ]
        float_offset += capacity
        physics_radii = float_state[float_offset:float_offset + capacity]

        int_offset = 0
        levels = int_state[int_offset:int_offset + capacity]
        int_offset += capacity
        fruit_ids = int_state[int_offset:int_offset + capacity]
        int_offset += capacity
        age_frames = int_state[int_offset:int_offset + capacity]
        int_offset += capacity
        active = int_state[int_offset:int_offset + capacity].to(torch.bool)
        int_offset += capacity
        queue = int_state[
            int_offset:int_offset + self.config.queue_length
        ]
        int_offset += self.config.queue_length
        score = int(int_state[int_offset])
        step_count = int(int_state[int_offset + 1])
        physics_frame = int(int_state[int_offset + 2])
        done = bool(int_state[int_offset + 3])
        fail_frames = int(int_state[int_offset + 4])
        stable_count = int(int_state[int_offset + 5])

        active_slots = torch.nonzero(active, as_tuple=False).flatten()
        if active_slots.numel():
            active_slots = active_slots[
                torch.argsort(fruit_ids[active_slots])
            ]
        fruits = []
        over_danger_line = False
        for slot in active_slots.tolist():
            level = int(levels[slot])
            radius = float(physics_radii[slot])
            x, y = (float(value) for value in positions[slot])
            over_danger_line |= y - radius < float(self.config.spawn_y)
            fruits.append({
                'id': int(fruit_ids[slot]),
                'level': level,
                'x': round(x, 3),
                'y': round(y, 3),
                'vx': round(float(velocities[slot, 0]), 3),
                'vy': round(float(velocities[slot, 1]), 3),
                'angle': round(float(angles[slot]), 6),
                'angular_velocity': round(
                    float(angular_velocities[slot]), 6
                ),
                'age_frames': int(age_frames[slot]),
                'physics_radius': round(radius, 3),
            })
        stable = (
            not fruits
            or (
                stable_count >= self.config.stable_frames
                and not done
            )
        )
        return {
            'format_version': 2,
            'sequence': self._sequence,
            'physics_backend': self.backend,
            'physics_device': str(self.device),
            'training_physics_equivalent': True,
            'physics_fps': self.config.physics_fps,
            'publish_fps': self.publish_fps,
            'paused': self.paused,
            'stable': stable,
            'done': done,
            'danger_progress': round(
                min(1.0, fail_frames / max(1, self.config.danger_frame_limit)),
                6,
            ),
            'over_danger_line': over_danger_line,
            'score': score,
            'step_count': step_count,
            'physics_frame': physics_frame,
            'queue': [int(value) for value in queue],
            'fruits': fruits,
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
