"""常驻设备的定长原始状态 Replay。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .observations import TensorState


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    current: TensorState
    action: torch.Tensor
    reward: torch.Tensor
    next_state: TensorState
    terminal: torch.Tensor
    stage: torch.Tensor


@dataclass(frozen=True, slots=True)
class ReplayWriteTicket:
    indices: torch.Tensor
    source_rows: torch.Tensor
    generations: torch.Tensor | None = None

    @property
    def count(self):
        return int(self.indices.numel())

    @property
    def reference(self):
        if self.generations is None:
            raise RuntimeError('versioned replay references are disabled')
        return ReplayStateReference(
            indices=self.indices,
            generations=self.generations,
        )


@dataclass(frozen=True, slots=True)
class ReplayStateReference:
    """带槽位代次的 Replay 只读引用。"""

    indices: torch.Tensor
    generations: torch.Tensor

    def __post_init__(self):
        if self.indices.shape != self.generations.shape:
            raise ValueError('replay reference tensors must have equal shapes')
        if self.indices.ndim != 1:
            raise ValueError('replay reference tensors must be one-dimensional')
        if self.indices.device != self.generations.device:
            raise ValueError('replay reference tensors must share one device')

    @property
    def count(self):
        return int(self.indices.numel())

    def index_select(self, rows):
        return type(self)(
            indices=self.indices.index_select(0, rows),
            generations=self.generations.index_select(0, rows),
        )


@dataclass(frozen=True, slots=True)
class ReferencedReplayState:
    """Replay 引用读取结果；``valid`` 为纯设备端代次校验。"""

    state: TensorState
    valid: torch.Tensor


class GpuReplayBuffer:
    """先写当前状态、物理步进后再提交下一状态的 GPU 环形缓冲。"""

    _STATE_FIELDS = (
        'positions',
        'velocities',
        'angular_velocities',
        'levels',
        'physics_radii',
        'age_frames',
        'active',
        'fruit_queue',
        'danger_progress',
        'over_danger_line',
    )

    def __init__(
            self,
            capacity,
            *,
            max_fruits=64,
            queue_length=4,
            device='cuda',
            physics_fps=30.0,
            seed=0,
            enable_state_references=False):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity = int(capacity)
        self.max_fruits = int(max_fruits)
        self.queue_length = int(queue_length)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.physics_fps = float(physics_fps)
        self.state_references_enabled = bool(enable_state_references)
        self._cursor = 0
        self._size = 0
        self._next_generation = 1
        self._pending = None
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(seed))

        state_shapes = {
            'positions': ((self.max_fruits, 2), torch.float32),
            'velocities': ((self.max_fruits, 2), torch.float32),
            'angular_velocities': ((self.max_fruits,), torch.float32),
            'levels': ((self.max_fruits,), torch.uint8),
            'physics_radii': ((self.max_fruits,), torch.float32),
            'age_frames': ((self.max_fruits,), torch.int32),
            'active': ((self.max_fruits,), torch.bool),
            'fruit_queue': ((self.queue_length,), torch.uint8),
            'danger_progress': ((), torch.float32),
            'over_danger_line': ((), torch.bool),
        }
        self._current = {}
        self._next = {}
        for name, (shape, dtype) in state_shapes.items():
            full_shape = (self.capacity,) + shape
            self._current[name] = torch.empty(
                full_shape, dtype=dtype, device=self.device
            )
            self._next[name] = torch.empty(
                full_shape, dtype=dtype, device=self.device
            )
        self._actions = torch.empty(
            self.capacity, dtype=torch.int64, device=self.device
        )
        self._rewards = torch.empty(
            self.capacity, dtype=torch.float32, device=self.device
        )
        self._terminals = torch.empty(
            self.capacity, dtype=torch.bool, device=self.device
        )
        self._stages = torch.empty(
            self.capacity, dtype=torch.uint8, device=self.device
        )
        self._slot_generations = (
            torch.zeros(
                self.capacity, dtype=torch.int64, device=self.device
            )
            if self.state_references_enabled
            else None
        )

    def __len__(self):
        return self._size

    @property
    def cursor(self):
        return self._cursor

    @property
    def memory_bytes(self):
        tensors = (
            tuple(self._current.values())
            + tuple(self._next.values())
            + tuple(value for value in (
                self._actions,
                self._rewards,
                self._terminals,
                self._stages,
                self._slot_generations,
            ) if value is not None)
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _validate_state(self, state):
        if not isinstance(state, TensorState):
            raise TypeError('state must be TensorState')
        if state.positions.shape[1:] != (self.max_fruits, 2):
            raise ValueError('state fruit shape does not match replay')
        if state.fruit_queue.shape[1:] != (self.queue_length,):
            raise ValueError('state queue shape does not match replay')
        if state.device != self.device:
            raise ValueError('state and replay must use the same device')

    def _copy_state(self, destination, indices, state, source_rows):
        for name in self._STATE_FIELDS:
            source = getattr(state, name).index_select(0, source_rows)
            destination[name].index_copy_(
                0, indices, source.to(destination[name].dtype)
            )

    @torch.no_grad()
    def begin_append(self, state, mask=None):
        self._validate_state(state)
        if self._pending is not None:
            raise RuntimeError('the previous replay write is still pending')
        if mask is None:
            source_rows = torch.arange(
                state.batch_size, dtype=torch.int64, device=self.device
            )
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            if mask.shape != (state.batch_size,):
                raise ValueError('mask shape does not match state batch')
            source_rows = torch.nonzero(mask, as_tuple=False).flatten()
        count = int(source_rows.numel())
        if count <= 0:
            raise ValueError('at least one transition must be appended')
        if count > self.capacity:
            raise ValueError('one append cannot exceed replay capacity')
        indices = (
            torch.arange(count, device=self.device, dtype=torch.int64)
            + self._cursor
        ).remainder(self.capacity)
        generations = None
        if self.state_references_enabled:
            generations = torch.arange(
                self._next_generation,
                self._next_generation + count,
                device=self.device,
                dtype=torch.int64,
            )
            self._next_generation += count
            self._slot_generations.index_copy_(0, indices, generations)
        ticket = ReplayWriteTicket(
            indices=indices,
            source_rows=source_rows,
            generations=generations,
        )
        self._copy_state(self._current, indices, state, source_rows)
        self._pending = ticket
        return ticket

    @torch.no_grad()
    def finish_append(
            self,
            ticket,
            next_state,
            actions,
            rewards,
            terminals,
            stages=None):
        if ticket is not self._pending:
            raise RuntimeError('replay write ticket is not active')
        self._validate_state(next_state)
        rows = ticket.source_rows
        indices = ticket.indices
        self._copy_state(self._next, indices, next_state, rows)

        def select(values, dtype):
            tensor = torch.as_tensor(values, dtype=dtype, device=self.device)
            if tensor.shape != (next_state.batch_size,):
                raise ValueError('transition vector shape does not match state')
            return tensor.index_select(0, rows)

        self._actions.index_copy_(0, indices, select(actions, torch.int64))
        self._rewards.index_copy_(0, indices, select(rewards, torch.float32))
        self._terminals.index_copy_(0, indices, select(terminals, torch.bool))
        if stages is None:
            stages = torch.zeros(
                next_state.batch_size,
                dtype=torch.uint8,
                device=self.device,
            )
        self._stages.index_copy_(0, indices, select(stages, torch.uint8))
        self._cursor = (self._cursor + ticket.count) % self.capacity
        self._size = min(self.capacity, self._size + ticket.count)
        self._pending = None

    @torch.no_grad()
    def append(
            self,
            current,
            actions,
            rewards,
            next_state,
            terminals,
            mask=None,
            stages=None):
        ticket = self.begin_append(current, mask=mask)
        self.finish_append(
            ticket, next_state, actions, rewards, terminals, stages
        )

    def _state_at(self, storage, indices):
        return TensorState(
            **{
                name: storage[name].index_select(0, indices)
                for name in self._STATE_FIELDS
            },
            physics_fps=self.physics_fps,
        )

    @torch.no_grad()
    def gather_current(self, reference):
        """读取被引用的决策前状态，并在设备端返回槽位是否仍有效。"""

        if not isinstance(reference, ReplayStateReference):
            raise TypeError('reference must be ReplayStateReference')
        if not self.state_references_enabled:
            raise RuntimeError('versioned replay references are disabled')
        if reference.indices.device != self.device:
            raise ValueError('reference and replay must use the same device')
        indices = reference.indices
        valid = self._slot_generations.index_select(0, indices).eq(
            reference.generations
        )
        return ReferencedReplayState(
            state=self._state_at(self._current, indices),
            valid=valid,
        )

    @torch.no_grad()
    def sample(self, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if self._pending is not None:
            raise RuntimeError('cannot sample while a replay write is pending')
        if self._size < batch_size:
            raise RuntimeError('replay does not contain a complete batch')
        indices = torch.randint(
            self._size,
            (batch_size,),
            generator=self._generator,
            device=self.device,
        )
        return ReplayBatch(
            current=self._state_at(self._current, indices),
            action=self._actions.index_select(0, indices),
            reward=self._rewards.index_select(0, indices),
            next_state=self._state_at(self._next, indices),
            terminal=self._terminals.index_select(0, indices),
            stage=self._stages.index_select(0, indices),
        )

    def metadata(self):
        return {
            'capacity': self.capacity,
            'size': self._size,
            'cursor': self._cursor,
            'next_generation': (
                self._next_generation if self.state_references_enabled else None
            ),
            'memory_bytes': self.memory_bytes,
            'physics_fps': self.physics_fps,
            'replay_saved_in_checkpoint': False,
            'stage_labels_saved': True,
            'versioned_state_references': self.state_references_enabled,
        }
