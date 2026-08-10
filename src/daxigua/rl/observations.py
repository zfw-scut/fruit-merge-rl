"""模拟器观察到模型输入之间的无 Python 对象张量契约。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from daxigua.simulator import BatchObservation


@dataclass(frozen=True, slots=True)
class TensorState:
    positions: torch.Tensor
    velocities: torch.Tensor
    angular_velocities: torch.Tensor
    levels: torch.Tensor
    physics_radii: torch.Tensor
    age_frames: torch.Tensor
    active: torch.Tensor
    fruit_queue: torch.Tensor
    danger_progress: torch.Tensor
    over_danger_line: torch.Tensor
    physics_fps: float = 30.0

    @classmethod
    def from_observation(
            cls,
            observation: BatchObservation,
            *,
            physics_fps: float,
            rows=None,
            clone=False):
        if not isinstance(observation, BatchObservation):
            raise TypeError('observation must be BatchObservation')
        if physics_fps <= 0:
            raise ValueError('physics_fps must be positive')

        def take(value):
            result = value if rows is None else value[rows]
            return result.clone() if clone else result

        return cls(
            positions=take(observation.positions),
            velocities=take(observation.velocities),
            angular_velocities=take(observation.angular_velocities),
            levels=take(observation.levels),
            physics_radii=take(observation.physics_radii),
            age_frames=take(observation.age_frames),
            active=take(observation.active),
            fruit_queue=take(observation.fruit_queue),
            danger_progress=take(observation.danger_progress),
            over_danger_line=take(observation.over_danger_line),
            physics_fps=float(physics_fps),
        )

    @property
    def batch_size(self):
        return int(self.positions.shape[0])

    @property
    def device(self):
        return self.positions.device

    def index_select(self, indices):
        return type(self)(
            **{
                name: getattr(self, name).index_select(0, indices)
                for name in self.__dataclass_fields__
                if name != 'physics_fps'
            },
            physics_fps=self.physics_fps,
        )

    def clone(self):
        return type(self)(
            **{
                name: getattr(self, name).clone()
                for name in self.__dataclass_fields__
                if name != 'physics_fps'
            },
            physics_fps=self.physics_fps,
        )

    def batch_slice(self, stop):
        """返回前 ``stop`` 个环境的零拷贝批视图。"""

        stop = int(stop)
        if stop < 0 or stop > self.batch_size:
            raise ValueError('batch slice is outside the state batch')
        return type(self)(
            **{
                name: getattr(self, name)[:stop]
                for name in self.__dataclass_fields__
                if name != 'physics_fps'
            },
            physics_fps=self.physics_fps,
        )

    @classmethod
    def cat(cls, states):
        states = tuple(states)
        if not states:
            raise ValueError('states cannot be empty')
        physics_fps = states[0].physics_fps
        if any(state.physics_fps != physics_fps for state in states):
            raise ValueError('all states must use the same physics_fps')
        return cls(
            **{
                name: torch.cat(
                    tuple(getattr(state, name) for state in states), dim=0
                )
                for name in cls.__dataclass_fields__
                if name != 'physics_fps'
            },
            physics_fps=physics_fps,
        )

    def to(self, device, *, non_blocking=False):
        return type(self)(
            **{
                name: getattr(self, name).to(
                    device, non_blocking=non_blocking
                )
                for name in self.__dataclass_fields__
                if name != 'physics_fps'
            },
            physics_fps=self.physics_fps,
        )
