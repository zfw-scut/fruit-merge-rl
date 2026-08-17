"""用合成步距预测的稳定区间切换标记候选关键节点。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


def merge_difficulty(probabilities):
    """将完整类别分布压成连续难度；数值越大表示越晚或不再合成。"""

    if probabilities.ndim < 1 or probabilities.shape[-1] < 2:
        raise ValueError('probabilities must contain at least two classes')
    class_values = torch.arange(
        probabilities.shape[-1],
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return (probabilities * class_values).sum(dim=-1)


@dataclass(frozen=True, slots=True)
class StableSwitch:
    previous_value: float
    new_value: float
    start_step: int
    settled_start_step: int
    confirmed_step: int

    @property
    def delta(self):
        return self.new_value - self.previous_value

    @property
    def direction(self):
        return 'worsened' if self.delta > 0 else 'improved'


@dataclass(frozen=True, slots=True)
class DetectorUpdate:
    transition_started: bool = False
    transition_cancelled: bool = False
    switch: StableSwitch | None = None


class StableSwitchDetector:
    """识别一个标量预测从稳定平台切换到另一个稳定平台。"""

    def __init__(
            self,
            *,
            window_size=4,
            stable_range=0.4,
            jump_threshold=0.75,
            transition_timeout=32):
        if int(window_size) < 2:
            raise ValueError('window_size must be at least 2')
        if float(stable_range) < 0:
            raise ValueError('stable_range must be non-negative')
        if float(jump_threshold) <= float(stable_range):
            raise ValueError('jump_threshold must exceed stable_range')
        if int(transition_timeout) < int(window_size):
            raise ValueError('transition_timeout must cover one window')
        self.window_size = int(window_size)
        self.stable_range = float(stable_range)
        self.jump_threshold = float(jump_threshold)
        self.transition_timeout = int(transition_timeout)
        self._history = deque(maxlen=self.window_size)
        self._baseline = None
        self._transition_start_step = None
        self._transition_previous = None
        self._transition_age = 0

    @property
    def is_stable(self):
        return self._baseline is not None and not self.in_transition

    @property
    def in_transition(self):
        return self._transition_start_step is not None

    @property
    def stable_value(self):
        return self._baseline

    def _stable_window(self):
        if len(self._history) < self.window_size:
            return None
        values = tuple(value for _, value in self._history)
        if max(values) - min(values) > self.stable_range:
            return None
        return sum(values) / len(values)

    def _clear_transition(self):
        self._transition_start_step = None
        self._transition_previous = None
        self._transition_age = 0

    def update(self, value, step):
        value = float(value)
        step = int(step)
        self._history.append((step, value))
        stable_mean = self._stable_window()

        if self._baseline is None:
            if stable_mean is not None:
                self._baseline = stable_mean
            return DetectorUpdate()

        if not self.in_transition:
            if abs(value - self._baseline) >= self.jump_threshold:
                self._transition_start_step = step
                self._transition_previous = self._baseline
                self._transition_age = 1
                return DetectorUpdate(transition_started=True)
            if stable_mean is not None:
                self._baseline = stable_mean
            return DetectorUpdate()

        self._transition_age += 1
        if stable_mean is not None:
            previous = self._transition_previous
            if abs(stable_mean - previous) >= self.jump_threshold:
                switch = StableSwitch(
                    previous_value=previous,
                    new_value=stable_mean,
                    start_step=self._transition_start_step,
                    settled_start_step=max(
                        self._transition_start_step,
                        int(self._history[0][0]),
                    ),
                    confirmed_step=step,
                )
                self._baseline = stable_mean
                self._clear_transition()
                return DetectorUpdate(switch=switch)
            self._baseline = stable_mean
            self._clear_transition()
            return DetectorUpdate(transition_cancelled=True)

        if self._transition_age >= self.transition_timeout:
            self._baseline = None
            self._clear_transition()
            return DetectorUpdate(transition_cancelled=True)
        return DetectorUpdate()


__all__ = [
    'DetectorUpdate',
    'StableSwitch',
    'StableSwitchDetector',
    'merge_difficulty',
]
