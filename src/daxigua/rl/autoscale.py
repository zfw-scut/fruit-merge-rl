"""只调整执行规模、不改变 DQN 学习语义的资源扩容控制器。"""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True, slots=True)
class ScaleDecision:
    action: str
    target_envs: int
    reason: str


class AdaptiveScaleController:
    def __init__(self, config, *, initial_envs, maximum_envs):
        self.config = config
        self.maximum_envs = int(maximum_envs)
        candidates = sorted(set(
            int(value)
            for value in config.candidate_envs
            if int(value) <= self.maximum_envs
        ))
        if initial_envs not in candidates:
            candidates.append(int(initial_envs))
            candidates.sort()
        self.candidates = tuple(candidates)
        self.current_envs = int(initial_envs)
        self.low_since = None
        self.last_change = float('-inf')
        self.trial = None

    def _next_candidate(self):
        for value in self.candidates:
            if value > self.current_envs:
                return value
        return None

    def observe(self, resources, throughput, *, now=None):
        if not self.config.enabled:
            return None
        now = time.monotonic() if now is None else float(now)
        throughput = float(throughput)
        if self.trial is not None:
            self.trial['samples'].append(throughput)
            baseline = self.trial['baseline']
            previous = self.trial['previous_envs']
            if throughput < baseline * (1.0 - self.config.rollback_loss):
                loss = (baseline - throughput) / max(baseline, 1e-9)
                self.trial = None
                self.current_envs = previous
                self.last_change = now
                return ScaleDecision(
                    'rollback', previous,
                    f'端到端吞吐立即下降 {loss:.1%}',
                )
            if now - self.trial['started'] < self.config.trial_seconds:
                return None
            measured = sum(self.trial['samples']) / max(
                1, len(self.trial['samples'])
            )
            gain = (measured - baseline) / max(baseline, 1e-9)
            self.trial = None
            self.last_change = now
            if gain >= self.config.minimum_throughput_gain:
                return ScaleDecision(
                    'commit', self.current_envs,
                    f'端到端吞吐提高 {gain:.1%}',
                )
            self.current_envs = previous
            return ScaleDecision(
                'rollback', previous,
                f'端到端吞吐增益仅 {gain:.1%}',
            )

        required = (
            'gpu_utilization', 'cpu_utilization',
            'gpu_memory_used_mb', 'gpu_memory_total_mb',
        )
        if any(resources.get(name) is None for name in required):
            self.low_since = None
            return None
        memory_percent = (
            float(resources['gpu_memory_used_mb'])
            / max(float(resources['gpu_memory_total_mb']), 1.0)
            * 100.0
        )
        low = (
            float(resources['gpu_utilization'])
            < self.config.low_gpu_utilization
            and float(resources['cpu_utilization'])
            < self.config.low_cpu_utilization
            and memory_percent < self.config.max_memory_utilization
        )
        if not low:
            self.low_since = None
            return None
        if now - self.last_change < self.config.cooldown_seconds:
            return None
        if self.low_since is None:
            self.low_since = now
            return None
        if now - self.low_since < self.config.observation_seconds:
            return None
        target = self._next_candidate()
        self.low_since = None
        if target is None:
            return None
        previous = self.current_envs
        self.current_envs = target
        self.trial = {
            'previous_envs': previous,
            'baseline': throughput,
            'started': now,
            'samples': [],
        }
        return ScaleDecision(
            'trial', target,
            'GPU 与 CPU 持续低负载，测试更高活跃环境数',
        )
