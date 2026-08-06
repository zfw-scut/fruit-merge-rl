"""场景实验室中基于当前 checkpoint 的持续 greedy 决策控制器。"""

from __future__ import annotations

from copy import deepcopy
import math
from threading import Event, Lock, Thread


def _scene_from_snapshot(snapshot):
    return {
        'name': '模型持续决策实时状态',
        'fps': int(snapshot['physics_fps']),
        'queue': list(snapshot['queue']),
        'probe_action': 10,
        'score': int(snapshot['score']),
        'step_count': int(snapshot['step_count']),
        'fruits': [dict(fruit) for fruit in snapshot['fruits']],
    }


class ScenarioModelController:
    """稳定边界到达后推理并执行一次真实离散投放。"""

    def __init__(
            self,
            live_session,
            model_evaluator,
            *,
            stable_window_seconds=0.125,
            max_decisions=1000):
        stable_window_seconds = float(stable_window_seconds)
        if (
                not math.isfinite(stable_window_seconds)
                or stable_window_seconds <= 0.0):
            raise ValueError('stable_window_seconds must be positive')
        if int(max_decisions) <= 0:
            raise ValueError('max_decisions must be positive')
        self.live_session = live_session
        self.model_evaluator = model_evaluator
        self.stable_window_seconds = stable_window_seconds
        self.max_decisions = int(max_decisions)
        self._shutdown = Event()
        self._enabled = Event()
        self._lock = Lock()
        self._thread = None
        self._status = {
            'available': True,
            'running': False,
            'phase': 'idle',
            'decision_count': 0,
            'max_decisions': self.max_decisions,
            'stable_samples': 0,
            'required_stable_samples': self._required_stable_samples(),
            'last_evaluation': None,
            'last_drop': None,
            'message': '等待启动模型持续决策。',
            'error': None,
        }

    def _required_stable_samples(self):
        return max(
            1,
            int(math.ceil(
                self.stable_window_seconds * self.live_session.publish_fps
                - 1e-12
            )),
        )

    def start_service(self):
        if self._thread is None:
            self._shutdown.clear()
            self._thread = Thread(
                target=self._run,
                name='scenario-lab-model-controller',
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self):
        self._enabled.clear()
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def status(self):
        with self._lock:
            return deepcopy(self._status)

    @property
    def running(self):
        return self._enabled.is_set()

    def start(self):
        if self._thread is None:
            raise RuntimeError('model controller service is not running')
        snapshot = self.live_session.snapshot()
        if snapshot['done']:
            raise RuntimeError('当前对局已结束，请先清空或载入新场景')
        if self._enabled.is_set():
            return self.status()
        with self._lock:
            self._status.update({
                'running': True,
                'phase': 'waiting_stable',
                'decision_count': 0,
                'stable_samples': 0,
                'last_evaluation': None,
                'last_drop': None,
                'message': (
                    '模型持续决策已启动，等待当前局面连续稳定。'
                    if not snapshot['paused'] else
                    '模型持续决策已启动，等待物理恢复。'
                ),
                'error': None,
            })
        self._enabled.set()
        self._refresh_live_state()
        return self.status()

    def stop(self, *, reason='user'):
        was_running = self._enabled.is_set()
        self._enabled.clear()
        message = {
            'user': '模型持续决策已停止，当前局面保持不变。',
            'manual_control': '检测到手工场景操作，模型持续决策已停止。',
            'done': '对局已经结束，模型持续决策自动停止。',
            'limit': '已达到持续决策次数上限，模型自动停止。',
        }.get(reason, '模型持续决策已停止。')
        with self._lock:
            self._status.update({
                'running': False,
                'phase': reason,
                'stable_samples': 0,
                'message': message,
            })
        if was_running:
            self._refresh_live_state()
        return self.status()

    def _refresh_live_state(self):
        try:
            self.live_session.execute({'type': 'refresh'}, timeout=1.0)
        except (RuntimeError, TimeoutError):
            pass

    def _set_waiting(self, phase, message, stable_samples=0):
        with self._lock:
            self._status.update({
                'running': True,
                'phase': phase,
                'stable_samples': int(stable_samples),
                'message': message,
            })

    def _fail(self, error):
        self._enabled.clear()
        with self._lock:
            self._status.update({
                'running': False,
                'phase': 'error',
                'stable_samples': 0,
                'message': '模型持续决策因错误停止。',
                'error': f'{type(error).__name__}: {error}',
            })
        self._refresh_live_state()

    def _run(self):
        sequence = -1
        topology = None
        stable_samples = 0
        last_decision_step = None
        while not self._shutdown.is_set():
            snapshot = self.live_session.wait_for_snapshot(
                sequence, timeout=0.5
            )
            next_sequence = int(snapshot['sequence'])
            if next_sequence == sequence:
                continue
            sequence = next_sequence
            if not self._enabled.is_set():
                topology = None
                stable_samples = 0
                last_decision_step = None
                continue
            if snapshot['done']:
                self.stop(reason='done')
                continue
            if snapshot['paused']:
                topology = None
                stable_samples = 0
                self._set_waiting(
                    'paused', '物理已暂停，恢复后继续模型决策。'
                )
                continue

            next_topology = tuple(sorted(
                (int(fruit['id']), int(fruit['level']))
                for fruit in snapshot['fruits']
            ))
            if next_topology != topology:
                topology = next_topology
                stable_samples = 0
            if not snapshot['stable']:
                stable_samples = 0
                self._set_waiting(
                    'waiting_stable', '水果仍在运动，等待连续稳定。'
                )
                continue
            stable_samples += 1
            required = self._required_stable_samples()
            if stable_samples < required:
                self._set_waiting(
                    'waiting_stable',
                    f'连续稳定确认 {stable_samples}/{required}。',
                    stable_samples,
                )
                continue
            step_count = int(snapshot['step_count'])
            if step_count == last_decision_step:
                continue

            try:
                self._set_waiting('inference', '正在执行模型推理。', required)
                evaluation = self.model_evaluator.evaluate(
                    _scene_from_snapshot(snapshot),
                    danger_progress=snapshot.get('danger_progress', 0.0),
                    over_danger_line=snapshot.get('over_danger_line'),
                )
                if not self._enabled.is_set():
                    continue
                latest = self.live_session.snapshot()
                latest_topology = tuple(sorted(
                    (int(fruit['id']), int(fruit['level']))
                    for fruit in latest['fruits']
                ))
                if (
                        latest['paused']
                        or latest['done']
                        or int(latest['step_count']) != step_count
                        or latest_topology != topology):
                    stable_samples = 0
                    continue
                drop = self.live_session.execute({
                    'type': 'drop_action',
                    'action': int(evaluation['action']),
                })
                last_decision_step = step_count
                stable_samples = 0
                with self._lock:
                    decision_count = self._status['decision_count'] + 1
                    self._status.update({
                        'running': True,
                        'phase': 'physics',
                        'decision_count': decision_count,
                        'stable_samples': 0,
                        'last_evaluation': evaluation,
                        'last_drop': drop,
                        'message': (
                            f"已执行 A{evaluation['action']}，等待物理稳定。"
                        ),
                        'error': None,
                    })
                if decision_count >= self.max_decisions:
                    self.stop(reason='limit')
            except Exception as error:  # 控制器失败不得拖垮实时物理线程。
                self._fail(error)


__all__ = ['ScenarioModelController']
