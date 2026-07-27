"""强化学习环境壳层。

本模块属于 `daxigua_rl`，只通过 `daxigua.core.engine.HeadlessGame` 访问游戏。
它不 import pygame，也不 import 手动游戏的 `Board`，以保持 RL 代码和游戏表现层隔离。

当前是 v0 环境接口，目标是先跑通“动作 -> 投放 -> 等待稳定 -> 返回状态”的训练闭环。
模型、GNN 图构建和 replay buffer 后续再独立添加。
"""

from dataclasses import dataclass, field

from daxigua.config import FPS
from daxigua.core.engine import HeadlessGame

from .attribution import (
    ANALYSIS_ACTION_COUNT,
    StateAnalyzer,
    StateAnalyzerConfig,
)
from .reward import RewardConfig, compute_reward
from .training.identity import TransitionKey


@dataclass
class DaxiguaEnvConfig:
    """RL 环境配置。"""

    action_count: int = 15
    physics_fps: int = FPS
    max_physics_frames: int = 720
    stable_frames: int = 15
    space_iterations: int = 32
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    state_analyzer_config: StateAnalyzerConfig = field(
        default_factory=StateAnalyzerConfig
    )


class DaxiguaEnv:
    """类 Gymnasium 的合成大西瓜环境。

    `step(action_index)` 中的一步表示一次完整投放，而不是一帧游戏画面。
    """

    def __init__(self, config=None, game=None, state_analyzer=None):
        self.config = config or DaxiguaEnvConfig()

        # 允许外部注入 HeadlessGame，便于后续做不同场地尺寸或固定队列实验。
        self.game = game or HeadlessGame(
            fps=self.config.physics_fps,
            space_iterations=self.config.space_iterations,
        )
        if state_analyzer is None:
            state_analyzer = StateAnalyzer(
                config=self.config.state_analyzer_config
            )
        if not isinstance(state_analyzer, StateAnalyzer):
            raise TypeError('state_analyzer must be StateAnalyzer')
        self.state_analyzer = state_analyzer

        # StateAnalysis 只在当前 worker 的环境内缓存，不写入主 ReplayBuffer。
        # 采集器会为正式训练显式传入 TransitionKey；直接使用环境时则采用本地键。
        self._cached_state_analysis = None
        self._direct_episode_id = -1
        self._episode_done = True

    @classmethod
    def from_snapshot(cls, snapshot, config=None, state_analyzer=None):
        """从稳定的引擎快照创建可直接执行下一动作的环境。

        ``EngineSnapshot`` 不保存 RL 奖励或状态分析配置；调用方传入的
        ``DaxiguaEnvConfig`` 因此继续控制这些环境侧语义，但其中的物理步频和
        Pymunk 迭代次数必须与快照完全一致。
        """

        if config is not None and not isinstance(config, DaxiguaEnvConfig):
            raise TypeError('config must be DaxiguaEnvConfig or None')

        game = HeadlessGame.from_snapshot(snapshot)
        if config is None:
            config = DaxiguaEnvConfig(
                physics_fps=game.fps,
                space_iterations=game.space.iterations,
            )
        if config.physics_fps != game.fps:
            raise ValueError(
                'config.physics_fps must match EngineSnapshot fps: '
                f'{config.physics_fps!r} != {game.fps!r}'
            )
        if config.space_iterations != game.space.iterations:
            raise ValueError(
                'config.space_iterations must match EngineSnapshot '
                f'space_iterations: {config.space_iterations!r} != '
                f'{game.space.iterations!r}'
            )
        if game.is_done():
            raise ValueError(
                'EngineSnapshot must represent a live, step-ready game'
            )

        env = cls(
            config=config,
            game=game,
            state_analyzer=state_analyzer,
        )
        env._cached_state_analysis = None
        # 快照不携带 rollout 的 worker/episode 身份；显式 TransitionKey 会原样
        # 使用。直接调用 step() 时则从一个新的本地 episode 0 开始。
        env._direct_episode_id = 0
        env._episode_done = False
        return env

    def reset(self, seed=None, fruit_queue=None):
        """重置环境。

        返回：
        - obs: `GameState`
        - info: 辅助调试信息
        """

        obs = self.game.reset(seed=seed, fruit_queue=fruit_queue)
        self._cached_state_analysis = None
        self._direct_episode_id += 1
        self._episode_done = False
        info = {
            'action_candidates': self.action_candidates(),
        }
        return obs, info

    def action_candidates(self):
        """返回当前可选离散投放动作。"""

        return self.game.get_action_candidates(self.config.action_count)

    def step(self, action_index, *, transition_key=None):
        """执行一次投放动作。

        正式 rollout 由 collector 传入动作前 ``TransitionKey``。直接调用环境时会
        使用 worker 0 的本地 episode 编号；终止或截断后必须先 ``reset()``。
        """

        if self._episode_done:
            raise RuntimeError('environment must be reset before step')

        candidates = self.action_candidates()
        if action_index < 0 or action_index >= len(candidates):
            raise IndexError('action_index out of range')

        previous_obs = self.game.get_state()
        transition_key = self._resolve_transition_key(
            previous_obs,
            transition_key,
        )
        analysis_calls = 0
        analysis_seconds = 0.0
        degraded_count = 0
        cache_hit = False

        previous_analysis = self._cached_state_analysis
        if previous_analysis is None:
            previous_analysis = self._analyze_state(
                previous_obs,
                transition_key,
                stable_boundary=True,
            )
            analysis_calls += 1
            analysis_seconds += previous_analysis.diagnostics.analysis_seconds
            degraded_count += int(previous_analysis.diagnostics.degraded)
        elif previous_analysis.transition_key == transition_key:
            cache_hit = True
        else:
            raise RuntimeError(
                'cached StateAnalysis key is out of sync with the current '
                f'transition: {previous_analysis.transition_key!r} != '
                f'{transition_key!r}'
            )

        action = candidates[action_index]
        engine_action_outcome = self.game.execute_action(
            action.drop_x,
            max_frames=self.config.max_physics_frames,
            stable_frames=self.config.stable_frames,
        )
        drop_result = engine_action_outcome.drop_result
        physics_result = engine_action_outcome.physics_result

        obs = engine_action_outcome.final_state
        terminated = physics_result.done
        truncated = physics_result.truncated

        next_key = TransitionKey(
            worker_id=transition_key.worker_id,
            episode_id=transition_key.episode_id,
            step_index=transition_key.step_index + 1,
        )
        post_action_analysis = self._analyze_state(
            obs,
            next_key,
            # 真实终局是吸收边界；即使物理循环因终局提前停止，该快照也不会再推进。
            stable_boundary=bool(physics_result.stable or terminated),
            incoming_transition_key=transition_key,
        )
        analysis_calls += 1
        analysis_seconds += (
            post_action_analysis.diagnostics.analysis_seconds
        )
        degraded_count += int(
            post_action_analysis.diagnostics.degraded
        )
        # Reward V2 的 terminal next potential 仍必须为 0；终局分析只供归因器使用。
        next_analysis = None if terminated else post_action_analysis

        reward, reward_breakdown = compute_reward(
            previous_analysis=previous_analysis,
            next_analysis=next_analysis,
            physics_result=physics_result,
            config=self.config.reward_config,
        )

        self._episode_done = bool(terminated or truncated)
        if self._episode_done:
            # truncated 的 final analysis 只用于本 transition 的 shaping/bootstrap，
            # 不得跨 reset 复用到新 episode。
            self._cached_state_analysis = None
        else:
            self._cached_state_analysis = next_analysis

        info = {
            'action': action,
            'engine_action_outcome': engine_action_outcome,
            'drop_result': drop_result,
            'physics_result': physics_result,
            'reward_breakdown': reward_breakdown,
            'previous_state_analysis': previous_analysis,
            'next_state_analysis': next_analysis,
            'post_action_state_analysis': post_action_analysis,
            'state_analysis_calls': analysis_calls,
            'state_analysis_seconds': float(analysis_seconds),
            'state_analysis_cache_hit': cache_hit,
            'state_analysis_degraded_count': degraded_count,
            'score_delta': physics_result.score_delta,
            'merge_events': physics_result.merge_events,
            'frames_simulated': physics_result.frames_simulated,
            'stable': physics_result.stable,
            'action_candidates': self.action_candidates() if not terminated else (),
        }
        return obs, reward, terminated, truncated, info

    def _resolve_transition_key(self, state, transition_key):
        """选择 collector 显式身份或直接环境调用的本地身份。"""

        if transition_key is None:
            if self._direct_episode_id < 0:
                raise RuntimeError('environment must be reset before step')
            transition_key = TransitionKey(
                worker_id=0,
                episode_id=self._direct_episode_id,
                step_index=state.step_count,
            )
        elif not isinstance(transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey or None')

        if transition_key.step_index != state.step_count:
            raise ValueError(
                'transition_key.step_index must equal the current '
                'GameState.step_count'
            )
        return transition_key

    def _analyze_state(
            self,
            state,
            transition_key,
            *,
            stable_boundary,
            incoming_transition_key=None):
        """使用固定 15 条分析列生成只读状态快照。"""

        analysis_candidates = self.game.get_action_candidates(
            ANALYSIS_ACTION_COUNT
        )
        return self.state_analyzer.analyze(
            state,
            analysis_candidates,
            transition_key,
            stable_boundary=stable_boundary,
            incoming_transition_key=incoming_transition_key,
        )
