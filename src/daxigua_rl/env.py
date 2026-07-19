"""强化学习环境壳层。

本模块属于 `daxigua_rl`，只通过 `daxigua.core.engine.HeadlessGame` 访问游戏。
它不 import pygame，也不 import 手动游戏的 `Board`，以保持 RL 代码和游戏表现层隔离。

当前是 v0 环境接口，目标是先跑通“动作 -> 投放 -> 等待稳定 -> 返回状态”的训练闭环。
模型、GNN 图构建和 replay buffer 后续再独立添加。
"""

from dataclasses import dataclass, field

from daxigua.config import FPS
from daxigua.core.engine import HeadlessGame

from .reward import RewardConfig, compute_reward


@dataclass
class DaxiguaEnvConfig:
    """RL 环境配置。"""

    action_count: int = 15
    physics_fps: int = FPS
    max_physics_frames: int = 720
    stable_frames: int = 15
    space_iterations: int = 32
    reward_config: RewardConfig = field(default_factory=RewardConfig)


class DaxiguaEnv:
    """类 Gymnasium 的合成大西瓜环境。

    `step(action_index)` 中的一步表示一次完整投放，而不是一帧游戏画面。
    """

    def __init__(self, config=None, game=None):
        self.config = config or DaxiguaEnvConfig()

        # 允许外部注入 HeadlessGame，便于后续做不同场地尺寸或固定队列实验。
        self.game = game or HeadlessGame(
            fps=self.config.physics_fps,
            space_iterations=self.config.space_iterations,
        )

    def reset(self, seed=None, fruit_queue=None):
        """重置环境。

        返回：
        - obs: `GameState`
        - info: 辅助调试信息
        """

        obs = self.game.reset(seed=seed, fruit_queue=fruit_queue)
        info = {
            'action_candidates': self.action_candidates(),
        }
        return obs, info

    def action_candidates(self):
        """返回当前可选离散投放动作。"""

        return self.game.get_action_candidates(self.config.action_count)

    def step(self, action_index):
        """执行一次投放动作。"""

        candidates = self.action_candidates()
        if action_index < 0 or action_index >= len(candidates):
            raise IndexError('action_index out of range')

        previous_obs = self.game.get_state()
        action = candidates[action_index]
        drop_result = self.game.drop_at(action.drop_x)
        physics_result = self.game.advance_physics(
            max_frames=self.config.max_physics_frames,
            until_stable=True,
            stable_frames=self.config.stable_frames,
        )

        obs = self.game.get_state()
        terminated = physics_result.done
        truncated = physics_result.truncated

        reward, reward_breakdown = compute_reward(
            previous_state=previous_obs,
            next_state=obs,
            physics_result=physics_result,
            config=self.config.reward_config,
        )

        info = {
            'action': action,
            'drop_result': drop_result,
            'reward_breakdown': reward_breakdown,
            'score_delta': physics_result.score_delta,
            'merge_events': physics_result.merge_events,
            'frames_simulated': physics_result.frames_simulated,
            'stable': physics_result.stable,
            'action_candidates': self.action_candidates() if not terminated else (),
        }
        return obs, reward, terminated, truncated, info
