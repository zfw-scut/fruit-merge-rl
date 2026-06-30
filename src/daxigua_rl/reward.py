"""RL reward shaping 逻辑。

游戏本体只负责规则和分数；强化学习奖励属于训练接口的一部分，因此放在
`daxigua_rl` 中实现。这样后续可以反复调整奖励，而不污染手动游戏逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """奖励函数配置。

    第一版 reward 由几部分组成：

    - 合成得分奖励：鼓励真实得分。
    - 存活奖励：轻微鼓励成功完成一次投放。
    - 高度变化奖励/惩罚：鼓励让堆叠变低，惩罚让堆叠变高。
    - 危险高度惩罚：局面越接近顶部死亡线，持续惩罚越大。
    - 终局惩罚：游戏失败时给较大负奖励。
    """

    score_scale: float = 1.0
    survival_bonus: float = 0.05
    height_delta_weight: float = 0.02
    danger_height_weight: float = 1.0
    terminal_penalty: float = -100.0


@dataclass(frozen=True)
class RewardBreakdown:
    """一次 step 的 reward 明细。

    这些字段会放入 `info["reward_breakdown"]`，方便训练日志观察模型到底在被
    哪些奖励项驱动。
    """

    total: float
    score_reward: float
    survival_bonus: float
    height_delta_reward: float
    danger_penalty: float
    terminal_penalty: float
    previous_height_ratio: float
    next_height_ratio: float
    height_delta_ratio: float

    def to_dict(self):
        """转换成普通 dict，方便日志、JSON 或终端打印。"""

        return {
            'total': self.total,
            'score_reward': self.score_reward,
            'survival_bonus': self.survival_bonus,
            'height_delta_reward': self.height_delta_reward,
            'danger_penalty': self.danger_penalty,
            'terminal_penalty': self.terminal_penalty,
            'previous_height_ratio': self.previous_height_ratio,
            'next_height_ratio': self.next_height_ratio,
            'height_delta_ratio': self.height_delta_ratio,
        }


def compute_reward(previous_state, next_state, physics_result, config=None):
    """根据前后状态和物理结果计算 reward。

    参数：
    - `previous_state`: 执行动作前的 `GameState`。
    - `next_state`: 物理稳定后或终止后的 `GameState`。
    - `physics_result`: `HeadlessGame.advance_physics(...)` 的结果。
    - `config`: `RewardConfig`。

    返回：
    - `(reward, RewardBreakdown)`
    """

    config = config or RewardConfig()

    previous_height_ratio = _height_ratio(previous_state)
    next_height_ratio = _height_ratio(next_state)
    height_delta_ratio = next_height_ratio - previous_height_ratio

    # 真实游戏分数仍然是主奖励来源。
    score_reward = float(physics_result.score_delta) * config.score_scale

    # 只要这一步没有结束游戏，就给一点点存活奖励。
    # 数值必须较小，避免模型为了苟活而忽视合成得分。
    survival_bonus = 0.0 if physics_result.done else float(config.survival_bonus)

    # 高度变高时 height_delta_ratio 为正，给负奖励；
    # 高度变低时为负，转成正奖励，鼓励通过合成/滚动降低堆叠。
    height_delta_reward = -float(config.height_delta_weight) * height_delta_ratio

    # 局面整体越高，越接近顶部危险线，持续负奖励越大。
    danger_penalty = -float(config.danger_height_weight) * next_height_ratio

    # 游戏失败时给终局惩罚。truncated 暂不额外惩罚，后续如果发现物理不稳定被利用再加。
    terminal_penalty = float(config.terminal_penalty) if physics_result.done else 0.0

    total = (
        score_reward
        + survival_bonus
        + height_delta_reward
        + danger_penalty
        + terminal_penalty
    )

    breakdown = RewardBreakdown(
        total=float(total),
        score_reward=float(score_reward),
        survival_bonus=float(survival_bonus),
        height_delta_reward=float(height_delta_reward),
        danger_penalty=float(danger_penalty),
        terminal_penalty=float(terminal_penalty),
        previous_height_ratio=float(previous_height_ratio),
        next_height_ratio=float(next_height_ratio),
        height_delta_ratio=float(height_delta_ratio),
    )
    return float(total), breakdown


def _height_ratio(state):
    """把当前最大堆叠高度归一化到 `[0, 1]`。

    `GameState.max_height` 表示从地板往上算的最高堆叠高度。
    可玩高度约等于 `height - spawn_y`；越接近 1，说明越接近顶部危险线。
    """

    playable_height = max(1.0, float(state.geometry.height - state.geometry.spawn_y))
    ratio = float(state.max_height) / playable_height
    return max(0.0, min(1.0, ratio))
