"""游戏规则常量和纯规则辅助函数。

这个模块不依赖 pygame，也不依赖 RL。它只保存游戏本体中最稳定的规则：
水果等级、半径、随机生成范围、合成计分和物理半径换算。

手动游戏、无渲染训练环境和后续状态图构建都应该优先复用这里的规则，
避免同一条规则在多个地方重复写、后续改动时发生漂移。
"""


# 顶部待投放水果序列长度。当前设计为 q0 到 q3，共 4 颗。
FRUIT_QUEUE_LENGTH = 4

# 游戏中最小和最大水果等级。
MIN_FRUIT_LEVEL = 1
MAX_FRUIT_LEVEL = 11

# 新投放水果只从 1 到 4 级中随机生成。
SPAWN_FRUIT_MIN_LEVEL = 1
SPAWN_FRUIT_MAX_LEVEL = 4

# 每一级水果的显示半径，单位是像素。
# 数值来自旧 Fruit 类中的 `2 * 基础半径`，集中到这里后可供 headless 环境复用。
FRUIT_RADII = {
    1: 20,
    2: 30,
    3: 42,
    4: 46,
    5: 58,
    6: 70,
    7: 74,
    8: 100,
    9: 118,
    10: 120,
    11: 156,
}


def fruit_radius(level):
    """返回指定水果等级的显示半径。"""

    return FRUIT_RADII[level]


def fruit_mass(level):
    """返回指定水果等级对应的物理质量。"""

    # 当前手动游戏也是按半径除以 10 得到质量，最小值保护为 1。
    return max(1, fruit_radius(level) // 10)


def dropped_fruit_physics_radius(level):
    """返回新投放水果使用的物理碰撞半径。"""

    radius = fruit_radius(level)

    # 手动游戏投放时会把半径压到 5 的倍数，保持物理手感一致。
    return radius - radius % 5


def merged_fruit_physics_radius(level):
    """返回合成后新水果使用的物理碰撞半径。"""

    # 旧合成逻辑使用显示半径减 1，避免合成瞬间和周围水果过度重叠。
    return fruit_radius(level) - 1


def merge_target_level(level):
    """返回同级水果合成后的等级。"""

    if level >= MAX_FRUIT_LEVEL:
        return None
    return level + 1


def merge_score(level):
    """返回合成到指定等级时获得的分数。"""

    if level < MAX_FRUIT_LEVEL:
        return level
    if level == MAX_FRUIT_LEVEL:
        return 100
    return 0


def random_spawn_level(rng):
    """使用传入随机数生成器创建一个可投放水果等级。"""

    return rng.randrange(SPAWN_FRUIT_MIN_LEVEL, SPAWN_FRUIT_MAX_LEVEL + 1)
