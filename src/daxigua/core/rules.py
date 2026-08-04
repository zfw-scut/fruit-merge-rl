"""游戏规则常量和纯规则辅助函数。

这个模块不依赖 pygame，也不依赖 RL。它只保存游戏本体中最稳定的规则：
水果等级、名称、半径、随机生成范围、合成结果、计分和物理半径换算。

手动游戏、无渲染训练环境和后续状态图构建都应该优先复用这里的规则，
避免同一条规则在多个地方重复写、后续改动时发生漂移。
"""


# 顶部待投放水果序列长度。当前设计为 q0 到 q3，共 4 颗。
FRUIT_QUEUE_LENGTH = 4

# 游戏中最小和最大水果等级。
MIN_FRUIT_LEVEL = 1
MAX_FRUIT_LEVEL = 11

# 新投放水果从前 5 级中随机生成。
SPAWN_FRUIT_MIN_LEVEL = 1
SPAWN_FRUIT_MAX_LEVEL = 5

# 原版《合成大西瓜》的水果等级顺序。
FRUIT_NAMES = {
    1: '樱桃',
    2: '草莓',
    3: '葡萄',
    4: '凸顶柑',
    5: '柿子',
    6: '苹果',
    7: '梨',
    8: '桃子',
    9: '菠萝',
    10: '甜瓜',
    11: '西瓜',
}

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

# 同级水果合成时，按来源等级获得三角数分值。
MERGE_SCORES = {
    level: level * (level + 1) // 2
    for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1)
}


def fruit_name(level):
    """返回指定水果等级的标准中文名称。"""

    return FRUIT_NAMES[level]


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
    """返回同级水果合成后生成的等级；西瓜合成后不生成水果。"""

    if level >= MAX_FRUIT_LEVEL:
        return None
    return level + 1


def is_mergeable_level(level):
    """返回指定等级的两颗同级水果是否应触发合成。"""

    return MIN_FRUIT_LEVEL <= level <= MAX_FRUIT_LEVEL


def merge_score(level):
    """返回两颗指定来源等级的水果合成时获得的游戏分数。"""

    return MERGE_SCORES.get(level, 0)


def merge_position(position_a, position_b):
    """返回合成水果的生成位置，即两个来源水果圆心的算术中点。"""

    x1, y1 = position_a
    x2, y2 = position_b
    return (x1 + x2) / 2, (y1 + y2) / 2


def random_spawn_level(rng):
    """使用传入随机数生成器创建一个可投放水果等级。"""

    return rng.randrange(SPAWN_FRUIT_MIN_LEVEL, SPAWN_FRUIT_MAX_LEVEL + 1)
