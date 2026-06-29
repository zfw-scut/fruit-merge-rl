"""水果显示对象和水果资源加载。

这个模块只负责 pygame 表现层需要的水果精灵：
- 根据水果等级加载对应图片。
- 缓存缩放后的图片。
- 维护图片在画布上的 rect。
- 把 pymunk 圆心坐标同步成 pygame 左上角坐标。

水果半径、等级范围、合成分数等规则不在这里定义，而在 `core.rules`。
无渲染训练环境也不依赖这个模块。
"""

import pygame as pg

from ..config import FRUIT_ASSET_DIR
from .rules import fruit_radius


# 图片缓存，key 是 `(图片路径, 缩放尺寸)`。
# pygame 加载图片和 smoothscale 都有成本；游戏运行中会频繁创建水果，
# 所以同一张贴图、同一尺寸只处理一次，后续直接复用 Surface。
_IMAGE_CACHE = {}


def load_fruit_image(path, size):
    """加载并缩放水果贴图。"""

    # 使用字符串路径和尺寸组成缓存键，避免 Path 对象实例差异影响查找。
    key = (str(path), size)

    # 缓存未命中时才真正访问磁盘和执行缩放。
    if key not in _IMAGE_CACHE:
        # `convert_alpha()` 会转换为适合当前显示模式的透明 Surface，
        # 后续 blit 更快，也能保留 PNG 的透明边缘。
        image = pg.image.load(str(path)).convert_alpha()

        # `smoothscale` 比普通 scale 更柔和，适合当前水果贴图的视觉风格。
        _IMAGE_CACHE[key] = pg.transform.smoothscale(image, size)

    return _IMAGE_CACHE[key]


def fruit_image_path(fruit_type):
    """根据水果类型编号得到贴图路径。"""

    # 水果资源统一命名为 `01.png` 到 `11.png`，所以这里用 `:02d` 补齐两位。
    return FRUIT_ASSET_DIR / f'{fruit_type:02d}.png'


def create_fruit(level, x, y):
    """按水果等级创建显示精灵。

    外部仍然通过这个工厂函数创建水果，避免调用方关心图片路径、半径和 rect 细节。
    """

    return Fruit(level, x, y)


class Fruit:
    """单个水果的 pygame 显示精灵。"""

    def __init__(self, level, x, y):
        # type 是旧代码沿用字段，表现层和物理层仍会用它表示水果等级。
        self.type = level

        # 半径统一来自 rules.py，避免显示层和 headless 训练环境各维护一份半径表。
        self.r = fruit_radius(level)
        self.size = (self.r * 2, self.r * 2)

        # 加载等级对应贴图，并创建 pygame rect。
        self.image = load_fruit_image(fruit_image_path(level), self.size)
        self.rect = self.image.get_rect()

        # 当前图片没有实际旋转绘制，但保留角度字段，方便和 pymunk body.angle 同步。
        self.angle_degree = 0

        # 调用方传入的是圆心坐标，这里统一转换成 pygame rect 左上角。
        self.update_position(x, y)

    def update_position(self, x, y, angle_degree=0):
        """把水果显示位置更新到指定圆心。"""

        # pymunk 中圆形刚体的位置是圆心；pygame 贴图矩形使用左上角。
        self.rect.x = x - self.r
        self.rect.y = y - self.r

        # 记录物理角度。当前没有旋转贴图，避免圆形水果贴图在运动时产生额外模糊。
        self.angle_degree = angle_degree

    def draw(self, surface):
        """把水果贴图绘制到目标画布。"""

        surface.blit(self.image, self.rect)
