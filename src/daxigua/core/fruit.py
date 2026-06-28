"""水果显示对象和水果资源加载。

这个模块只关心水果本身：
- 每一级水果对应哪个类型编号。
- 每一级水果的半径是多少。
- 每一级水果使用哪张贴图。
- 水果在 pygame 画布上的矩形位置如何随物理刚体同步。

注意：真正的碰撞、重力、合成和计分逻辑不在这里，而在 `core.board`。
这里的对象更接近“显示层用的水果精灵数据”。
"""

import pygame as pg

from ..config import FRUIT_ASSET_DIR


# 图片缓存，key 是 `(图片路径, 缩放尺寸)`。
# pygame 加载图片和 smoothscale 都有成本；游戏运行中会频繁创建水果，
# 所以同一张贴图、同一尺寸只处理一次，后续直接复用 Surface。
_IMAGE_CACHE = {}


def load_fruit_image(path, size):
    """加载并缩放水果贴图。

    参数：
    - path: 图片文件路径。
    - size: 目标显示尺寸，例如 `(40, 40)`。

    返回：
    - 一个已经带透明通道、并缩放到指定尺寸的 pygame Surface。
    """

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
    """根据水果类型编号得到贴图路径。

    水果资源统一命名为 `01.png` 到 `11.png`，所以这里用 `:02d` 补齐两位。
    """

    return FRUIT_ASSET_DIR / f'{fruit_type:02d}.png'


def create_fruit(type, x, y):
    """按类型编号创建具体水果对象。

    当前项目保留了原始实现里的具体水果类命名，例如 `PT`、`YT`。
    后续如果要把水果配置表化，可以先从这个工厂函数入手，因为外部代码
    基本都通过它创建水果，不需要直接知道每个具体类。

    参数：
    - type: 水果等级，1 到 11。数字越大，水果越大。
    - x, y: 水果圆心坐标。具体类会把圆心转换成 pygame rect 的左上角。
    """

    fruit = None

    # 这里保持显式分支，是为了让学习者能直观看到“类型编号 -> 具体类”的关系。
    # 游戏中相同类型碰撞后会合成 `type + 1` 的水果。
    if type == 1:
        fruit = PT(x, y)
    elif type == 2:
        fruit = YT(x, y)
    elif type == 3:
        fruit = JZ(x, y)
    elif type == 4:
        fruit = NM(x, y)
    elif type == 5:
        fruit = MHT(x, y)
    elif type == 6:
        fruit = XHS(x, y)
    elif type == 7:
        fruit = TZ(x, y)
    elif type == 8:
        fruit = BL(x, y)
    elif type == 9:
        fruit = YZ(x, y)
    elif type == 10:
        fruit = XG(x, y)
    elif type == 11:
        fruit = DXG(x, y)

    return fruit


class Fruit:
    """所有水果显示对象的基类。

    每个水果都有：
    - `r`: 半径，单位是像素。
    - `type`: 水果类型编号。
    - `size`: 贴图尺寸，通常是直径乘直径。
    - `image`: 已加载并缩放好的 pygame Surface。
    - `rect`: pygame 用来定位贴图的矩形。
    """

    def __init__(self, x, y):
        # 子类会在进入基类前设置好 `r`、`type`、`size`。
        # 基类调用 `load_images()` 时，会执行子类重写后的版本。
        self.load_images()

        # pygame 的 blit 使用 rect 左上角定位，不直接使用圆心定位。
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        # 当前图片没有实际旋转绘制，但保留角度字段，方便和 pymunk body.angle 同步。
        self.angle_degree = 0

    def load_images(self):
        """由子类实现：根据水果等级加载对应贴图。"""

        pass

    def update_position(self, x, y, angle_degree=0):
        """把水果显示位置更新到指定圆心。

        pymunk 中圆形刚体的位置是圆心；pygame 贴图矩形使用左上角。
        因此这里需要减去半径，把圆心坐标转换成 rect 坐标。
        """

        # 圆心 x/y -> 贴图左上角 x/y。
        self.rect.x = x - self.r
        self.rect.y = y - self.r

        # 记录物理角度。当前没有旋转贴图，避免圆形水果贴图在运动时产生额外模糊。
        self.angle_degree = angle_degree
        # 如果后续想显示旋转，可以在这里基于原图生成旋转后的 image。
        # self.image = pg.transform.rotate(self.image, self.angle_degree)

    def draw(self, surface):
        """把水果贴图绘制到目标画布。"""

        surface.blit(self.image, self.rect)


class PT(Fruit):
    """1 级水果。"""

    def __init__(self, x, y):
        # 原始资源半径按 `2 * 基础值` 放大，保持和旧版本手感一致。
        self.r = 2 * 10
        self.type = 1

        # 贴图是正方形，宽高都是直径。
        self.size = (self.r * 2, self.r * 2)

        # 调用基类时传入左上角，所以从圆心坐标减去半径。
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        # 1 级水果读取 `assets/fruits/01.png`。
        self.image = load_fruit_image(fruit_image_path(1), self.size)


class YT(Fruit):
    """2 级水果。"""

    def __init__(self, x, y):
        self.r = 2 * 15
        self.type = 2
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(2), self.size)


class JZ(Fruit):
    """3 级水果。"""

    def __init__(self, x, y):
        self.r = 2 * 21
        self.type = 3
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(3), self.size)


class NM(Fruit):
    """4 级水果。"""

    def __init__(self, x, y):
        self.r = 2 * 23
        self.type = 4
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(4), self.size)


class MHT(Fruit):
    """5 级水果。"""

    def __init__(self, x, y):
        self.r = 2 * 29
        self.type = 5
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(5), self.size)


class XHS(Fruit):
    """6 级水果。"""

    def __init__(self, x, y):
        self.r = 2 * 35
        self.type = 6
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(6), self.size)


class TZ(Fruit):
    """7 级水果。"""

    def __init__(self, x, y):
        self.r = 2 * 37
        self.type = 7
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(7), self.size)


class BL(Fruit):
    """8 级水果。"""

    def __init__(self, x, y):
        self.r = 2 * 50
        self.type = 8
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(8), self.size)


class YZ(Fruit):
    """9 级水果。"""

    def __init__(self, x, y):
        self.r = 2 * 59
        self.type = 9
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(9), self.size)


class XG(Fruit):
    """10 级水果。"""

    def __init__(self, x, y):
        self.r = 2 * 60
        self.type = 10
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(10), self.size)


class DXG(Fruit):
    """11 级水果，也就是最终的大西瓜。"""

    def __init__(self, x, y):
        self.r = 2 * 78
        self.type = 11
        self.size = (self.r * 2, self.r * 2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = load_fruit_image(fruit_image_path(11), self.size)
