"""pygame 版本合成大西瓜的完整应用层。

这个文件目前承担“可玩游戏”的大部分外层工作：
- 读取键盘和鼠标输入。
- 管理当前待投放水果、待投放水果队列和冷却时间。
- 驱动 pymunk 物理世界每帧前进。
- 把核心层的水果刚体同步到 pygame 贴图位置。
- 绘制背景、场地、HUD、水果、预览线和各种反馈特效。
- 播放简单的合成、投放、失败音效。

核心规则在 `daxigua.core.board.GameBoard`，而这里更像“游戏壳”和“表现层”。
后续如果继续拆分表现层，再按真实需要新增模块，避免提前创建空包占位。
"""

import math
import random
from array import array
from dataclasses import dataclass

import pygame as pg

from .core.board import GameBoard
from .core.fruit import create_fruit
from .core.rules import FRUIT_QUEUE_LENGTH, fruit_radius as radius_for_level, random_spawn_level


# 顶部独立信息层。分数和待投放队列都绘制在这条区域内，
# 当前悬浮水果会出现在它下方，避免两者视野冲突。
TOP_INFO_LAYER_TOP = 4
TOP_INFO_LAYER_HEIGHT = 68


# 每一级水果对应一组主色，用于粒子、冲击波、飘字等效果。
# 贴图本身来自 assets/fruits，这里的颜色只服务于反馈特效。
FRUIT_COLORS = {
    1: (255, 112, 94),
    2: (255, 183, 77),
    3: (255, 218, 92),
    4: (135, 214, 108),
    5: (91, 196, 191),
    6: (92, 155, 238),
    7: (173, 118, 238),
    8: (239, 120, 187),
    9: (255, 143, 92),
    10: (90, 210, 120),
    11: (255, 96, 96),
}


@dataclass
class Particle:
    """一次性小粒子，用于投放和合成反馈。"""

    # 粒子当前位置。
    x: float
    y: float

    # 粒子速度，单位近似为像素/秒。
    vx: float
    vy: float

    # 粒子半径和颜色。
    radius: float
    color: tuple

    # 当前剩余生命和初始生命，用于计算透明度。
    life: float
    max_life: float

    def update(self, dt):
        """推进粒子一帧。

        返回 True 表示粒子仍然存活；False 表示可以从列表中删除。
        """

        # 生命周期按真实帧耗时减少。
        self.life -= dt

        # 给粒子一个向下的加速度，让它像碎屑一样落下。
        self.vy += 540 * dt

        # 根据速度积分位置。
        self.x += self.vx * dt
        self.y += self.vy * dt

        # 半径轻微缩小，消失过程更柔和。
        self.radius *= 0.992

        return self.life > 0

    def draw(self, surface, offset):
        """把粒子绘制到目标画布。"""

        # 生命越少越透明。
        alpha = max(0, min(255, int(255 * self.life / self.max_life)))

        # 半径至少为 1，避免 pygame 画 0 半径圆。
        radius = max(1, int(self.radius))

        # 单独创建带透明通道的小 surface，才能绘制半透明圆。
        dot = pg.Surface((radius * 2 + 2, radius * 2 + 2), pg.SRCALPHA)
        pg.draw.circle(dot, (*self.color, alpha), (radius + 1, radius + 1), radius)

        # offset 用于屏幕震动：世界坐标不变，只是绘制位置发生偏移。
        surface.blit(dot, (self.x + offset[0] - radius, self.y + offset[1] - radius))


@dataclass
class FloatingText:
    """向上漂浮并逐渐消失的文字，例如合成后的 `+10`。"""

    x: float
    y: float
    text: str
    color: tuple
    life: float
    max_life: float

    # size 不直接创建字体；绘制时由 Board 选择大号或普通弹字字体。
    size: int = 30

    def update(self, dt):
        """推进飘字一帧。"""

        # 生命减少，用于控制透明度和删除时机。
        self.life -= dt

        # y 减小表示文字向屏幕上方移动。
        self.y -= 54 * dt

        return self.life > 0

    def draw(self, surface, font, offset=(0, 0)):
        """绘制飘字。"""

        # 剩余生命映射为透明度。
        alpha = max(0, min(255, int(255 * self.life / self.max_life)))

        # pygame 字体先渲染为 surface，再设置整体透明度。
        text_surface = font.render(self.text, True, self.color)
        text_surface.set_alpha(alpha)

        # 用中心点定位，方便让文字从合成点正上方飘起。
        rect = text_surface.get_rect(center=(self.x + offset[0], self.y + offset[1]))
        surface.blit(text_surface, rect)


@dataclass
class ImpactRing:
    """合成或投放时向外扩散的圆环。"""

    x: float
    y: float
    color: tuple
    life: float
    max_life: float
    start_radius: float
    end_radius: float

    def update(self, dt):
        """推进圆环一帧。"""

        self.life -= dt
        return self.life > 0

    def draw(self, surface, offset):
        """绘制扩散圆环。"""

        # progress 从 0 走到 1，用于在起始半径和结束半径之间插值。
        progress = 1 - max(0, self.life / self.max_life)
        radius = int(self.start_radius + (self.end_radius - self.start_radius) * progress)

        # 圆环越接近结束越透明。
        alpha = max(0, min(190, int(190 * self.life / self.max_life)))

        # 单独 surface 的尺寸要能完整容纳圆环和线宽。
        size = max(8, radius * 2 + 8)
        ring = pg.Surface((size, size), pg.SRCALPHA)

        # 线宽为 3，保持可见但不压过水果贴图。
        pg.draw.circle(ring, (*self.color, alpha), (size // 2, size // 2), radius, 3)

        surface.blit(ring, (self.x + offset[0] - size // 2, self.y + offset[1] - size // 2))


class SoundBank:
    """简单音效库。

    为了不额外引入音频资源文件，这里用正弦波即时合成几个短音效：
    - drop: 投放水果。
    - merge: 水果合成。
    - game_over: 失败。
    """

    def __init__(self):
        # 名称到 pygame Sound 的映射。
        self.sounds = {}

        # 某些环境没有音频设备，例如 CI 或远程桌面；失败时关闭音效但不影响游戏。
        self.enabled = False

        try:
            # 如果 mixer 没初始化，先按单声道 44.1kHz 初始化。
            if not pg.mixer.get_init():
                pg.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

            # 生成三种反馈音。参数依次是频率、时长、音量、衰减曲线。
            self.sounds = {
                'drop': self._tone(220, 0.055, 0.16, 1.7),
                'merge': self._tone(660, 0.09, 0.13, 1.2),
                'game_over': self._tone(120, 0.18, 0.15, 2.4),
            }
            self.enabled = True
        except pg.error:
            # 音频不可用时静默降级，避免因为声音设备问题导致游戏启动失败。
            self.enabled = False

    def _tone(self, frequency, duration, volume, decay):
        """生成一个短促的正弦波音效。"""

        sample_rate = 44100
        sample_count = int(sample_rate * duration)
        samples = array('h')

        for index in range(sample_count):
            # 当前采样点对应的时间。
            t = index / sample_rate

            # 衰减包络让声音逐渐变小，避免“啪”的截断感。
            envelope = (1 - index / sample_count) ** decay

            # 16-bit signed PCM 范围是 -32768 到 32767。
            value = int(32767 * volume * envelope * math.sin(2 * math.pi * frequency * t))
            samples.append(value)

        # pygame 可以直接从 bytes 创建 Sound。
        return pg.mixer.Sound(buffer=samples.tobytes())

    def play(self, name):
        """播放指定名称的音效。"""

        # 音频可用且音效存在时才播放。
        if self.enabled and name in self.sounds:
            self.sounds[name].play()


class Board(GameBoard):
    """当前可玩的 pygame 游戏应用。

    `GameBoard` 管核心规则；这个子类补齐用户实际玩游戏所需的内容：
    输入、固定窗口渲染、特效、音效和主循环。
    """

    def __init__(self):
        # 当前手动游戏使用固定窗口，避免人工体验和后续训练观察受窗口尺寸变化影响。
        self.display_flags = 0

        # 失败判定的持续时间，传给核心层使用。
        self.create_time = 2.0

        # pymunk 重力，y 轴正方向向下，所以这里是向下 1800。
        self.gravity = (0, 1800)

        # 初始化核心棋盘、pygame surface、pymunk space 等基础对象。
        GameBoard.__init__(self, self.create_time, self.gravity)

        # 设置窗口标题。
        pg.display.set_caption('Merge Melon')

        # 提高求解迭代次数，让堆叠和合成更稳定。
        self.space.iterations = 32

        # 阻尼略低于 1，水果运动会慢慢停下来，而不是永远滑动。
        self.space.damping = 0.995

        # 墙体显示宽度。物理边界厚度在 core.board 中创建，这里用于渲染和边界夹取。
        self.wall_width = 20

        # 投放后的冷却时间，单位毫秒。
        self.cooldown_ms = 360

        # 键盘左右移动投放位置的速度，单位近似像素/秒。
        self.keyboard_speed = 360

        # aim_x 是输入目标位置，mouse_x 是实际平滑跟随位置。
        # 分开两者可以让水果移动不那么生硬。
        self.aim_x = self.init_x
        self.mouse_x = self.init_x

        # 当前输入来源：mouse 或 keyboard。用于决定是否继续跟随鼠标。
        self.input_mode = 'mouse'

        # 待投放水果队列。q0 是当前即将投放的水果，q1 到 q3 是后续水果。
        # 这个字段目前只服务手动游戏的视觉预览；后续训练环境也可以复用同样概念。
        self.queue_length = FRUIT_QUEUE_LENGTH
        self.fruit_queue = []

        # 本次程序运行期间的最高分。
        self.best_score = 0

        # 下次允许投放或生成新水果的时间点，单位是 pygame ticks 毫秒。
        self.drop_ready_at = 0

        # 震动和闪屏强度。每帧会衰减。
        self.shake = 0
        self.flash = 0

        # 运行时特效容器。
        self.particles = []
        self.rings = []
        self.floating_texts = []

        # HUD 预览图缓存，避免每帧重新缩放水果图片。
        self.preview_cache = {}

        # 字体对象创建成本不高但也没必要每帧创建，所以初始化时统一准备。
        self.font_title = pg.font.Font(None, 30)
        self.font_score = pg.font.Font(None, 44)
        self.font_label = pg.font.Font(None, 22)
        self.font_popup = pg.font.Font(None, 34)
        self.font_big_popup = pg.font.Font(None, 46)

        # 背景和声音系统。
        self.background = self._build_background()
        self.sound = SoundBank()

        # 创建物理边界、注册合成碰撞回调，并开始第一轮投放。
        self.init_segment()
        self.setup_collision_handler()
        self._start_round()

    def _build_background(self):
        """生成整张窗口背景。"""

        # 先创建 1 像素宽的竖向渐变，再横向拉伸到窗口宽度。
        gradient = pg.Surface((1, self.HEIGHT))

        # 顶部和底部颜色都偏深，保证水果贴图和 HUD 有足够对比度。
        top = (14, 23, 31)
        bottom = (24, 45, 51)

        for y in range(self.HEIGHT):
            # t 是当前 y 在高度中的比例，用于线性插值。
            t = y / max(1, self.HEIGHT - 1)
            color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
            gradient.set_at((0, y), color)

        # convert() 转成显示格式，后续每帧 blit 更快。
        return pg.transform.scale(gradient, self.RES).convert()

    def _pick_fruit_type(self):
        """随机选择可投放水果类型。"""

        # 初始投放只出现 1 到 4 级水果，避免一开始就生成过大的水果。
        return random_spawn_level(random)

    def _fill_fruit_queue(self):
        """把待投放水果队列补足到固定长度。"""

        # 队列长度固定后，HUD 可以稳定绘制 q0 到 q3，也方便后续训练接口取状态。
        while len(self.fruit_queue) < self.queue_length:
            self.fruit_queue.append(self._pick_fruit_type())

    def _advance_fruit_queue(self):
        """成功投放一颗水果后推进队列。"""

        # q0 对应刚刚被投放的水果；投放完成后移除它。
        if self.fruit_queue:
            self.fruit_queue.pop(0)

        # 末尾补一颗新的随机水果，让用户始终能看到固定长度的未来序列。
        self._fill_fruit_queue()

    def _start_round(self, ready_delay=180):
        """准备一轮新的可投放水果。"""

        # 确保队列存在；q0 就是当前要投放的水果。
        self._fill_fruit_queue()
        self.i = self.fruit_queue[0]

        # waiting=True 表示有水果悬在顶部，等待玩家选择 x 坐标并投放。
        self.waiting = True

        # current_fruit 只是预览用显示对象，此时还没有加入物理世界。
        self.current_fruit = create_fruit(self.i, self.mouse_x, self._preview_y())

        # 给一点准备延迟，避免新水果生成的同一瞬间被误触投放。
        self.drop_ready_at = pg.time.get_ticks() + ready_delay

    def _restart_game(self):
        """玩家按 R 时主动重开。"""

        # 重开前刷新最高分。
        self.best_score = max(self.best_score, self.score)

        # 清空核心状态和表现层特效。
        self.reset()
        self.fruit_queue.clear()
        self.particles.clear()
        self.rings.clear()
        self.floating_texts.clear()

        # 开始新一轮投放。
        self._start_round()

    def _clamp_drop_x(self, x, fruit_type=None):
        """把投放 x 坐标限制在左右墙之间。"""

        # 默认使用当前预览水果半径；没有预览水果时给一个保守值。
        fruit_radius = self.current_fruit.r if self.current_fruit else 24

        # 如果调用者明确传入 fruit_type，就直接读取规则表中的半径。
        if fruit_type:
            fruit_radius = radius_for_level(fruit_type)

        # 左右都留出墙体宽度、半径和少量安全间距。
        left = self.wall_width + fruit_radius + 2
        right = self.WIDTH - self.wall_width - fruit_radius - 2
        return max(left, min(right, x))

    def _preview_y(self):
        """当前顶部预览水果的 y 坐标。"""

        # 用当前水果半径决定它悬停在生成线上方多少。
        radius = self.current_fruit.r if self.current_fruit else 28

        # 轻微上下浮动，让待投放状态更有生命感。
        bob = math.sin(pg.time.get_ticks() * 0.005) * 2
        return self.init_y - radius - 12 + bob

    def _can_drop(self):
        """当前是否允许投放水果。"""

        # 必须处于等待状态、有当前类型，并且冷却时间已经结束。
        return self.waiting and self.i is not None and pg.time.get_ticks() >= self.drop_ready_at

    def _drop_current(self):
        """把当前预览水果投放进物理世界。"""

        # 冷却没结束或没有当前水果时，忽略本次投放输入。
        if not self._can_drop():
            return

        # 投放位置取当前平滑后的 mouse_x，并再次夹到合法范围。
        x = int(self._clamp_drop_x(self.mouse_x))

        # 创建显示水果并加入显示列表。
        fruit = create_fruit(self.i, x, self.init_y)
        self.fruits.append(fruit)

        # 创建对应物理刚体。半径取接近显示半径的值，质量随水果尺寸增加。
        ball = self.create_ball(
            self.space, x, self.init_y, m=max(1, fruit.r // 10),
            r=fruit.r - fruit.r % 5, i=self.i)

        # 给一个很小的初速度，让水果刚投放时更自然地下落。
        ball.body.velocity = (0, 80)
        self.balls.append(ball)

        # 投放反馈：粒子、圆环、震动和音效。
        color = FRUIT_COLORS.get(self.i, (255, 255, 255))
        self._burst(x, self.init_y, color, 7, speed=120)
        self.rings.append(ImpactRing(x, self.init_y, color, 0.28, 0.28, fruit.r * 0.4, fruit.r * 1.25))
        self.shake = max(self.shake, 0.08)
        self.sound.play('drop')

        # 预览水果已经变成真实水果，清空等待状态。
        self.current_fruit = None
        self.i = None
        self.waiting = False

        # 队列左移：q1 变成新的 q0，末尾补一颗新的随机水果。
        self._advance_fruit_queue()

        # 设置下一次生成水果的时间点。
        self.drop_ready_at = pg.time.get_ticks() + self.cooldown_ms

    def _spawn_after_cooldown(self):
        """投放冷却结束后生成下一颗待投放水果。"""

        # i 为 None 表示当前没有等待投放的水果。
        if self.i is None and pg.time.get_ticks() >= self.drop_ready_at:
            # 队列当前 q0 成为顶部待投放水果。
            self._fill_fruit_queue()
            self.i = self.fruit_queue[0]

            # 创建新的顶部预览水果。
            self.waiting = True
            self.current_fruit = create_fruit(self.i, self.mouse_x, self._preview_y())

    def _handle_events(self):
        """处理 pygame 事件队列。"""

        for event in pg.event.get():
            if event.type == pg.QUIT:
                raise SystemExit

            # 当前窗口固定；如果某些平台仍发送尺寸变化事件，直接忽略。
            if event.type in (
                    pg.VIDEORESIZE,
                    getattr(pg, 'WINDOWRESIZED', -1),
                    getattr(pg, 'WINDOWSIZECHANGED', -2)):
                continue

            if event.type == pg.MOUSEMOTION:
                # 鼠标移动时切回鼠标模式，投放目标跟随鼠标 x 坐标。
                self.input_mode = 'mouse'
                self.aim_x = self._clamp_drop_x(event.pos[0])
            elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                # 鼠标左键松开时投放水果。
                self.input_mode = 'mouse'
                self.aim_x = self._clamp_drop_x(event.pos[0])
                self._drop_current()
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_SPACE, pg.K_RETURN):
                    # 空格或回车投放。
                    self._drop_current()
                elif event.key == pg.K_r:
                    # R 重新开始。
                    self._restart_game()
                elif event.key == pg.K_ESCAPE:
                    # ESC 退出。
                    raise SystemExit

    def _update_input(self, dt):
        """根据键盘/鼠标更新投放位置。"""

        keys = pg.key.get_pressed()
        direction = 0

        # A/左箭头向左移动。
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            direction -= 1

        # D/右箭头向右移动。
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            direction += 1

        if direction:
            # 键盘输入时，目标位置按速度和 dt 移动。
            self.input_mode = 'keyboard'
            self.aim_x = self._clamp_drop_x(self.aim_x + direction * self.keyboard_speed * dt)
        elif self.input_mode == 'mouse' and pg.mouse.get_focused():
            # 鼠标模式下，如果窗口有焦点，就继续读取当前鼠标位置。
            mouse_x, _ = pg.mouse.get_pos()
            self.aim_x = self._clamp_drop_x(mouse_x)

        # 让实际位置平滑追向目标位置，避免输入变化导致水果瞬间跳动。
        self.mouse_x += (self.aim_x - self.mouse_x) * min(1, dt * 18)

        if self.current_fruit:
            # 每帧都重新夹取，确保预览水果始终留在左右墙内。
            self.mouse_x = self._clamp_drop_x(self.mouse_x)

            # 更新顶部预览水果的位置。
            self.current_fruit.update_position(int(self.mouse_x), int(self._preview_y()))

    def _sync_fruits(self):
        """把 pymunk 物理刚体位置同步到 pygame 水果贴图。"""

        # 合成过程中会删除和新增列表元素，lock=True 时先不同步，避免下标错乱。
        if self.lock:
            return

        for index, ball in enumerate(self.balls):
            if index >= len(self.fruits) or not ball:
                continue

            # pymunk position 是浮点圆心，Fruit.update_position 需要整数圆心。
            x, y = int(ball.body.position[0]), int(ball.body.position[1])
            self.fruits[index].update_position(x, y, ball.body.angle)

    def _update_effects(self, dt):
        """推进所有视觉反馈效果。"""

        # 列表推导同时完成 update 和删除已死亡对象。
        self.particles = [p for p in self.particles if p.update(dt)]
        self.rings = [r for r in self.rings if r.update(dt)]
        self.floating_texts = [t for t in self.floating_texts if t.update(dt)]

        # 震动和闪屏逐帧衰减到 0。
        self.shake = max(0, self.shake - dt * 1.8)
        self.flash = max(0, self.flash - dt * 1.7)

    def _game_over(self):
        """处理失败后的重置和反馈。"""

        # reset 会清空 score，所以先保存当前分数并更新最高分。
        old_score = self.score
        self.best_score = max(self.best_score, old_score)

        # 失败反馈。
        self.sound.play('game_over')
        self.flash = 0.55
        self.shake = 0.45

        # 清空核心游戏状态和特效。
        self.reset()
        self.fruit_queue.clear()
        self.particles.clear()
        self.rings.clear()

        # 留一个 GAME OVER 飘字，作为失败后的视觉反馈。
        self.floating_texts = [
            FloatingText(self.WIDTH / 2, self.HEIGHT * 0.42, 'GAME OVER', (255, 235, 210), 1.2, 1.2, 46)
        ]

        # 稍长延迟后重新给出可投放水果。
        self._start_round(ready_delay=760)

    def on_fruit_merged(self, fruit_type, x, y, score_delta):
        """核心层合成水果后调用的表现层钩子。"""

        # 根据合成后的水果类型选择特效颜色。
        color = FRUIT_COLORS.get(fruit_type, (255, 255, 255))

        # 合成后刷新最高分。
        self.best_score = max(self.best_score, self.score)

        # 合成反馈比普通投放更强。
        self._burst(x, y, color, 16, speed=210)
        self.rings.append(ImpactRing(x, y, color, 0.36, 0.36, 12, 54 + fruit_type * 4))

        # 有分数变化时显示 +N。
        if score_delta:
            self.floating_texts.append(
                FloatingText(x, y - 18, '+' + str(score_delta), (255, 246, 170), 0.72, 0.72, 34)
            )

        # 水果等级越高，合成震动略强，但限制最大值避免眩晕。
        self.shake = max(self.shake, min(0.28, 0.08 + fruit_type * 0.015))
        self.sound.play('merge')

    def _burst(self, x, y, color, count, speed=180):
        """在指定位置生成一组粒子。"""

        for _ in range(count):
            # 随机方向。
            angle = random.uniform(0, math.tau)

            # 随机速度，避免粒子形成机械的圆。
            velocity = random.uniform(speed * 0.35, speed)

            # 随机生命，让粒子消失时间错开。
            life = random.uniform(0.32, 0.68)

            self.particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(angle) * velocity,
                    vy=math.sin(angle) * velocity - random.uniform(30, 130),
                    radius=random.uniform(2.5, 5.5),
                    color=color,
                    life=life,
                    max_life=life,
                )
            )

    def _shake_offset(self):
        """根据当前震动强度生成绘制偏移。"""

        if self.shake <= 0:
            return (0, 0)

        # 震动强度越高，随机偏移范围越大。
        strength = 10 * self.shake
        return (random.uniform(-strength, strength), random.uniform(-strength, strength))

    def _fruit_preview_image(self, fruit_type, max_size):
        """获取 HUD 中的水果预览图。"""

        key = (fruit_type, max_size)

        # 同一类型、同一最大尺寸只缩放一次。
        if key in self.preview_cache:
            return self.preview_cache[key]

        # 临时创建水果对象，是为了复用已有贴图加载逻辑。
        fruit = create_fruit(fruit_type, 0, 0)
        image = fruit.image

        # 等比缩放，保证预览图不超过 max_size。
        scale = min(max_size / image.get_width(), max_size / image.get_height(), 1)
        size = (max(1, int(image.get_width() * scale)), max(1, int(image.get_height() * scale)))

        preview = pg.transform.smoothscale(image, size)
        self.preview_cache[key] = preview
        return preview

    def _draw_background(self):
        """绘制窗口背景。"""

        self.surface.blit(self.background, (0, 0))

    def _draw_header_panel(self):
        """绘制顶部 HUD 背板。"""

        # 顶部区域高度跟随生成线，保证 HUD 和场地分界一致。
        outer_rect = pg.Rect(0, 0, self.WIDTH, self.init_y + 6)
        inner_rect = pg.Rect(18, 8, self.WIDTH - 36, self.init_y - 18)

        # 统一使用和场地下方接近的深色面板，避免顶部视觉突兀。
        pg.draw.rect(self.surface, (16, 25, 35), outer_rect)
        pg.draw.rect(self.surface, (26, 42, 52), inner_rect, border_radius=6)
        pg.draw.rect(self.surface, (42, 66, 73), inner_rect, 1, border_radius=6)

        # 横向细线模拟场地网格，让顶部和底部视觉语言一致。
        for y in range(inner_rect.top + 18, inner_rect.bottom - 4, 22):
            pg.draw.line(self.surface, (36, 58, 66), (inner_rect.left + 10, y), (inner_rect.right - 10, y), 1)

        # 分界线：上方是 HUD，下方是水果物理场地。
        trim_y = self.init_y - 1
        pg.draw.line(self.surface, (78, 58, 43), (0, trim_y - 2), (self.WIDTH, trim_y - 2), 5)
        pg.draw.line(self.surface, (168, 88, 74), (24, trim_y), (self.WIDTH - 24, trim_y), 2)

    def _draw_playfield(self, offset):
        """绘制游戏场地、墙体和警戒线。"""

        ox, oy = offset

        # 场地外框和内框略微内缩，形成层次。
        play_rect = pg.Rect(18 + ox, self.init_y + oy, self.WIDTH - 36, self.HEIGHT - self.init_y - 18)
        inner_rect = play_rect.inflate(-12, -10)
        pg.draw.rect(self.surface, (16, 25, 35), play_rect, border_radius=8)
        pg.draw.rect(self.surface, (26, 42, 52), inner_rect, border_radius=6)

        # 横向网格线，帮助观察水果高度和堆叠。
        for y in range(int(self.init_y + 22 + oy), int(self.HEIGHT - 32 + oy), 34):
            pg.draw.line(self.surface, (32, 54, 64), (28 + ox, y), (self.WIDTH - 28 + ox, y), 1)

        # 绘制左右墙和地板。物理墙体在 core.board 中，这里只是视觉呈现。
        wall_color = (98, 75, 58)
        wall_light = (145, 107, 73)
        left_wall = pg.Rect(0 + ox, self.init_y - 4 + oy, 24, self.HEIGHT - self.init_y + 4)
        right_wall = pg.Rect(self.WIDTH - 24 + ox, self.init_y - 4 + oy, 24, self.HEIGHT - self.init_y + 4)
        floor = pg.Rect(0 + ox, self.HEIGHT - 28 + oy, self.WIDTH, 28)

        for rect in (left_wall, right_wall, floor):
            pg.draw.rect(self.surface, wall_color, rect, border_radius=6)
            pg.draw.rect(self.surface, wall_light, rect, 2, border_radius=6)

        # 红色警戒线对应失败检测的 init_y。
        line_y = self.init_y + oy
        warning = pg.Surface((self.WIDTH, 10), pg.SRCALPHA)
        pg.draw.line(warning, (255, 116, 92, 190), (22, 5), (self.WIDTH - 22, 5), 2)
        self.surface.blit(warning, (0 + ox, line_y - 5))

    def _draw_aim(self, offset):
        """绘制当前投放位置的竖向虚线。"""

        if not self.current_fruit:
            return

        ox, oy = offset
        x = int(self.mouse_x + ox)

        # 冷却结束后使用亮色；冷却中使用灰色，提示还不能投放。
        color = (255, 234, 150) if self._can_drop() else (130, 140, 150)

        # 用短线段组成虚线，从生成线延伸到底部附近。
        for y in range(self.init_y + 12, self.HEIGHT - 42, 18):
            pg.draw.line(self.surface, color, (x, y + oy), (x, y + 8 + oy), 2)

        # 顶部小圆点标出水果会从哪里掉落。
        pg.draw.circle(self.surface, color, (x, int(self.init_y + oy)), 5)

    def _draw_fruit(self, fruit, offset=(0, 0), alpha=255, glow=False):
        """绘制单个水果，并添加影子和可选高光。"""

        ox, oy = offset
        rect = fruit.rect.move(ox, oy)

        # 椭圆阴影让水果更贴地，也能增强堆叠时的空间感。
        shadow = pg.Surface((fruit.r * 2, max(8, fruit.r // 2)), pg.SRCALPHA)
        pg.draw.ellipse(shadow, (0, 0, 0, 78), shadow.get_rect())
        self.surface.blit(shadow, (rect.centerx - fruit.r, rect.centery + fruit.r * 0.42))

        if glow:
            # 当前待投放水果加一层淡淡光晕，和已经落下的水果区分开。
            glow_surface = pg.Surface((fruit.r * 3, fruit.r * 3), pg.SRCALPHA)
            pg.draw.circle(glow_surface, (255, 246, 180, 44), (fruit.r * 3 // 2, fruit.r * 3 // 2), int(fruit.r * 1.25))
            self.surface.blit(glow_surface, (rect.centerx - fruit.r * 1.5, rect.centery - fruit.r * 1.5))

        image = fruit.image

        # 冷却中绘制当前水果时会降低透明度，所以需要复制一份 Surface 再设 alpha。
        if alpha < 255:
            image = image.copy()
            image.set_alpha(alpha)

        self.surface.blit(image, rect)

    def _draw_fruits(self, offset):
        """绘制所有已经进入物理世界的水果。"""

        for fruit in self.fruits:
            self._draw_fruit(fruit, offset)

    def _draw_current_fruit(self, offset):
        """绘制顶部当前待投放水果。"""

        if not self.current_fruit:
            return

        # 冷却未结束时半透明显示。
        alpha = 255 if self._can_drop() else 165
        self._draw_fruit(self.current_fruit, offset, alpha=alpha, glow=True)

    def _draw_effects(self, offset):
        """绘制圆环、粒子和飘字。"""

        # 先画圆环，再画粒子和文字，层次更自然。
        for ring in self.rings:
            ring.draw(self.surface, offset)

        for particle in self.particles:
            particle.draw(self.surface, offset)

        for text in self.floating_texts:
            # GAME OVER 使用大字体，普通加分使用小字体。
            font = self.font_big_popup if text.size >= 40 else self.font_popup
            text.draw(self.surface, font, offset)

    def _draw_fruit_queue(self):
        """绘制顶部待投放水果队列。"""

        if not self.fruit_queue:
            return

        # 队列区域使用整条顶部信息层的右半部分。信息层独立于当前悬浮水果高度，
        # 因此这里不再靠“盖住水果”解决遮挡，而是从布局上错开。
        queue_rect = pg.Rect(8, TOP_INFO_LAYER_TOP, self.WIDTH - 16, TOP_INFO_LAYER_HEIGHT)
        pg.draw.rect(self.surface, (26, 42, 52), queue_rect, border_radius=6)
        pg.draw.rect(self.surface, (42, 66, 73), queue_rect, 1, border_radius=6)

        label = self.font_label.render('QUEUE', True, (167, 202, 194))
        self.surface.blit(label, (self.WIDTH - 178, queue_rect.top + 7))

        # q0 到 q3 横向排列。q0 是当前即将投放的水果，用克制的槽位高亮标出。
        start_x = self.WIDTH - 172
        gap = 44
        center_y = queue_rect.top + 45

        for index, fruit_type in enumerate(self.fruit_queue[:self.queue_length]):
            center = (start_x + index * gap, center_y)

            if index == 0:
                # q0 是当前水果。使用同色系底座和短横线提示队首，
                # 避免亮黄色圆环破坏顶部 HUD 的整体风格。
                pg.draw.circle(self.surface, (31, 58, 62), center, 18)
                pg.draw.circle(self.surface, (88, 139, 132), center, 18, 1)
                pg.draw.line(
                    self.surface,
                    (167, 202, 194),
                    (center[0] - 11, center[1] + 22),
                    (center[0] + 11, center[1] + 22),
                    2,
                )
            else:
                # 后续水果用更低调的边框表示顺序。
                pg.draw.circle(self.surface, (54, 82, 88), center, 16, 1)

            # 为了在 400px 固定宽度内排下 4 个水果，队列缩略图使用较小尺寸。
            image = self._fruit_preview_image(fruit_type, 29 if index == 0 else 27)
            rect = image.get_rect(center=center)
            self.surface.blit(image, rect)

    def _draw_hud(self):
        """绘制分数、最高分和待投放水果队列。"""

        # 先绘制整条顶部信息层，再把左侧分数和右侧队列放进去。
        self._draw_fruit_queue()

        title = self.font_title.render('MERGE MELON', True, (222, 236, 230))
        self.surface.blit(title, (18, TOP_INFO_LAYER_TOP + 6))

        score_text = self.font_score.render(str(self.score), True, (255, 240, 176))
        self.surface.blit(score_text, (18, TOP_INFO_LAYER_TOP + 30))

        best_text = self.font_label.render('BEST ' + str(max(self.best_score, self.score)), True, (167, 202, 194))
        self.surface.blit(best_text, (64, TOP_INFO_LAYER_TOP + 45))

        if self.flash > 0:
            # 失败时短暂红色闪屏。
            overlay = pg.Surface(self.RES, pg.SRCALPHA)
            overlay.fill((255, 86, 70, int(120 * self.flash)))
            self.surface.blit(overlay, (0, 0))

    def _draw_scene(self):
        """按固定顺序绘制完整画面。"""

        # 震动只影响场地、水果和特效；HUD 保持稳定，读数不抖。
        offset = self._shake_offset()

        # 绘制顺序从背景到前景，后画的内容覆盖先画的内容。
        self._draw_background()
        self._draw_header_panel()
        self._draw_playfield(offset)
        self._draw_aim(offset)
        self._draw_fruits(offset)
        self._draw_current_fruit(offset)
        self._draw_effects(offset)
        self._draw_hud()

    def next_frame(self):
        """推进并绘制一帧游戏。

        这是理解游戏运行逻辑的最佳入口：
        限帧 -> 处理输入 -> 更新投放状态 -> 推进物理 -> 同步显示对象 -> 更新特效
        -> 检查失败 -> 绘制 -> 翻转显示缓冲。
        """

        # tick 会等待以维持目标 FPS，并返回上一帧到这一帧经过的毫秒数。
        dt_ms = self.clock.tick(self.FPS)

        # dt 用真实时间驱动输入和特效；同时限制上下界，避免窗口拖动或卡顿导致步长过大。
        dt = min(max(dt_ms / 1000, 1 / self.FPS), 1 / 30)

        # 先处理用户输入和窗口事件。
        self._handle_events()

        # 更新顶部投放位置。
        self._update_input(dt)

        # 如果投放冷却结束，生成下一颗待投放水果。
        self._spawn_after_cooldown()

        # 物理世界使用固定步长推进，这样不同机器上的碰撞稳定性更接近。
        self.space.step(1 / self.FPS)

        # 物理位置变化后，同步到 pygame 水果贴图。
        self._sync_fruits()

        # 粒子、飘字、震动、闪屏等使用真实 dt 更新。
        self._update_effects(dt)

        # 检查是否有水果持续越过警戒线。
        if self.check_fail():
            self._game_over()

        # 绘制完整场景并显示到窗口。
        self._draw_scene()
        pg.display.flip()

    def run(self):
        """运行游戏主循环。"""

        while True:
            self.next_frame()


def main():
    """创建游戏对象并启动主循环。"""

    game = Board()
    game.run()


if __name__ == '__main__':
    # 允许直接运行 `python src/daxigua/app.py` 做调试。
    main()
