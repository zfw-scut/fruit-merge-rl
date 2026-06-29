"""游戏核心棋盘和物理世界。

`GameBoard` 是游戏本体的核心基类，负责：
- 创建 pygame 窗口和 pymunk 物理世界。
- 创建场地边界和水果刚体。
- 处理相同水果碰撞后的合成。
- 维护分数、失败检测、存活状态等核心游戏状态。

当前 `daxigua.app.Board` 继承它，并在上层补充输入、渲染、音效和特效。
这样拆分后，未来 RL 接口可以优先围绕这里的状态和动作建立，而不是直接依赖 UI。
"""

import pygame as pg
import pymunk.pygame_util

from .fruit import create_fruit
from ..config import DEFAULT_WINDOW_SIZE, FPS, SPAWN_LINE_Y


class GameBoard(object):
    """合成大西瓜的核心游戏板。

    这个类不直接实现完整游戏循环；它提供物理世界、合成规则和失败判断。
    子类可以决定如何读取输入、如何渲染、如何驱动每一帧。
    """

    def __init__(self, create_time, gravity):
        # 当前窗口尺寸和画布尺寸。`RES` 是 pygame 常用的 `(width, height)`。
        self.RES = self.WIDTH, self.HEIGHT = DEFAULT_WINDOW_SIZE

        # 目标帧率由配置文件统一管理，物理步进也会使用这个值。
        self.FPS = FPS

        # `balls` 存 pymunk 的 Circle 形状；`fruits` 存对应的 pygame 显示对象。
        # 两个列表用相同下标保持同步：`balls[i]` 对应 `fruits[i]`。
        self.balls = []
        self.fruits = []

        # 初始化分数、等待状态、当前水果等运行状态。
        # 注意：此时 `self.space` 还没创建，但 `self.balls` 是空列表，所以 reset 不会移除物体。
        self.reset()

        # 水果生成线的 y 坐标。当前手动游戏窗口固定，顶部预留独立信息层和悬浮水果层。
        self.init_y = SPAWN_LINE_Y

        # 默认投放 x 坐标在窗口中间。
        self.init_x = int(self.WIDTH / 2)

        # 初始化 pygame 基础模块。表现层会继续设置标题、字体、音效等。
        pg.init()

        # 子类可以在调用基类前设置 display_flags。当前手动游戏使用固定窗口，所以通常为 0。
        self.display_flags = getattr(self, 'display_flags', 0)

        # 创建主窗口 Surface。后续所有绘制都会 blit 到这个 surface 上。
        self.surface = pg.display.set_mode(self.RES, self.display_flags)

        # pygame Clock 用于限制帧率，并计算每帧耗时。
        self.clock = pg.time.Clock()

        # pymunk 的调试绘制选项。当前正式渲染不直接使用，但保留给调试物理世界。
        self.draw_options = pymunk.pygame_util.DrawOptions(self.surface)

        # 创建 pymunk 物理空间，并设置重力。
        self.space = pymunk.Space()
        self.space.gravity = gravity

        # 场地边界线段会存在这里，窗口缩放时需要先移除旧边界再创建新边界。
        self.segments = []

        # 失败判定使用的持续时间：水果超过生成线超过这个时间才算失败。
        self.create_time = create_time

    def reset(self):
        """重置一局游戏的核心状态。"""

        # 如果已经有物理水果，先从 pymunk 空间里移除形状和刚体。
        # reset 既会在初始化早期调用，也会在游戏结束后调用，因此需要兼容空列表。
        for ball in self.balls:
            self.space.remove(ball, ball.body)

        # 删除再重建列表，是原始实现中用来确保旧引用断开的写法。
        # 后续如果有外部接口持有列表引用，这里可能需要改成 `.clear()`。
        del self.fruits
        del self.balls
        self.fruits = []
        self.balls = []

        # 当前局分数。
        self.score = 0

        # 上一次计分前的分数。保留字段便于 UI 或 RL 计算即时分数变化。
        self.last_score = 0

        # 历史字段，当前游戏循环中没有核心用途，保留以减少行为变更。
        self.count = 1

        # 合成锁。碰撞回调中会短暂置 True，避免一次合成过程被重复进入。
        self.lock = False

        # 是否正在等待玩家投放当前水果。
        self.waiting = False

        # 当前悬在顶部、还没被投放的水果显示对象。
        self.current_fruit = None

        # 当前待投放水果类型编号。
        self.i = None

        # 水果越线的累计帧数，用于“持续越线才失败”的缓冲判定。
        self.fail_count = 0

        # 当前局是否仍然存活。
        self.alive = True

    def init_segment(self):
        """创建或重建物理场地边界。"""

        # 窗口尺寸变化时旧边界已经不匹配，需要先从物理空间移除。
        for segment in self.segments:
            try:
                self.space.remove(segment)
            except Exception:
                # 某些边界可能已经被移除；这里容错，避免 resize 中断游戏。
                pass
        self.segments = []

        # 四个角点：左上、左下、右下、右上。
        B1, B2, B3, B4 = (0, 0), (0, self.HEIGHT), (self.WIDTH, self.HEIGHT), (self.WIDTH, 0)

        # 只创建左墙、底板、右墙；顶部保持开放，让水果从生成线落下。
        borders = (B1, B2), (B2, B3), (B3, B4)
        for border in borders:
            self.segments.append(
                self.create_segment(*border, 20, self.space, 'darkslategray'))

    def resize_world(self, width, height, recreate_display=False):
        """按新窗口尺寸调整游戏世界。

        参数：
        - width, height: 新的窗口尺寸。
        - recreate_display: 是否重新创建 pygame display。当前手动游戏不走窗口拖拽缩放，
          这个参数主要留给内部调试或未来实验场景。

        返回：
        - True 表示尺寸确实发生变化。
        - False 表示新旧尺寸相同，无需后续刷新。
        """

        # 子类可以定义最小尺寸；没有定义时使用较保守的默认值。
        min_width = getattr(self, 'min_width', 320)
        min_height = getattr(self, 'min_height', 520)

        # 限制最小窗口，避免边界反转或投放区域过窄。
        width = max(min_width, int(width))
        height = max(min_height, int(height))

        # 尺寸没有变化时直接退出，减少 resize 抖动时的重复工作。
        if width == self.WIDTH and height == self.HEIGHT:
            return False

        # 更新尺寸字段，后续渲染、边界、生成线都依赖这些值。
        self.RES = self.WIDTH, self.HEIGHT = width, height
        self.init_y = SPAWN_LINE_Y
        self.init_x = int(self.WIDTH / 2)

        if recreate_display:
            # 某些运行环境需要显式 set_mode 才能拿到新 Surface。
            self.surface = pg.display.set_mode(self.RES, self.display_flags)
        else:
            # 正常拖拽 resize 时优先复用 pygame 当前窗口 Surface，减少闪烁和重开窗口。
            self.surface = pg.display.get_surface() or self.surface

        # Surface 变化后，pymunk 调试绘制选项也要指向新的 surface。
        self.draw_options = pymunk.pygame_util.DrawOptions(self.surface)

        # 重新创建物理边界，使水果能在新的场地大小中滚动和碰撞。
        self.init_segment()
        return True

    def setup_collision_handler(self):
        """注册同级水果碰撞后的合成回调。"""

        def post_solve_bird_line(arbiter, space, data):
            """相同 collision_type 的两个水果碰撞后合成更高级水果。

            这个函数名保留了旧代码的命名习惯。实际含义是：
            当两个相同类型的圆形水果碰撞并完成物理解算后，删除原来的两个水果，
            在碰撞位置创建 `类型 + 1` 的新水果，并增加分数。
            """

            # 防止同一次碰撞合成在回调链中重复进入。
            if not self.lock:
                self.lock = True

                b1, b2 = None, None

                # 两个形状的 collision_type 相同，所以取第一个 + 1 就是合成后的类型。
                i = arbiter.shapes[0].collision_type + 1

                # 读取两个碰撞水果的圆心位置。
                x1, y1 = arbiter.shapes[0].body.position
                x2, y2 = arbiter.shapes[1].body.position

                # 新水果尽量生成在较低的那个水果位置，减少合成后突然向上弹的违和感。
                if y1 > y2:
                    x, y = x1, y1
                else:
                    x, y = x2, y2

                # 如果第一个碰撞形状仍在当前球列表中，移除它的物理形状和显示水果。
                if arbiter.shapes[0] in self.balls:
                    b1 = self.balls.index(arbiter.shapes[0])
                    space.remove(arbiter.shapes[0], arbiter.shapes[0].body)
                    self.balls.remove(arbiter.shapes[0])
                    fruit1 = self.fruits[b1]
                    self.fruits.remove(fruit1)

                # 同样移除第二个碰撞形状。
                if arbiter.shapes[1] in self.balls:
                    b2 = self.balls.index(arbiter.shapes[1])
                    space.remove(arbiter.shapes[1], arbiter.shapes[1].body)
                    self.balls.remove(arbiter.shapes[1])
                    fruit2 = self.fruits[b2]
                    self.fruits.remove(fruit2)

                # 创建合成后的水果显示对象。这里传入的 y 是生成线附近，
                # 但随后 `_sync_fruits()` 会用新刚体的位置重新同步显示坐标。
                fruit = create_fruit(i, x, self.init_y)
                self.fruits.append(fruit)

                # 创建合成后的物理圆形刚体，并放在刚才选定的碰撞位置。
                ball = self.create_ball(
                    self.space, x, y, m=fruit.r // 10, r=fruit.r - 1, i=i)
                self.balls.append(ball)

                # 计分规则：1 到 10 级合成给等级分，合成 11 级大西瓜给 100 分。
                score_delta = 0
                if i < 11:
                    self.last_score = self.score
                    score_delta = i
                    self.score += score_delta
                elif i == 11:
                    self.last_score = self.score
                    score_delta = 100
                    self.score += score_delta

                # 表现层可以定义 `on_fruit_merged()`，用来播放粒子、飘字、音效等。
                # 核心层只检查有没有这个钩子，不直接 import 表现层。
                if hasattr(self, 'on_fruit_merged'):
                    self.on_fruit_merged(i, x, y, score_delta)

                # 合成流程结束，释放锁。
                self.lock = False

        # 只注册 1 到 10 级的“同级碰撞合成”。
        # 11 级是终点，不再合成 12 级。
        for i in range(1, 11):
            if hasattr(self.space, 'add_collision_handler'):
                # pymunk 旧版 API。
                self.space.add_collision_handler(i, i).post_solve = post_solve_bird_line
            else:
                # pymunk 7.x API。
                self.space.on_collision(i, i, post_solve=post_solve_bird_line)

    def create_ball(self, space, x, y, m=1, r=7, i=1):
        """创建一个可参与物理模拟的圆形水果刚体。"""

        # 圆形刚体的转动惯量。质量和半径会影响碰撞后的滚动手感。
        ball_moment = pymunk.moment_for_circle(m, 0, r)

        # 创建动态刚体，并设置圆心位置。
        ball_body = pymunk.Body(m, ball_moment)
        ball_body.position = x, y

        # Circle 是真正参与碰撞的形状。
        ball_shape = pymunk.Circle(ball_body, r)

        # 弹性较低，水果碰撞后不会像橡胶球一样弹飞。
        ball_shape.elasticity = 0.18

        # 摩擦较高，水果落到底部后更容易堆叠稳定。
        ball_shape.friction = 0.88

        # collision_type 用于区分水果等级，同等级碰撞才会触发合成。
        ball_shape.collision_type = i

        # 同时加入刚体和形状，pymunk 才会模拟运动和碰撞。
        space.add(ball_body, ball_shape)
        return ball_shape

    def create_segment(self, from_, to_, thickness, space, color):
        """创建静态边界线段。"""

        # Segment 绑定在 static_body 上，表示不会被重力和碰撞推动。
        segment_shape = pymunk.Segment(space.static_body, from_, to_, thickness)

        # 颜色主要供 pymunk 调试绘制使用；正式画面由 app.py 自己绘制。
        segment_shape.color = pg.color.THECOLORS[color]

        # 边界摩擦影响水果贴墙和落底后的滑动程度。
        segment_shape.friction = 0.6

        # 加入物理空间后才会参与碰撞。
        space.add(segment_shape)
        return segment_shape

    def show_score(self):
        """旧版简易分数绘制函数。

        当前正式 HUD 在 `daxigua.app.Board._draw_hud()` 中绘制。
        这个函数保留给调试或后续简化版本复用。
        """

        score_font = pg.font.Font(None, 36)
        score_text = score_font.render(
            'score: {}'.format(str(self.score)), True, (255, 165, 0))
        text_rect = score_text.get_rect()
        text_rect.topleft = [10, 10]
        self.surface.blit(score_text, text_rect)

    def check_fail(self):
        """检测游戏是否失败。

        失败规则：
        - 如果已有水果持续停留在生成线 `init_y` 上方；
        - 并且持续时间超过 `create_time` 秒；
        - 则判定本局失败。
        """

        exist = False

        if len(self.balls):
            # 跳过最后一个球，是继承自旧逻辑的缓冲处理：
            # 刚生成或刚合成的水果可能短暂位于生成线附近，不立即计入失败。
            for i, ball in enumerate(self.balls[:-1]):
                if ball:
                    # pygame/pymunk 坐标系中 y 越小越靠上；小于 init_y 表示越过警戒线。
                    if int(ball.body.position[1]) < self.init_y:
                        self.fail_count += 1
                        exist = True
                        break

        if exist:
            # `FPS * create_time` 把秒转换为帧数。
            # 只有连续越线超过阈值，才结束游戏，避免瞬间弹跳导致误判。
            if self.fail_count > self.FPS * self.create_time:
                self.alive = False
                return True
            return False

        # 没有水果越线时清空累计计数。
        self.fail_count = 0
        return False

    def run(self):
        """由子类实现完整游戏循环。"""

        pass
