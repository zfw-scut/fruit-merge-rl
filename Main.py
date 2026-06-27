import math
import random
from array import array
from dataclasses import dataclass

import pygame as pg

from Fruit import create_fruit
from Game import GameBoard


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
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: tuple
    life: float
    max_life: float

    def update(self, dt):
        self.life -= dt
        self.vy += 540 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.radius *= 0.992
        return self.life > 0

    def draw(self, surface, offset):
        alpha = max(0, min(255, int(255 * self.life / self.max_life)))
        radius = max(1, int(self.radius))
        dot = pg.Surface((radius * 2 + 2, radius * 2 + 2), pg.SRCALPHA)
        pg.draw.circle(dot, (*self.color, alpha), (radius + 1, radius + 1), radius)
        surface.blit(dot, (self.x + offset[0] - radius, self.y + offset[1] - radius))


@dataclass
class FloatingText:
    x: float
    y: float
    text: str
    color: tuple
    life: float
    max_life: float
    size: int = 30

    def update(self, dt):
        self.life -= dt
        self.y -= 54 * dt
        return self.life > 0

    def draw(self, surface, font, offset=(0, 0)):
        alpha = max(0, min(255, int(255 * self.life / self.max_life)))
        text_surface = font.render(self.text, True, self.color)
        text_surface.set_alpha(alpha)
        rect = text_surface.get_rect(center=(self.x + offset[0], self.y + offset[1]))
        surface.blit(text_surface, rect)


@dataclass
class ImpactRing:
    x: float
    y: float
    color: tuple
    life: float
    max_life: float
    start_radius: float
    end_radius: float

    def update(self, dt):
        self.life -= dt
        return self.life > 0

    def draw(self, surface, offset):
        progress = 1 - max(0, self.life / self.max_life)
        radius = int(self.start_radius + (self.end_radius - self.start_radius) * progress)
        alpha = max(0, min(190, int(190 * self.life / self.max_life)))
        size = max(8, radius * 2 + 8)
        ring = pg.Surface((size, size), pg.SRCALPHA)
        pg.draw.circle(ring, (*self.color, alpha), (size // 2, size // 2), radius, 3)
        surface.blit(ring, (self.x + offset[0] - size // 2, self.y + offset[1] - size // 2))


class SoundBank:
    def __init__(self):
        self.sounds = {}
        self.enabled = False
        try:
            if not pg.mixer.get_init():
                pg.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self.sounds = {
                'drop': self._tone(220, 0.055, 0.16, 1.7),
                'merge': self._tone(660, 0.09, 0.13, 1.2),
                'game_over': self._tone(120, 0.18, 0.15, 2.4),
            }
            self.enabled = True
        except pg.error:
            self.enabled = False

    def _tone(self, frequency, duration, volume, decay):
        sample_rate = 44100
        sample_count = int(sample_rate * duration)
        samples = array('h')
        for index in range(sample_count):
            t = index / sample_rate
            envelope = (1 - index / sample_count) ** decay
            value = int(32767 * volume * envelope * math.sin(2 * math.pi * frequency * t))
            samples.append(value)
        return pg.mixer.Sound(buffer=samples.tobytes())

    def play(self, name):
        if self.enabled and name in self.sounds:
            self.sounds[name].play()


class Board(GameBoard):
    def __init__(self):
        self.min_width = 360
        self.min_height = 560
        self.display_flags = pg.RESIZABLE
        self.create_time = 2.0
        self.gravity = (0, 1800)
        GameBoard.__init__(self, self.create_time, self.gravity)

        pg.display.set_caption('Merge Melon')
        self.space.iterations = 32
        self.space.damping = 0.995

        self.wall_width = 20
        self.cooldown_ms = 360
        self.keyboard_speed = 360
        self.aim_x = self.init_x
        self.mouse_x = self.init_x
        self.input_mode = 'mouse'
        self.next_i = None
        self.best_score = 0
        self.drop_ready_at = 0
        self.shake = 0
        self.flash = 0

        self.particles = []
        self.rings = []
        self.floating_texts = []
        self.preview_cache = {}

        self.font_title = pg.font.Font(None, 30)
        self.font_score = pg.font.Font(None, 52)
        self.font_label = pg.font.Font(None, 22)
        self.font_popup = pg.font.Font(None, 34)
        self.font_big_popup = pg.font.Font(None, 46)

        self.background = self._build_background()
        self.sound = SoundBank()

        self.init_segment()
        self.setup_collision_handler()
        self._start_round()

    def _build_background(self):
        gradient = pg.Surface((1, self.HEIGHT))
        top = (14, 23, 31)
        bottom = (24, 45, 51)
        for y in range(self.HEIGHT):
            t = y / max(1, self.HEIGHT - 1)
            color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
            gradient.set_at((0, y), color)
        return pg.transform.scale(gradient, self.RES).convert()

    def _pick_fruit_type(self):
        return random.randrange(1, 5)

    def _start_round(self, ready_delay=180):
        self.i = self._pick_fruit_type()
        self.next_i = self._pick_fruit_type()
        self.waiting = True
        self.current_fruit = create_fruit(self.i, self.mouse_x, self._preview_y())
        self.drop_ready_at = pg.time.get_ticks() + ready_delay

    def _restart_game(self):
        self.best_score = max(self.best_score, self.score)
        self.reset()
        self.particles.clear()
        self.rings.clear()
        self.floating_texts.clear()
        self._start_round()

    def _event_size(self, event):
        if hasattr(event, 'size'):
            return event.size
        width = getattr(event, 'w', getattr(event, 'x', self.WIDTH))
        height = getattr(event, 'h', getattr(event, 'y', self.HEIGHT))
        return width, height

    def _resize_window(self, width, height, recreate_display=False):
        old_width, old_height = self.WIDTH, self.HEIGHT
        if not self.resize_world(width, height, recreate_display=recreate_display):
            return

        self.background = self._build_background()
        self.aim_x = self._clamp_drop_x(self.aim_x)
        self.mouse_x = self._clamp_drop_x(self.mouse_x)
        self._keep_balls_in_bounds(old_width, old_height)
        if self.current_fruit:
            self.current_fruit.update_position(int(self.mouse_x), int(self._preview_y()))

    def _keep_balls_in_bounds(self, old_width, old_height):
        if self.WIDTH >= old_width and self.HEIGHT >= old_height:
            return

        for ball in self.balls:
            radius = getattr(ball, 'radius', 20)
            x, y = ball.body.position
            clamped_x = max(self.wall_width + radius, min(self.WIDTH - self.wall_width - radius, x))
            clamped_y = min(self.HEIGHT - self.wall_width - radius, y)
            if clamped_x != x or clamped_y != y:
                ball.body.position = clamped_x, clamped_y
                vx, vy = ball.body.velocity
                ball.body.velocity = vx * 0.45, min(vy, 120)

    def _clamp_drop_x(self, x, fruit_type=None):
        fruit_radius = self.current_fruit.r if self.current_fruit else 24
        if fruit_type:
            fruit_radius = create_fruit(fruit_type, 0, 0).r
        left = self.wall_width + fruit_radius + 2
        right = self.WIDTH - self.wall_width - fruit_radius - 2
        return max(left, min(right, x))

    def _preview_y(self):
        radius = self.current_fruit.r if self.current_fruit else 28
        bob = math.sin(pg.time.get_ticks() * 0.005) * 2
        return self.init_y - radius - 12 + bob

    def _can_drop(self):
        return self.waiting and self.i is not None and pg.time.get_ticks() >= self.drop_ready_at

    def _drop_current(self):
        if not self._can_drop():
            return
        x = int(self._clamp_drop_x(self.mouse_x))
        fruit = create_fruit(self.i, x, self.init_y)
        self.fruits.append(fruit)
        ball = self.create_ball(
            self.space, x, self.init_y, m=max(1, fruit.r // 10),
            r=fruit.r - fruit.r % 5, i=self.i)
        ball.body.velocity = (0, 80)
        self.balls.append(ball)

        color = FRUIT_COLORS.get(self.i, (255, 255, 255))
        self._burst(x, self.init_y, color, 7, speed=120)
        self.rings.append(ImpactRing(x, self.init_y, color, 0.28, 0.28, fruit.r * 0.4, fruit.r * 1.25))
        self.shake = max(self.shake, 0.08)
        self.sound.play('drop')

        self.current_fruit = None
        self.i = None
        self.waiting = False
        self.drop_ready_at = pg.time.get_ticks() + self.cooldown_ms

    def _spawn_after_cooldown(self):
        if self.i is None and pg.time.get_ticks() >= self.drop_ready_at:
            self.i = self.next_i or self._pick_fruit_type()
            self.next_i = self._pick_fruit_type()
            self.waiting = True
            self.current_fruit = create_fruit(self.i, self.mouse_x, self._preview_y())

    def _handle_events(self):
        pending_resize = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                raise SystemExit
            if event.type in (
                    pg.VIDEORESIZE,
                    getattr(pg, 'WINDOWRESIZED', -1),
                    getattr(pg, 'WINDOWSIZECHANGED', -2)):
                pending_resize = self._event_size(event)
                continue
            if event.type == pg.MOUSEMOTION:
                self.input_mode = 'mouse'
                self.aim_x = self._clamp_drop_x(event.pos[0])
            elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                self.input_mode = 'mouse'
                self.aim_x = self._clamp_drop_x(event.pos[0])
                self._drop_current()
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_SPACE, pg.K_RETURN):
                    self._drop_current()
                elif event.key == pg.K_r:
                    self._restart_game()
                elif event.key == pg.K_ESCAPE:
                    raise SystemExit

        if pending_resize:
            self._resize_window(*pending_resize)

    def _update_input(self, dt):
        keys = pg.key.get_pressed()
        direction = 0
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            direction -= 1
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            direction += 1
        if direction:
            self.input_mode = 'keyboard'
            self.aim_x = self._clamp_drop_x(self.aim_x + direction * self.keyboard_speed * dt)
        elif self.input_mode == 'mouse' and pg.mouse.get_focused():
            mouse_x, _ = pg.mouse.get_pos()
            self.aim_x = self._clamp_drop_x(mouse_x)

        self.mouse_x += (self.aim_x - self.mouse_x) * min(1, dt * 18)
        if self.current_fruit:
            self.mouse_x = self._clamp_drop_x(self.mouse_x)
            self.current_fruit.update_position(int(self.mouse_x), int(self._preview_y()))

    def _sync_fruits(self):
        if self.lock:
            return
        for index, ball in enumerate(self.balls):
            if index >= len(self.fruits) or not ball:
                continue
            x, y = int(ball.body.position[0]), int(ball.body.position[1])
            self.fruits[index].update_position(x, y, ball.body.angle)

    def _update_effects(self, dt):
        self.particles = [p for p in self.particles if p.update(dt)]
        self.rings = [r for r in self.rings if r.update(dt)]
        self.floating_texts = [t for t in self.floating_texts if t.update(dt)]
        self.shake = max(0, self.shake - dt * 1.8)
        self.flash = max(0, self.flash - dt * 1.7)

    def _game_over(self):
        old_score = self.score
        self.best_score = max(self.best_score, old_score)
        self.sound.play('game_over')
        self.flash = 0.55
        self.shake = 0.45
        self.reset()
        self.particles.clear()
        self.rings.clear()
        self.floating_texts = [
            FloatingText(self.WIDTH / 2, self.HEIGHT * 0.42, 'GAME OVER', (255, 235, 210), 1.2, 1.2, 46)
        ]
        self._start_round(ready_delay=760)

    def on_fruit_merged(self, fruit_type, x, y, score_delta):
        color = FRUIT_COLORS.get(fruit_type, (255, 255, 255))
        self.best_score = max(self.best_score, self.score)
        self._burst(x, y, color, 16, speed=210)
        self.rings.append(ImpactRing(x, y, color, 0.36, 0.36, 12, 54 + fruit_type * 4))
        if score_delta:
            self.floating_texts.append(
                FloatingText(x, y - 18, '+' + str(score_delta), (255, 246, 170), 0.72, 0.72, 34)
            )
        self.shake = max(self.shake, min(0.28, 0.08 + fruit_type * 0.015))
        self.sound.play('merge')

    def _burst(self, x, y, color, count, speed=180):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            velocity = random.uniform(speed * 0.35, speed)
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
        if self.shake <= 0:
            return (0, 0)
        strength = 10 * self.shake
        return (random.uniform(-strength, strength), random.uniform(-strength, strength))

    def _fruit_preview_image(self, fruit_type, max_size):
        key = (fruit_type, max_size)
        if key in self.preview_cache:
            return self.preview_cache[key]
        fruit = create_fruit(fruit_type, 0, 0)
        image = fruit.image
        scale = min(max_size / image.get_width(), max_size / image.get_height(), 1)
        size = (max(1, int(image.get_width() * scale)), max(1, int(image.get_height() * scale)))
        preview = pg.transform.smoothscale(image, size)
        self.preview_cache[key] = preview
        return preview

    def _draw_background(self):
        self.surface.blit(self.background, (0, 0))

    def _draw_header_panel(self):
        outer_rect = pg.Rect(0, 0, self.WIDTH, self.init_y + 6)
        inner_rect = pg.Rect(18, 8, self.WIDTH - 36, self.init_y - 18)

        pg.draw.rect(self.surface, (16, 25, 35), outer_rect)
        pg.draw.rect(self.surface, (26, 42, 52), inner_rect, border_radius=6)
        pg.draw.rect(self.surface, (42, 66, 73), inner_rect, 1, border_radius=6)

        for y in range(inner_rect.top + 18, inner_rect.bottom - 4, 22):
            pg.draw.line(self.surface, (36, 58, 66), (inner_rect.left + 10, y), (inner_rect.right - 10, y), 1)

        trim_y = self.init_y - 1
        pg.draw.line(self.surface, (78, 58, 43), (0, trim_y - 2), (self.WIDTH, trim_y - 2), 5)
        pg.draw.line(self.surface, (168, 88, 74), (24, trim_y), (self.WIDTH - 24, trim_y), 2)

    def _draw_playfield(self, offset):
        ox, oy = offset
        play_rect = pg.Rect(18 + ox, self.init_y + oy, self.WIDTH - 36, self.HEIGHT - self.init_y - 18)
        inner_rect = play_rect.inflate(-12, -10)
        pg.draw.rect(self.surface, (16, 25, 35), play_rect, border_radius=8)
        pg.draw.rect(self.surface, (26, 42, 52), inner_rect, border_radius=6)

        for y in range(int(self.init_y + 22 + oy), int(self.HEIGHT - 32 + oy), 34):
            pg.draw.line(self.surface, (32, 54, 64), (28 + ox, y), (self.WIDTH - 28 + ox, y), 1)

        wall_color = (98, 75, 58)
        wall_light = (145, 107, 73)
        left_wall = pg.Rect(0 + ox, self.init_y - 4 + oy, 24, self.HEIGHT - self.init_y + 4)
        right_wall = pg.Rect(self.WIDTH - 24 + ox, self.init_y - 4 + oy, 24, self.HEIGHT - self.init_y + 4)
        floor = pg.Rect(0 + ox, self.HEIGHT - 28 + oy, self.WIDTH, 28)
        for rect in (left_wall, right_wall, floor):
            pg.draw.rect(self.surface, wall_color, rect, border_radius=6)
            pg.draw.rect(self.surface, wall_light, rect, 2, border_radius=6)

        line_y = self.init_y + oy
        warning = pg.Surface((self.WIDTH, 10), pg.SRCALPHA)
        pg.draw.line(warning, (255, 116, 92, 190), (22, 5), (self.WIDTH - 22, 5), 2)
        self.surface.blit(warning, (0 + ox, line_y - 5))

    def _draw_aim(self, offset):
        if not self.current_fruit:
            return
        ox, oy = offset
        x = int(self.mouse_x + ox)
        color = (255, 234, 150) if self._can_drop() else (130, 140, 150)
        for y in range(self.init_y + 12, self.HEIGHT - 42, 18):
            pg.draw.line(self.surface, color, (x, y + oy), (x, y + 8 + oy), 2)
        pg.draw.circle(self.surface, color, (x, int(self.init_y + oy)), 5)

    def _draw_fruit(self, fruit, offset=(0, 0), alpha=255, glow=False):
        ox, oy = offset
        rect = fruit.rect.move(ox, oy)
        shadow = pg.Surface((fruit.r * 2, max(8, fruit.r // 2)), pg.SRCALPHA)
        pg.draw.ellipse(shadow, (0, 0, 0, 78), shadow.get_rect())
        self.surface.blit(shadow, (rect.centerx - fruit.r, rect.centery + fruit.r * 0.42))

        if glow:
            glow_surface = pg.Surface((fruit.r * 3, fruit.r * 3), pg.SRCALPHA)
            pg.draw.circle(glow_surface, (255, 246, 180, 44), (fruit.r * 3 // 2, fruit.r * 3 // 2), int(fruit.r * 1.25))
            self.surface.blit(glow_surface, (rect.centerx - fruit.r * 1.5, rect.centery - fruit.r * 1.5))

        image = fruit.image
        if alpha < 255:
            image = image.copy()
            image.set_alpha(alpha)
        self.surface.blit(image, rect)

    def _draw_fruits(self, offset):
        for fruit in self.fruits:
            self._draw_fruit(fruit, offset)

    def _draw_current_fruit(self, offset):
        if not self.current_fruit:
            return
        alpha = 255 if self._can_drop() else 165
        self._draw_fruit(self.current_fruit, offset, alpha=alpha, glow=True)

    def _draw_effects(self, offset):
        for ring in self.rings:
            ring.draw(self.surface, offset)
        for particle in self.particles:
            particle.draw(self.surface, offset)
        for text in self.floating_texts:
            font = self.font_big_popup if text.size >= 40 else self.font_popup
            text.draw(self.surface, font, offset)

    def _draw_hud(self):
        title = self.font_title.render('MERGE MELON', True, (222, 236, 230))
        self.surface.blit(title, (18, 12))

        score_text = self.font_score.render(str(self.score), True, (255, 240, 176))
        self.surface.blit(score_text, (18, 40))

        best_text = self.font_label.render('BEST ' + str(max(self.best_score, self.score)), True, (167, 202, 194))
        self.surface.blit(best_text, (22, 88))

        next_label = self.font_label.render('NEXT', True, (167, 202, 194))
        self.surface.blit(next_label, (self.WIDTH - 86, 18))
        if self.next_i:
            image = self._fruit_preview_image(self.next_i, 54)
            rect = image.get_rect(center=(self.WIDTH - 58, 64))
            self.surface.blit(image, rect)

        if self.flash > 0:
            overlay = pg.Surface(self.RES, pg.SRCALPHA)
            overlay.fill((255, 86, 70, int(120 * self.flash)))
            self.surface.blit(overlay, (0, 0))

    def _draw_scene(self):
        offset = self._shake_offset()
        self._draw_background()
        self._draw_header_panel()
        self._draw_playfield(offset)
        self._draw_aim(offset)
        self._draw_fruits(offset)
        self._draw_current_fruit(offset)
        self._draw_effects(offset)
        self._draw_hud()

    def next_frame(self):
        dt_ms = self.clock.tick(self.FPS)
        dt = min(max(dt_ms / 1000, 1 / self.FPS), 1 / 30)

        self._handle_events()
        self._update_input(dt)
        self._spawn_after_cooldown()

        self.space.step(1 / self.FPS)
        self._sync_fruits()
        self._update_effects(dt)

        if self.check_fail():
            self._game_over()

        self._draw_scene()
        pg.display.flip()

    def run(self):
        while True:
            self.next_frame()


if __name__ == '__main__':
    game = Board()
    game.run()
