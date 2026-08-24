"""生成一次性稳定随机场景，并绘制水果对堵塞风险人工检查画廊。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from daxigua.core.rules import merged_fruit_physics_radius  # noqa: E402
from daxigua.rl.scenario_pair_risk_evaluator import (  # noqa: E402
    ScenarioPairRiskEvaluator,
)
from daxigua.simulator import TensorVectorSimulator  # noqa: E402
from daxigua.simulator.config import SimulatorConfig  # noqa: E402


COLORS = (
    '#7d3c98', '#d35400', '#e67e22', '#f4d03f', '#73c66b', '#d94b45',
    '#f1948a', '#d4a017', '#8d6e63', '#239b56', '#117864',
)
LEVELS = tuple(range(1, 12))
LEVEL_WEIGHTS = (
    0.400, 0.250, 0.150, 0.080, 0.050, 0.030,
    0.018, 0.010, 0.005, 0.002, 0.001,
)
TARGET_MAX_FRUITS = {7: 60, 8: 60, 9: 56, 10: 52, 11: 45}
TARGET_MIN_FRUITS = {7: 42, 8: 44, 9: 46, 10: 44, 11: 35}
SEED_STRIDE = 1_000_003


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            '在CPU上生成稳定随机场景，并把冻结风险模型的全部同级高等级'
            '水果对预测绘制为静态图片。'
        ),
    )
    parser.add_argument('--scenes', type=int, default=256)
    parser.add_argument(
        '--workers', type=int,
        default=max(1, min(8, (os.cpu_count() or 4) // 2)),
        help='独立Tensor CPU物理进程数，默认不超过8。',
    )
    parser.add_argument('--min-fruits', type=int, default=30)
    parser.add_argument('--max-fruits', type=int, default=60)
    parser.add_argument('--fps', type=int, choices=(30, 120), default=30)
    parser.add_argument('--seed-base', type=int, default=202_608_240)
    parser.add_argument('--max-attempts', type=int, default=256)
    parser.add_argument(
        '--settle-timeout', type=float, default=12.0,
        help='每个候选场景允许的模拟物理秒数；离线快速推进，不等待墙钟。',
    )
    parser.add_argument(
        '--checkpoint', type=Path,
        help='省略时从runs/pair_risk或runs/pair-risk自动选择最新best.pt。',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        help='省略时在系统临时目录创建独立画廊目录。',
    )
    parser.add_argument(
        '--max-line-labels', type=int, default=12,
        help='画布上显示大号百分比的最高风险配对数；全部配对仍绘制连线并写入JSON。',
    )
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args(argv)
    if args.scenes <= 0:
        parser.error('--scenes must be positive')
    if args.workers <= 0:
        parser.error('--workers must be positive')
    if not 1 <= args.min_fruits <= args.max_fruits <= 60:
        parser.error('fruit range must satisfy 1 <= min <= max <= 60')
    if args.min_fruits < 2:
        parser.error('--min-fruits must leave room for a target pair')
    if args.max_attempts <= 0 or args.settle_timeout <= 0.0:
        parser.error('attempt and timeout limits must be positive')
    if args.max_line_labels < 0:
        parser.error('--max-line-labels must be non-negative')
    return args


def _discover_checkpoint():
    candidates = []
    for directory in ('pair_risk', 'pair-risk'):
        candidates.extend(
            (PROJECT_ROOT / 'runs' / directory).glob('**/best.pt')
        )
    return max(
        (path for path in candidates if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        default=None,
    )


def _weighted_level(rng, *, exclude=None):
    while True:
        level = rng.choices(LEVELS, weights=LEVEL_WEIGHTS, k=1)[0]
        if level != exclude:
            return level


def _candidate_position(rng, level, placed, config):
    radius = float(merged_fruit_physics_radius(level))
    left = float(config.wall_width) + radius + 1.0
    right = float(config.board_width - config.wall_width) - radius - 1.0
    top = float(config.spawn_y) + radius + 12.0
    bottom = float(config.board_height - config.wall_width) - radius - 1.0
    if left >= right or top >= bottom:
        return None
    # 先做普通二维无重叠采样，再交给训练同源重力物理自然压实。直接按
    # 垂直落点堆叠会制造过于疏松的高塔，使本可容纳的30～60水果提前越线。
    for _ in range(4096):
        x = rng.uniform(left, right)
        y = rng.uniform(top, bottom)
        overlaps = any(
            math.hypot(x - other['x'], y - other['y'])
            <= radius + other['physics_radius'] + 2.0
            for other in placed
        )
        if not overlaps:
            return y, x, radius
    return None


def generate_random_scene(index, seed, config, min_fruits, max_fruits):
    """生成无重叠的重力堆叠初态，并保证一个分层目标水果对。"""

    rng = random.Random(int(seed))
    target_level = 7 + (int(index) % 5)
    backup_level = 8 if target_level == 7 else 7
    upper = min(int(max_fruits), TARGET_MAX_FRUITS[target_level])
    lower = min(
        max(int(min_fruits), TARGET_MIN_FRUITS[target_level]), upper
    )
    fruit_count = rng.randint(lower, upper)
    remaining = [
        _weighted_level(rng, exclude=target_level)
        for _ in range(fruit_count - 4)
    ]
    # 先放大水果可明显降低随机圆装填的失败率；最终位置仍由真实重力与
    # 碰撞求解决定，不把这里的初始高度当成稳定状态。
    remaining.sort(key=lambda level: (level, rng.random()), reverse=True)
    placement_levels = [
        target_level, target_level, backup_level, backup_level,
    ] + remaining
    placed = []
    age_scale = int(config.physics_fps / 30)
    for slot, level in enumerate(placement_levels):
        candidate = _candidate_position(rng, level, placed, config)
        if candidate is None:
            return None
        y, x, radius = candidate
        placed.append({
            'id': slot + 1,
            'level': int(level),
            'physics_radius': radius,
            'x': round(x, 4),
            'y': round(y, 4),
            'vx': 0.0,
            'vy': 0.0,
            'angle': rng.uniform(-math.pi, math.pi),
            'angular_velocity': 0.0,
            'age_frames': int(
                max(30, (fruit_count - slot) * rng.randint(10, 42)
                    + rng.randint(0, 180)) * age_scale
            ),
        })
    queue = [_weighted_level(rng) for _ in range(4)]
    queue = [min(5, level) for level in queue]
    return {
        'name': f'random_pair_risk_{index + 1:04d}',
        'fps': int(config.physics_fps),
        'queue': queue,
        'score': rng.randint(1000, 12000),
        'step_count': rng.randint(max(60, fruit_count * 3), 900),
        'danger_progress': 0.0,
        'over_danger_line': False,
        'fruits': placed,
        '_generation': {
            'index': int(index),
            'seed': int(seed),
            'target_level': target_level,
            'backup_level': backup_level,
            'initial_fruit_count': fruit_count,
        },
    }


def _simulator_config(fps):
    factory = (
        SimulatorConfig.training_fast
        if int(fps) == 30 else SimulatorConfig.high_fidelity_fast
    )
    return factory(
        max_fruits=64,
        action_count=21,
        queue_length=4,
        use_cuda_extension=False,
        track_action_effects=False,
        drop_fast_forward=False,
    )


def _load_scene_batch(simulator, scenes):
    batch_size = simulator.num_envs
    count = len(scenes)
    seeds = torch.arange(
        batch_size, dtype=torch.int64, device=simulator.device
    ) + 202_608_240
    queues = torch.ones(
        (batch_size, 4), dtype=torch.int64, device=simulator.device
    )
    for row, scene in enumerate(scenes):
        queues[row] = torch.tensor(
            scene['queue'], dtype=torch.int64, device=simulator.device
        )
    simulator.reset(seeds=seeds, fruit_queue=queues)
    for row, scene in enumerate(scenes):
        fruits = scene['fruits']
        fruit_count = len(fruits)
        slots = slice(0, fruit_count)
        positions = torch.tensor(
            [(fruit['x'], fruit['y']) for fruit in fruits],
            dtype=torch.float32, device=simulator.device,
        )
        velocities = torch.tensor(
            [(fruit['vx'], fruit['vy']) for fruit in fruits],
            dtype=torch.float32, device=simulator.device,
        )
        levels = torch.tensor(
            [fruit['level'] for fruit in fruits],
            dtype=torch.int64, device=simulator.device,
        )
        radii = torch.tensor(
            [fruit['physics_radius'] for fruit in fruits],
            dtype=torch.float32, device=simulator.device,
        )
        masses = simulator._mass_table[levels]
        simulator.positions[row, slots] = positions
        simulator.velocities[row, slots] = velocities
        simulator.angles[row, slots] = torch.tensor(
            [fruit['angle'] for fruit in fruits],
            dtype=torch.float32, device=simulator.device,
        )
        simulator.angular_velocities[row, slots] = torch.tensor(
            [fruit['angular_velocity'] for fruit in fruits],
            dtype=torch.float32, device=simulator.device,
        )
        simulator.levels[row, slots] = levels
        simulator.physics_radii[row, slots] = radii
        simulator.fruit_ids[row, slots] = torch.tensor(
            [fruit['id'] for fruit in fruits],
            dtype=torch.int64, device=simulator.device,
        )
        simulator.age_frames[row, slots] = torch.tensor(
            [fruit['age_frames'] for fruit in fruits],
            dtype=torch.int64, device=simulator.device,
        )
        simulator.active[row, slots] = True
        simulator.masses[row, slots] = masses
        simulator.inverse_masses[row, slots] = masses.reciprocal()
        simulator.inverse_inertias[row, slots] = (
            0.5 * masses * radii.square()
        ).reciprocal()
        simulator.next_fruit_id[row] = max(
            fruit['id'] for fruit in fruits
        ) + 1
        simulator.score[row] = int(scene['score'])
        simulator.last_score[row] = int(scene['score'])
        simulator.step_count[row] = int(scene['step_count'])
    simulator.reset_incremental_progress(stable=False)
    if count < batch_size:
        simulator.needs_reset[count:] = True


def _settled_row_scene(simulator, row, raw):
    slots = torch.nonzero(
        simulator.active[row], as_tuple=False
    ).flatten().tolist()
    fruits = []
    for slot in slots:
        fruits.append({
            'id': int(simulator.fruit_ids[row, slot].item()),
            'level': int(simulator.levels[row, slot].item()),
            'physics_radius': round(
                float(simulator.physics_radii[row, slot].item()), 4
            ),
            'x': round(float(simulator.positions[row, slot, 0].item()), 4),
            'y': round(float(simulator.positions[row, slot, 1].item()), 4),
            'vx': round(float(simulator.velocities[row, slot, 0].item()), 4),
            'vy': round(float(simulator.velocities[row, slot, 1].item()), 4),
            'angle': round(float(simulator.angles[row, slot].item()), 6),
            'angular_velocity': round(
                float(simulator.angular_velocities[row, slot].item()), 6
            ),
            'age_frames': int(simulator.age_frames[row, slot].item()),
        })
    over_danger_line = any(
        fruit['y'] - fruit['physics_radius'] < simulator.config.spawn_y
        for fruit in fruits
    )
    return {
        'name': raw['name'],
        'fps': int(simulator.config.physics_fps),
        'queue': [
            int(value) for value in simulator.fruit_queue[row].tolist()
        ],
        'score': int(simulator.score[row].item()),
        'step_count': int(simulator.step_count[row].item()),
        'danger_progress': round(
            float(simulator.fail_frames[row].item())
            / max(1, simulator.config.danger_frame_limit),
            6,
        ),
        'over_danger_line': over_danger_line,
        'fruits': fruits,
        '_generation': {
            **raw['_generation'],
            'settled_fruit_count': len(fruits),
            'physics_frame': int(simulator.physics_frame[row].item()),
        },
    }


def _settle_scene_batch(simulator, scenes, simulated_seconds):
    _load_scene_batch(simulator, scenes)
    count = len(scenes)
    pending = torch.zeros(
        simulator.num_envs, dtype=torch.bool, device=simulator.device
    )
    pending[:count] = True
    results = [None] * count
    maximum_frames = max(
        1, round(float(simulated_seconds) * simulator.config.physics_fps)
    )
    for _ in range(maximum_frames):
        physics = simulator.advance_incremental_frame()
        stable = physics.stable & pending
        done = physics.done & pending
        for row in torch.nonzero(stable, as_tuple=False).flatten().tolist():
            results[row] = _settled_row_scene(simulator, row, scenes[row])
        finished = stable | done
        pending &= ~finished
        simulator.needs_reset |= finished
        if not bool(pending.any().item()):
            break
    return results


def _eligible_pair_count(scene):
    counts = {}
    for fruit in scene['fruits']:
        level = int(fruit['level'])
        if 7 <= level <= 11:
            counts[level] = counts.get(level, 0) + 1
    return sum(count * (count - 1) // 2 for count in counts.values())


_PROCESS_CONFIG = None
_PROCESS_SIMULATOR = None


def _initialize_process_worker(fps):
    global _PROCESS_CONFIG, _PROCESS_SIMULATOR
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _PROCESS_CONFIG = _simulator_config(fps)
    _PROCESS_SIMULATOR = TensorVectorSimulator(
        1, config=_PROCESS_CONFIG, device='cpu'
    )


def _build_stable_scene_process(index, settings):
    if _PROCESS_CONFIG is None or _PROCESS_SIMULATOR is None:
        raise RuntimeError('CPU physics worker was not initialized')
    for attempt in range(int(settings['max_attempts'])):
        seed = (
            int(settings['seed_base']) + int(index) * SEED_STRIDE
            + attempt * 97
        )
        candidate = generate_random_scene(
            index,
            seed,
            _PROCESS_CONFIG,
            settings['min_fruits'],
            settings['max_fruits'],
        )
        if candidate is None:
            continue
        candidate['_generation'].update({
            'accepted_seed': seed,
            'attempt': attempt + 1,
        })
        settled = _settle_scene_batch(
            _PROCESS_SIMULATOR,
            [candidate],
            settings['settle_timeout'],
        )[0]
        if settled is None:
            continue
        if (
                not max(12, int(settings['min_fruits']) // 2)
                <= len(settled['fruits']) <= int(settings['max_fruits'])
                or _eligible_pair_count(settled) <= 0
                or settled['over_danger_line']):
            continue
        return settled
    raise RuntimeError(
        f"scene {int(index) + 1} failed after "
        f"{settings['max_attempts']} attempts"
    )


def generate_stable_scenes(args):
    results = [None] * int(args.scenes)
    settings = {
        'seed_base': int(args.seed_base),
        'min_fruits': int(args.min_fruits),
        'max_fruits': int(args.max_fruits),
        'max_attempts': int(args.max_attempts),
        'settle_timeout': float(args.settle_timeout),
    }
    completed = 0
    with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_initialize_process_worker,
            initargs=(int(args.fps),)) as executor:
        futures = {
            executor.submit(
                _build_stable_scene_process, index, settings
            ): index
            for index in range(int(args.scenes))
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            if completed == 1 or completed % 16 == 0 or completed == args.scenes:
                print(
                    f'[scene] {completed}/{args.scenes} stable scenes',
                    flush=True,
                )
    return results


def risk_color(probability):
    risk = max(0.0, min(1.0, float(probability)))
    low = (33, 161, 121)
    high = (230, 57, 70)
    rgb = tuple(round(a + (b - a) * risk) for a, b in zip(low, high))
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def render_scene(scene, prediction, config, output_path, *, max_labels=12, dpi=150):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    figure = plt.figure(figsize=(10.4, 12.2))
    grid = figure.add_gridspec(1, 2, width_ratios=(3.35, 1.65), wspace=0.05)
    axis = figure.add_subplot(grid[0, 0])
    summary = figure.add_subplot(grid[0, 1])
    axis.set_facecolor('#f7f1e5')
    wall = float(config.wall_width)
    width = float(config.board_width)
    height = float(config.board_height)
    axis.add_patch(Rectangle(
        (wall, 0), width - 2.0 * wall, height - wall,
        fill=False, edgecolor='#4b3b2a', linewidth=3.0, zorder=6,
    ))
    axis.axhline(
        float(config.spawn_y), color='#c0392b', linewidth=1.5,
        linestyle='--', alpha=0.75, zorder=1,
    )
    fruits = {int(fruit['id']): fruit for fruit in scene['fruits']}
    pairs = prediction['pairs']
    label_count = min(int(max_labels), len(pairs))
    for rank, pair in enumerate(pairs):
        first = fruits.get(int(pair['fruit_id_i']))
        second = fruits.get(int(pair['fruit_id_j']))
        if first is None or second is None:
            continue
        probability = float(pair['probability'])
        color = risk_color(probability)
        x1, y1 = float(first['x']), float(first['y'])
        x2, y2 = float(second['x']), float(second['y'])
        axis.plot(
            (x1, x2), (y1, y2), color=color,
            linewidth=1.6 + probability * 3.2,
            alpha=0.86, linestyle='--', zorder=2,
        )
        if rank < label_count:
            dx, dy = x2 - x1, y2 - y1
            length = max(1.0, math.hypot(dx, dy))
            offset = ((rank % 5) - 2) * 8.0
            midpoint_x = (x1 + x2) * 0.5 - dy / length * offset
            midpoint_y = (y1 + y2) * 0.5 + dx / length * offset
            axis.text(
                midpoint_x,
                midpoint_y,
                f"L{int(pair['level'])}  {probability * 100:.1f}%",
                ha='center', va='center', fontsize=11.5,
                color='#182132', weight='bold', zorder=7,
                bbox={
                    'boxstyle': 'round,pad=0.28',
                    'facecolor': 'white',
                    'edgecolor': color,
                    'linewidth': 1.5,
                    'alpha': 0.94,
                },
            )
    for fruit in scene['fruits']:
        level = int(fruit['level'])
        x, y = float(fruit['x']), float(fruit['y'])
        radius = float(fruit['physics_radius'])
        axis.add_patch(Circle(
            (x, y), radius,
            facecolor=COLORS[level - 1], edgecolor='white',
            linewidth=1.1, alpha=0.94, zorder=3,
        ))
        axis.text(
            x, y, f"L{level}\n#{int(fruit['id'])}",
            ha='center', va='center', fontsize=7.5,
            color='white', weight='bold', zorder=4,
        )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f"Scene {scene['_generation']['index'] + 1:04d} · "
        f"targets L{scene['_generation']['target_level']}+"
        f"L{scene['_generation']['backup_level']} · "
        f"{len(scene['fruits'])} settled / "
        f"{scene['_generation']['initial_fruit_count']} generated",
        fontsize=13, weight='bold', pad=8,
    )

    summary.axis('off')
    horizon = int(prediction['forecast_horizon'])
    summary.text(
        0.0, 0.985,
        f'Pair-risk forecast\nwithin next {horizon} drops',
        transform=summary.transAxes, ha='left', va='top',
        fontsize=15, weight='bold', color='#182132',
    )
    maximum = float(pairs[0]['probability']) if pairs else 0.0
    metadata = (
        f"seed  {scene['_generation']['accepted_seed']}\n"
        f"pairs  {len(pairs)}\n"
        f"max risk  {maximum * 100:.1f}%\n"
        f"danger  {float(scene['danger_progress']) * 100:.1f}%\n"
        f"inference  {float(prediction['inference_ms']):.2f} ms"
    )
    summary.text(
        0.0, 0.89, metadata,
        transform=summary.transAxes, ha='left', va='top',
        fontsize=11.5, linespacing=1.45, color='#4f5d73',
    )
    summary.text(
        0.0, 0.72, 'Pairs sorted by risk',
        transform=summary.transAxes, ha='left', va='top',
        fontsize=12.5, weight='bold', color='#182132',
    )
    y = 0.685
    max_rows = 22
    for pair in pairs[:max_rows]:
        probability = float(pair['probability'])
        summary.text(
            0.0, y,
            f"L{int(pair['level'])}  #{int(pair['fruit_id_i'])} ↔ "
            f"#{int(pair['fruit_id_j'])}",
            transform=summary.transAxes, ha='left', va='top',
            fontsize=9.8, color='#364152',
        )
        summary.text(
            1.0, y, f'{probability * 100:.1f}%',
            transform=summary.transAxes, ha='right', va='top',
            fontsize=11.2, weight='bold', color=risk_color(probability),
        )
        y -= 0.028
    if len(pairs) > max_rows:
        summary.text(
            0.0, y, f'+ {len(pairs) - max_rows} more pairs in JSON',
            transform=summary.transAxes, ha='left', va='top',
            fontsize=9.5, color='#6b7280',
        )
    summary.text(
        0.0, 0.03,
        'Continuous proxy probability, not a current\n'
        'blockage label or a causal conclusion.',
        transform=summary.transAxes, ha='left', va='bottom',
        fontsize=9.5, color='#6b7280', style='italic',
    )
    figure.savefig(output_path, dpi=int(dpi), bbox_inches='tight')
    plt.close(figure)


def _json_ready_scene(scene):
    return {
        key: value for key, value in scene.items()
        if key != '_generation'
    }


def run(args):
    checkpoint = args.checkpoint or _discover_checkpoint()
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError('pair-risk best.pt was not found')
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix='xigua_pair_risk_random_'))
    else:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / 'images'
    scene_dir = output_dir / 'scenes'
    image_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    started = time.perf_counter()
    render_config = _simulator_config(args.fps)
    scenes = generate_stable_scenes(args)

    evaluator = ScenarioPairRiskEvaluator(checkpoint, device='cpu')
    rows = []
    for index, scene in enumerate(scenes):
        prediction = evaluator.evaluate(_json_ready_scene(scene))
        maximum = (
            float(prediction['pairs'][0]['probability'])
            if prediction['pairs'] else 0.0
        )
        stem = (
            f"scene_{index + 1:04d}_L{scene['_generation']['target_level']}"
            f"_max_{round(maximum * 1000):03d}"
        )
        image_path = image_dir / f'{stem}.png'
        json_path = scene_dir / f'{stem}.json'
        render_scene(
            scene,
            prediction,
            render_config,
            image_path,
            max_labels=args.max_line_labels,
            dpi=args.dpi,
        )
        json_path.write_text(json.dumps({
            'purpose': 'random_pair_risk_subjective_gallery',
            'warning': (
                'Synthetic random stable scene; not an accuracy label or '
                'in-distribution policy evaluation.'
            ),
            'generation': scene['_generation'],
            'scene': _json_ready_scene(scene),
            'prediction': prediction,
            'image': str(image_path.relative_to(output_dir)),
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        rows.append({
            'scene': index + 1,
            'target_level': scene['_generation']['target_level'],
            'seed': scene['_generation']['accepted_seed'],
            'generated_fruit_count': (
                scene['_generation']['initial_fruit_count']
            ),
            'settled_fruit_count': len(scene['fruits']),
            'pair_count': prediction['pair_count'],
            'max_probability': round(maximum, 6),
            'danger_progress': scene['danger_progress'],
            'inference_ms': prediction['inference_ms'],
            'image': str(image_path.relative_to(output_dir)),
            'json': str(json_path.relative_to(output_dir)),
        })
        if (index + 1) % 16 == 0 or index + 1 == len(scenes):
            print(f'[render] {index + 1}/{len(scenes)} images', flush=True)
    with (output_dir / 'summary.csv').open(
            'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        'purpose': 'random_pair_risk_subjective_gallery',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'scene_count': len(scenes),
        'parallel_cpu_environments': int(args.workers),
        'fps': int(args.fps),
        'generated_fruit_range': [
            int(args.min_fruits), int(args.max_fruits)
        ],
        'level_weights': {
            f'L{level}': weight
            for level, weight in zip(LEVELS, LEVEL_WEIGHTS)
        },
        'target_levels': [7, 8, 9, 10, 11],
        'model': evaluator.identity,
        'elapsed_seconds': round(time.perf_counter() - started, 3),
        'warning': (
            'This synthetic gallery supports subjective sanity checking only. '
            'It is not a labeled accuracy evaluation.'
        ),
        'files': {
            'images': 'images/',
            'scenes': 'scenes/',
            'summary': 'summary.csv',
        },
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'gallery: {output_dir}', flush=True)
    return output_dir


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except Exception as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
