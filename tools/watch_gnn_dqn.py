"""加载 GNN-DQN checkpoint，生成带模型决策信息的本地游戏页面。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daxigua.rl.viewer import (  # noqa: E402
    load_viewer_model,
    run_viewer_episode,
    write_viewer_episode,
)
from daxigua.simulator import write_replay_catalog  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='在浏览器游戏页面观看 GNN-DQN checkpoint 自动游玩。'
    )
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument(
        '--physics-fps', type=int, choices=(30, 120), default=120
    )
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260805)
    parser.add_argument('--max-drops', type=int, default=1000)
    parser.add_argument('--frame-stride', type=int, default=2)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'recordings' / 'model-viewer',
    )
    parser.add_argument(
        '--no-trace', action='store_true',
        help='不额外保存可重新渲染的 PT.GZ 逐帧追踪。',
    )
    parser.add_argument(
        '--open', action='store_true',
        help='生成完成后用系统默认浏览器打开游戏页面。',
    )
    return parser.parse_args()


def _episode_entry(index, episode, replay_path, trace_path):
    summary = episode.summary
    physics_frames = sum(
        int(trace.frame_numbers[0, int(trace.record_counts[0]) - 1])
        for trace in episode.traces
    )
    settle_timeouts = sum(
        int(trace.settle_timeout[0])
        for trace in episode.traces
        if trace.settle_timeout is not None
    )
    if summary['terminated']:
        end_kind = 'terminated'
    elif summary['truncated']:
        end_kind = 'truncated'
    else:
        end_kind = 'capped'
    return {
        'env_index': index,
        'step_count': summary['drops'],
        'score': summary['score'],
        'physics_frames_in_replay': physics_frames,
        'end_kind': end_kind,
        'settle_timeout_count': settle_timeouts,
        'settle_timeout_rate': (
            settle_timeouts / max(1, summary['drops'])
        ),
        'replay': replay_path,
        'trace': trace_path,
    }


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError('episodes must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_viewer_model(args.checkpoint, device=args.device)
    entries = []
    summaries = []
    for index in range(args.episodes):
        episode = run_viewer_episode(
            loaded,
            physics_fps=args.physics_fps,
            seed=args.seed + index,
            max_drops=args.max_drops,
            frame_stride=args.frame_stride,
        )
        replay_name = (
            'index.html'
            if args.episodes == 1
            else f'episode_{index + 1:03d}.html'
        )
        replay_path = args.output_dir / replay_name
        trace_path = (
            None
            if args.no_trace
            else args.output_dir / f'episode_{index + 1:03d}.pt.gz'
        )
        write_viewer_episode(
            replay_path,
            episode,
            loaded,
            trace_path=trace_path,
            title=(
                f'GNN-DQN 模型游玩 · 第 {index + 1} 局 · '
                f'{args.physics_fps} FPS'
            ),
        )
        entries.append(_episode_entry(
            index + 1, episode, replay_path, trace_path
        ))
        summaries.append(episode.summary)

    if args.episodes == 1:
        viewer_path = entries[0]['replay']
    else:
        viewer_path = write_replay_catalog(
            args.output_dir / 'index.html',
            entries,
            title='GNN-DQN 本地模型观看器',
            description=(
                f'{args.checkpoint.name} · greedy · '
                f'{args.physics_fps} FPS 真实物理；选择一局观看逐帧运动和 Q 值。'
            ),
        )
    report = {
        'viewer': str(Path(viewer_path).resolve()),
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'device': str(loaded.device),
        'physics_fps': args.physics_fps,
        'episodes': summaries,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.open:
        webbrowser.open(Path(viewer_path).resolve().as_uri())


if __name__ == '__main__':
    main()
