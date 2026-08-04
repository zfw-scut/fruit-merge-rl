"""不重跑物理模拟，直接把已保存的 PT/PT.GZ 追踪重新渲染成新版回放。"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from daxigua.simulator import (  # noqa: E402
    SimulatorConfig,
    load_trace_archive,
    write_replay_catalog,
    write_replay_html,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'recordings' / 'rendered-replays',
    )
    parser.add_argument('--catalog-name', default='replays.html')
    parser.add_argument('--max-fruits', type=int, default=128)
    parser.add_argument('--texture-dir', type=Path)
    parser.add_argument('--no-textures', action='store_true')
    parser.add_argument('--no-payload-compression', action='store_true')
    return parser.parse_args()


def archive_stem(path):
    name = path.name
    for suffix in ('.pt.gz', '.pth.gz', '.pt', '.pth'):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def trace_metadata(traces):
    first = traces[0]
    last = traces[-1]
    env_indices = first.env_indices.tolist()
    labels = ','.join(str(value) for value in env_indices)
    physics_frames = 0
    for trace in traces:
        count = int(trace.record_counts[0])
        physics_frames += int(trace.frame_numbers[0, count - 1])
    last_count = int(last.record_counts[0])
    done = bool(last.done[0])
    truncated = bool(last.truncated[0])
    return {
        'env_index': labels,
        'step_count': len(traces),
        'score': int(last.scores[0, last_count - 1]),
        'physics_frames_in_replay': physics_frames,
        'end_kind': (
            'terminated' if done else ('truncated' if truncated else 'capped')
        ),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SimulatorConfig(max_fruits=args.max_fruits)
    entries = []
    for trace_path in args.traces:
        traces = load_trace_archive(trace_path)
        metadata = trace_metadata(traces)
        output_path = args.output_dir / f'{archive_stem(trace_path)}.html'
        write_replay_html(
            output_path,
            traces,
            config,
            title=(
                f'环境 {metadata["env_index"]}：'
                f'{metadata["step_count"]} 次投放'
            ),
            texture_dir=args.texture_dir,
            use_textures=not args.no_textures,
            compress_payload=not args.no_payload_compression,
        )
        entries.append({
            **metadata,
            'replay': str(output_path.resolve()),
            'trace': str(trace_path.resolve()),
        })
    catalog_path = write_replay_catalog(
        args.output_dir / args.catalog_name,
        entries,
        title=f'{len(entries)} 条重新渲染的物理回放',
        description='由已有追踪归档生成，没有重新运行或改变物理模拟。',
    )
    print(json.dumps({
        'catalog': str(catalog_path.resolve()),
        'replays': entries,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
