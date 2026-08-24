"""把既有长期堵塞事件重新标注为当前堵塞状态数据集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daxigua.rl.pair_blockage import (  # noqa: E402
    BLOCKAGE_AREA,
    finalize_pair_blockage_dataset,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='用确认事件的onset～end区间生成当前堵塞标签。'
    )
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--confirmation-drops', type=int, default=24)
    parser.add_argument('--shard-rows', type=int, default=65_536)
    parser.add_argument('--output-area', default=BLOCKAGE_AREA)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = finalize_pair_blockage_dataset(
        args.dataset_dir,
        confirmation_drops=args.confirmation_drops,
        shard_rows=args.shard_rows,
        output_area=args.output_area,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
