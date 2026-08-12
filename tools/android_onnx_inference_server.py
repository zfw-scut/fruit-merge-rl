"""为一次性 Android Box2D 长局校验提供持久 ONNX 推理进程。"""

from __future__ import annotations

import sys

import numpy as np
import onnxruntime as ort


MAX_FRUITS = 64
EXPECTED_VALUES = 2 * 128 + 5 * 64 + 4 + 2


def decode(line):
    values = np.fromstring(line, sep=' ', dtype=np.float32)
    if values.size != EXPECTED_VALUES:
        raise ValueError(
            f'expected {EXPECTED_VALUES} state values, got {values.size}'
        )
    offset = 0

    def take(count):
        nonlocal offset
        result = values[offset:offset + count]
        offset += count
        return result

    return {
        'positions': take(128).reshape(1, 64, 2),
        'velocities': take(128).reshape(1, 64, 2),
        'angular_velocities': take(64).reshape(1, 64),
        'levels': take(64).astype(np.int64).reshape(1, 64),
        'physics_radii': take(64).reshape(1, 64),
        'age_frames': take(64).astype(np.int64).reshape(1, 64),
        'active': take(64).astype(np.int64).reshape(1, 64),
        'fruit_queue': take(4).astype(np.int64).reshape(1, 4),
        'danger_progress': take(1),
        'over_danger_line': take(1).astype(np.int64),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: android_onnx_inference_server.py MODEL')
    session = ort.InferenceSession(
        sys.argv[1], providers=('CPUExecutionProvider',)
    )
    print('READY', flush=True)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if line == 'STOP':
            return
        inputs = decode(line)
        q_values = session.run(('q_values',), inputs)[0]
        print(int(q_values.argmax(axis=1)[0]), flush=True)


if __name__ == '__main__':
    main()
