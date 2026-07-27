import tempfile
import unittest
from pathlib import Path

from tools.monitor_cgroup_memory import (
    event_deltas,
    read_memory_snapshot,
)


class CgroupMemoryMonitorTests(unittest.TestCase):

    def _write_fixture(
            self,
            root,
            *,
            maximum='26843545600',
            current='25379033088',
            inactive_file=18 * 1024 ** 3,
            events=None):
        root = Path(root)
        (root / 'memory.max').write_text(
            f'{maximum}\n',
            encoding='utf-8',
        )
        (root / 'memory.current').write_text(
            f'{current}\n',
            encoding='utf-8',
        )
        (root / 'memory.stat').write_text(
            f'anon 5368709120\ninactive_file {inactive_file}\n',
            encoding='utf-8',
        )
        values = events or {
            'low': 0,
            'high': 0,
            'max': 0,
            'oom': 0,
            'oom_kill': 0,
        }
        (root / 'memory.events').write_text(
            ''.join(f'{name} {value}\n' for name, value in values.items()),
            encoding='utf-8',
        )

    def test_snapshot_excludes_inactive_file_from_working_set(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_fixture(tmp_dir)
            snapshot = read_memory_snapshot(tmp_dir)

        self.assertEqual(snapshot.current, 25_379_033_088)
        self.assertEqual(snapshot.raw_available, 1_464_512_512)
        self.assertEqual(
            snapshot.working_set,
            25_379_033_088 - 18 * 1024 ** 3,
        )
        self.assertEqual(
            snapshot.effective_available,
            snapshot.maximum - snapshot.working_set,
        )
        self.assertGreater(
            snapshot.effective_available,
            4 * 1024 ** 3,
        )

    def test_snapshot_rejects_unlimited_cgroup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_fixture(tmp_dir, maximum='max')
            with self.assertRaisesRegex(
                    RuntimeError,
                    'memory.max is unlimited'):
                read_memory_snapshot(tmp_dir)

    def test_event_deltas_include_pressure_and_oom_counters(self):
        self.assertEqual(
            event_deltas(
                {
                    'low': 1,
                    'high': 2,
                    'max': 3,
                    'oom': 4,
                    'oom_kill': 5,
                },
                {
                    'low': 2,
                    'high': 4,
                    'max': 6,
                    'oom': 8,
                    'oom_kill': 10,
                },
            ),
            {
                'low': 1,
                'high': 2,
                'max': 3,
                'oom': 4,
                'oom_kill': 5,
            },
        )


if __name__ == '__main__':
    unittest.main()
