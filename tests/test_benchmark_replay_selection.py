"""长局基准回放抽样策略测试。"""

import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'run_random_until_stop.py'
)
SPEC = importlib.util.spec_from_file_location(
    'run_random_until_stop_for_test', SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReplaySelectionTest(unittest.TestCase):
    def setUp(self):
        self.timeout_counts = torch.tensor([0, 1, 2, 3, 4, 5])
        self.step_counts = torch.full((6,), 10)

    def test_most_timeouts_prefers_largest_counts(self):
        selected = MODULE.select_replay_environments(
            'most-timeouts',
            3,
            1,
            self.timeout_counts,
            self.step_counts,
        )
        self.assertEqual(selected, [5, 4, 3])

    def test_timeout_rate_stratification_covers_multiple_severities(self):
        selected = MODULE.select_replay_environments(
            'timeout-rate-stratified',
            3,
            1,
            self.timeout_counts,
            self.step_counts,
        )
        self.assertEqual(selected, [5, 3, 1])

    def test_uniform_selection_is_reproducible(self):
        first = MODULE.select_replay_environments(
            'uniform', 4, 99, self.timeout_counts, self.step_counts
        )
        second = MODULE.select_replay_environments(
            'uniform', 4, 99, self.timeout_counts, self.step_counts
        )
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
