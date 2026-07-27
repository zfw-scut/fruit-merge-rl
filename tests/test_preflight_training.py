"""正式训练环境预检的配置契约测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.preflight_training import (
    PROJECT_ROOT,
    _cgroup_memory_status,
    _formal_500k_config_audit,
    parse_training_args,
)


class PreflightTrainingTest(unittest.TestCase):
    def _config_args(self, name):
        return parse_training_args((
            '--config',
            str(PROJECT_ROOT / 'configs' / name),
        ))

    def test_formal_500k_config_satisfies_frozen_contract(self):
        audit = _formal_500k_config_audit(
            self._config_args('train_dqn_causal_500k.toml')
        )

        self.assertTrue(audit['passed'], audit['mismatches'])
        self.assertEqual(audit['mismatches'], {})

    def test_smoke_config_cannot_replace_formal_preflight_target(self):
        audit = _formal_500k_config_audit(
            self._config_args(
                'train_dqn_causal_smoke_5k.toml'
            )
        )

        self.assertFalse(audit['passed'])
        self.assertIn('total_updates', audit['mismatches'])

    def test_disabled_attribution_fails_formal_contract(self):
        args = self._config_args(
            'train_dqn_causal_500k.toml'
        )
        args.counterfactual_enabled = False
        args.shapley_enabled = False
        args.lambda_cf = 0.0

        audit = _formal_500k_config_audit(args)

        self.assertFalse(audit['passed'])
        self.assertIn(
            'counterfactual_enabled',
            audit['mismatches'],
        )
        self.assertIn('shapley_enabled', audit['mismatches'])
        self.assertIn('lambda_cf', audit['mismatches'])

    def test_cgroup_memory_uses_reclaimable_working_set_headroom(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / 'memory.max').write_text(
                str(25 * 1024 ** 3),
                encoding='ascii',
            )
            (root / 'memory.current').write_text(
                str(19 * 1024 ** 3),
                encoding='ascii',
            )
            (root / 'memory.stat').write_text(
                f'anon {1 * 1024 ** 3}\n'
                f'inactive_file {16 * 1024 ** 3}\n',
                encoding='ascii',
            )
            paths = ((
                root / 'memory.max',
                root / 'memory.current',
                root / 'memory.stat',
            ),)
            status = _cgroup_memory_status(paths)

        self.assertEqual(status['raw_available_bytes'], 6 * 1024 ** 3)
        self.assertEqual(
            status['effective_available_bytes'],
            22 * 1024 ** 3,
        )
        self.assertEqual(status['working_set_bytes'], 3 * 1024 ** 3)

    def test_cgroup_memory_missing_stat_remains_conservative(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / 'memory.max').write_text(
                str(25 * 1024 ** 3),
                encoding='ascii',
            )
            (root / 'memory.current').write_text(
                str(19 * 1024 ** 3),
                encoding='ascii',
            )
            paths = ((
                root / 'memory.max',
                root / 'memory.current',
                root / 'missing.stat',
            ),)
            status = _cgroup_memory_status(paths)

        self.assertEqual(
            status['effective_available_bytes'],
            6 * 1024 ** 3,
        )


if __name__ == '__main__':
    unittest.main()
