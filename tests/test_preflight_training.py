"""正式训练环境预检的配置契约测试。"""

from __future__ import annotations

import unittest

from tools.preflight_training import (
    PROJECT_ROOT,
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


if __name__ == '__main__':
    unittest.main()
