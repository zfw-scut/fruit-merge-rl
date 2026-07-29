"""独立训练 checkpoint 基础设施测试。"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import torch

from daxigua_rl.training import checkpointing


class _StatefulComponent:
    def __init__(self, value):
        self.value = value
        self.validated_manifest = None

    def state_dict(self):
        return {'value': self.value}

    def load_state_dict(self, state):
        self.value = state['value']

    def checkpoint_manifest(self):
        return {'kind': 'test-component', 'schema_version': 1}

    def validate_checkpoint_manifest(self, manifest):
        if manifest['kind'] != 'test-component':
            raise ValueError('wrong component kind')
        self.validated_manifest = manifest


class CheckpointingTest(unittest.TestCase):
    def _manifest(self, config, **kwargs):
        with mock.patch.object(
                checkpointing.torch.cuda,
                'is_available',
                return_value=False):
            return checkpointing.create_run_manifest(
                config,
                created_at_utc='2026-07-27T00:00:00Z',
                **kwargs,
            )

    def test_config_fingerprint_is_canonical_and_ignores_default_mutables(self):
        first = {
            'learning_rate': 1e-4,
            'model': {'layers': (64, 128), 'features': {'b', 'a'}},
            'run_dir': Path('runs/first'),
            'total_updates': 10,
            'log_interval': 5,
        }
        second = {
            'log_interval': 999,
            'total_updates': 500_000,
            'run_dir': Path('runs/resumed'),
            'model': {'features': {'a', 'b'}, 'layers': [64, 128]},
            'learning_rate': 1e-4,
        }

        self.assertEqual(
            checkpointing.config_fingerprint(first),
            checkpointing.config_fingerprint(second),
        )
        second['learning_rate'] = 2e-4
        self.assertNotEqual(
            checkpointing.config_fingerprint(first),
            checkpointing.config_fingerprint(second),
        )

    def test_config_fingerprint_accepts_namespace_and_dataclass(self):
        @dataclass
        class Config:
            seed: int
            total_updates: int

        namespace = argparse.Namespace(seed=7, total_updates=10)
        dataclass_config = Config(seed=7, total_updates=999)
        self.assertEqual(
            checkpointing.config_fingerprint(namespace),
            checkpointing.config_fingerprint(dataclass_config),
        )

    def test_non_finite_or_ambiguous_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            checkpointing.config_fingerprint({'learning_rate': float('nan')})
        with self.assertRaisesRegex(TypeError, 'key'):
            checkpointing.config_fingerprint({1: 'ambiguous'})

    def test_manifest_round_trip_and_tamper_detection(self):
        config = {
            'seed': 3,
            'learning_rate': 1e-4,
            'total_updates': 100,
        }
        manifest = self._manifest(
            config,
            metadata={'experiment': 'first-large-run'},
        )

        restored = checkpointing.RunManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored.config_fingerprint, manifest.config_fingerprint)
        self.assertEqual(restored.metadata['experiment'], 'first-large-run')
        self.assertEqual(
            restored.schema_version,
            checkpointing.RUN_MANIFEST_SCHEMA_VERSION,
        )

        tampered = manifest.to_dict()
        tampered['config']['seed'] = 4
        with self.assertRaisesRegex(
                checkpointing.CheckpointError,
                'fingerprint'):
            checkpointing.RunManifest.from_dict(tampered)

    def test_resume_validation_allows_only_recorded_mutable_fields(self):
        original = {
            'seed': 3,
            'learning_rate': 1e-4,
            'total_updates': 100,
            'logging': {'interval': 10, 'format': 'csv'},
        }
        manifest = self._manifest(
            original,
            mutable_fields={
                'total_updates',
                'logging.interval',
            },
        )
        resumed = {
            **original,
            'total_updates': 500_000,
            'logging': {'interval': 100, 'format': 'csv'},
        }
        checkpointing.validate_resume_config(manifest, resumed)

        resumed['logging']['format'] = 'json'
        with self.assertRaises(
                checkpointing.ResumeConfigMismatchError) as caught:
            checkpointing.validate_resume_config(manifest, resumed)
        self.assertIn('logging.format', caught.exception.changed_fields)

    def test_resume_validation_can_explicitly_expand_mutable_fields(self):
        original = {'seed': 3, 'device': 'cpu', 'total_updates': 10}
        manifest = self._manifest(original)
        resumed = {'seed': 3, 'device': 'cuda', 'total_updates': 20}

        with self.assertRaises(checkpointing.ResumeConfigMismatchError):
            checkpointing.validate_resume_config(manifest, resumed)
        checkpointing.validate_resume_config(
            manifest,
            resumed,
            mutable_fields={
                *checkpointing.DEFAULT_RESUME_MUTABLE_FIELDS,
                'device',
            },
        )

    def test_weights_only_source_path_is_mutable_on_later_resume(self):
        original = {
            'seed': 3,
            'init_checkpoint': 'old/source.pt',
            'total_updates': 100,
        }
        manifest = self._manifest(original)
        resumed = {
            **original,
            'init_checkpoint': None,
            'total_updates': 200,
        }

        checkpointing.validate_resume_config(manifest, resumed)

    def test_python_and_torch_cpu_rng_round_trip(self):
        random.seed(1234)
        torch.manual_seed(5678)
        with mock.patch.object(
                checkpointing.torch.cuda,
                'is_available',
                return_value=False):
            snapshot = checkpointing.capture_rng_state()
        expected_python = [random.random() for _ in range(4)]
        expected_torch = torch.rand(4)

        random.seed(999)
        torch.manual_seed(999)
        checkpointing.restore_rng_state(snapshot)

        self.assertEqual(
            [random.random() for _ in range(4)],
            expected_python,
        )
        torch.testing.assert_close(torch.rand(4), expected_torch)

    def test_all_cuda_rng_states_are_captured_and_restored(self):
        fake_states = (
            torch.tensor([1, 2, 3], dtype=torch.uint8),
            torch.tensor([4, 5, 6], dtype=torch.uint8),
        )
        with (
                mock.patch.object(
                    checkpointing.torch.cuda,
                    'is_available',
                    return_value=True,
                ),
                mock.patch.object(
                    checkpointing.torch.cuda,
                    'get_rng_state_all',
                    return_value=list(fake_states),
                )):
            snapshot = checkpointing.capture_rng_state()

        fake_states[0][0] = 99
        self.assertEqual(snapshot['cuda_device_count'], 2)
        self.assertEqual(snapshot['torch_cuda'][0][0].item(), 1)

        with (
                mock.patch.object(
                    checkpointing.torch.cuda,
                    'is_available',
                    return_value=True,
                ),
                mock.patch.object(
                    checkpointing.torch.cuda,
                    'device_count',
                    return_value=2,
                ),
                mock.patch.object(
                    checkpointing.torch.cuda,
                    'set_rng_state_all',
                ) as set_states):
            checkpointing.restore_rng_state(snapshot)

        restored_states = set_states.call_args.args[0]
        self.assertEqual(len(restored_states), 2)
        torch.testing.assert_close(
            restored_states[1],
            torch.tensor([4, 5, 6], dtype=torch.uint8),
        )

    def test_cuda_device_mismatch_is_detected_before_rng_mutation(self):
        random.seed(11)
        python_before = random.getstate()
        state = {
            'schema_version': checkpointing.RNG_STATE_SCHEMA_VERSION,
            'python': random.Random(99).getstate(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': (
                torch.tensor([1], dtype=torch.uint8),
                torch.tensor([2], dtype=torch.uint8),
            ),
            'cuda_device_count': 2,
        }
        with (
                mock.patch.object(
                    checkpointing.torch.cuda,
                    'is_available',
                    return_value=True,
                ),
                mock.patch.object(
                    checkpointing.torch.cuda,
                    'device_count',
                    return_value=1,
                )):
            with self.assertRaisesRegex(
                    checkpointing.CheckpointError,
                    'count mismatch'):
                checkpointing.restore_rng_state(state)
        self.assertEqual(random.getstate(), python_before)

    def test_atomic_save_uses_same_directory_and_replaces_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / 'nested' / 'checkpoint.pt'
            with mock.patch.object(
                    checkpointing.os,
                    'replace',
                    wraps=os.replace) as replace:
                saved_path = checkpointing.atomic_torch_save(
                    {'value': torch.tensor([1, 2, 3])},
                    destination,
                )

            self.assertEqual(saved_path, destination)
            source_arg, destination_arg = replace.call_args.args
            self.assertEqual(Path(source_arg).parent, destination.parent)
            self.assertEqual(Path(destination_arg), destination)
            loaded = torch.load(
                destination,
                map_location='cpu',
                weights_only=False,
            )
            torch.testing.assert_close(
                loaded['value'],
                torch.tensor([1, 2, 3]),
            )
            self.assertEqual(
                list(destination.parent.glob('*.tmp')),
                [],
            )

    def test_atomic_save_file_cache_hint_is_best_effort(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / 'checkpoint.pt'
            with (
                    mock.patch.object(
                        checkpointing.os,
                        'POSIX_FADV_DONTNEED',
                        4,
                        create=True,
                    ),
                    mock.patch.object(
                        checkpointing.os,
                        'posix_fadvise',
                        side_effect=OSError('unsupported'),
                        create=True,
                    ) as fadvise):
                checkpointing.atomic_torch_save(
                    {'value': torch.tensor([1])},
                    destination,
                )

            self.assertTrue(destination.is_file())
            fadvise.assert_called_once()

    def test_atomic_save_cleans_temporary_file_when_serialization_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / 'checkpoint.pt'
            with mock.patch.object(
                    checkpointing.torch,
                    'save',
                    side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    checkpointing.atomic_torch_save({}, destination)

            self.assertFalse(destination.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_atomic_save_preserves_old_file_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / 'checkpoint.pt'
            destination.write_bytes(b'old-checkpoint')
            with mock.patch.object(
                    checkpointing.os,
                    'replace',
                    side_effect=PermissionError('locked')):
                with self.assertRaisesRegex(PermissionError, 'locked'):
                    checkpointing.atomic_torch_save(
                        {'new': True},
                        destination,
                    )

            self.assertEqual(destination.read_bytes(), b'old-checkpoint')
            self.assertEqual(list(root.glob('*.tmp')), [])

    def test_atomic_clone_hardlink_keeps_old_version_after_source_replace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'latest.pt'
            step = root / 'step_0001.pt'
            checkpointing.atomic_torch_save({'version': 1}, source)

            cloned, method = checkpointing.atomic_clone_file(source, step)

            self.assertEqual(cloned, step)
            self.assertIn(method, {'hardlink', 'copy'})
            self.assertEqual(
                torch.load(
                    step,
                    map_location='cpu',
                    weights_only=False,
                )['version'],
                1,
            )
            if method == 'hardlink':
                self.assertEqual(os.stat(source).st_ino, os.stat(step).st_ino)

            checkpointing.atomic_torch_save({'version': 2}, source)
            self.assertEqual(
                torch.load(
                    source,
                    map_location='cpu',
                    weights_only=False,
                )['version'],
                2,
            )
            self.assertEqual(
                torch.load(
                    step,
                    map_location='cpu',
                    weights_only=False,
                )['version'],
                1,
            )
            self.assertEqual(list(root.glob('*.tmp')), [])

    def test_atomic_clone_falls_back_to_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'latest.pt'
            destination = root / 'best.pt'
            source.write_bytes(b'checkpoint-bytes')

            with mock.patch.object(
                    checkpointing.os,
                    'link',
                    side_effect=OSError('unsupported')):
                cloned, method = checkpointing.atomic_clone_file(
                    source,
                    destination,
                )

            self.assertEqual(cloned, destination)
            self.assertEqual(method, 'copy')
            self.assertEqual(
                destination.read_bytes(),
                b'checkpoint-bytes',
            )
            self.assertEqual(list(root.glob('*.tmp')), [])

    def test_optional_component_without_state_or_manifest_is_skipped(self):
        snapshots = checkpointing.capture_optional_components(
            {'cold_replay': object()},
        )
        self.assertEqual(snapshots, {})

    def test_training_checkpoint_round_trip_restores_generic_component(self):
        config = {
            'seed': 7,
            'learning_rate': 1e-4,
            'total_updates': 100,
        }
        source_component = _StatefulComponent(value=42)
        target_component = _StatefulComponent(value=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'checkpoint.pt'
            with mock.patch.object(
                    checkpointing.torch.cuda,
                    'is_available',
                    return_value=False):
                checkpointing.save_training_checkpoint(
                    path,
                    training_state={
                        'update_step': 25,
                        'weights': torch.tensor([3.0]),
                    },
                    config=config,
                    components={'causal_replay': source_component},
                )
            resumed_config = {**config, 'total_updates': 500_000}
            inspected_payload = checkpointing.load_training_checkpoint(
                path,
                current_config=resumed_config,
            )
            self.assertEqual(
                inspected_payload['training_state']['update_step'],
                25,
            )
            self.assertEqual(target_component.value, 0)
            payload = checkpointing.load_training_checkpoint(
                path,
                current_config=resumed_config,
                components={'causal_replay': target_component},
            )

        self.assertEqual(payload['training_state']['update_step'], 25)
        torch.testing.assert_close(
            payload['training_state']['weights'],
            torch.tensor([3.0]),
        )
        self.assertEqual(target_component.value, 42)
        self.assertEqual(
            target_component.validated_manifest,
            {'kind': 'test-component', 'schema_version': 1},
        )

    def test_config_mismatch_happens_before_component_restore(self):
        config = {'seed': 7, 'learning_rate': 1e-4}
        source_component = _StatefulComponent(value=42)
        target_component = _StatefulComponent(value=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'checkpoint.pt'
            with mock.patch.object(
                    checkpointing.torch.cuda,
                    'is_available',
                    return_value=False):
                checkpointing.save_training_checkpoint(
                    path,
                    training_state={'update_step': 25},
                    config=config,
                    components={'causal_replay': source_component},
                )
            with self.assertRaises(
                    checkpointing.ResumeConfigMismatchError):
                checkpointing.load_training_checkpoint(
                    path,
                    current_config={**config, 'learning_rate': 2e-4},
                    components={'causal_replay': target_component},
                )

        self.assertEqual(target_component.value, 0)
        self.assertIsNone(target_component.validated_manifest)

    def test_invalid_checkpoint_schema_is_rejected(self):
        payload = {
            'schema_version': 999,
            'run_manifest': {},
            'training_state': {},
            'rng_state': {},
            'component_snapshots': {},
        }
        with self.assertRaisesRegex(
                checkpointing.CheckpointError,
                'schema version'):
            checkpointing.validate_training_checkpoint(payload)

    def test_manifest_is_plain_json_compatible_data(self):
        manifest = self._manifest(
            {'seed': 7, 'total_updates': 10},
            metadata={'tags': {'fast', 'causal'}},
        )
        serialized = json.dumps(
            manifest.to_dict(),
            sort_keys=True,
            allow_nan=False,
        )
        self.assertIn('"schema_version": 1', serialized)

    def test_inference_extractor_supports_versioned_and_legacy_checkpoints(self):
        config = {
            'hidden_dim': 32,
            'message_layers': 2,
            'total_updates': 10,
        }
        weights = {'layer.weight': torch.tensor([1.0])}
        with mock.patch.object(
                checkpointing.torch.cuda,
                'is_available',
                return_value=False):
            payload = checkpointing.build_training_checkpoint(
                training_state={'online_model': weights},
                config=config,
            )
        extracted_args, extracted_weights = (
            checkpointing.extract_inference_checkpoint(payload)
        )
        self.assertEqual(extracted_args['hidden_dim'], 32)
        torch.testing.assert_close(
            extracted_weights['layer.weight'],
            weights['layer.weight'],
        )

        legacy_args, legacy_weights = (
            checkpointing.extract_inference_checkpoint({
                'args': config,
                'online_model': weights,
            })
        )
        self.assertEqual(legacy_args, config)
        self.assertIs(legacy_weights, weights)


if __name__ == '__main__':
    unittest.main()
