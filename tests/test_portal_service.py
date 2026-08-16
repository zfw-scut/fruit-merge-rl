from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Thread
from tempfile import TemporaryDirectory
import unittest
from urllib.request import Request, urlopen

from daxigua.portal.service import (
    PortalServer,
    _subprocess_environment,
    build_tool_command,
    document_revision,
    scan_documents,
)
from daxigua.portal.analysis_data import scan_analysis_datasets
from daxigua.rl.merge_distance_status import scan_merge_distance_runs
from daxigua.rl.merge_potential_status import scan_merge_potential_runs
from tools.open_project_portal import _read_cloud_ssh_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PortalServiceTests(unittest.TestCase):
    def test_analysis_dataset_exposes_small_csv_tables(self):
        with TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            run_dir = project_root / 'runs' / 'analysis' / 'formal'
            analysis_dir = run_dir / 'analysis'
            analysis_dir.mkdir(parents=True)
            (run_dir / 'manifest.json').write_text(json.dumps({
                'purpose': 'merge_potential_t_merge_collection',
                'status': 'complete',
                'completed_episodes': 20_000,
                'transitions': 123_456,
                'checkpoint': '/cloud/final.pt',
                'simulator_config': {
                    'physics_fps': 30,
                    'drop_fast_forward': False,
                },
                'parameters': {'max_drops': 0},
            }), encoding='utf-8')
            (analysis_dir / 'analysis_manifest.json').write_text(json.dumps({
                'horizons': [8, 32],
                'factor_bins': 10,
                'interaction_bins': 5,
                'result': {
                    'episodes': 20_000,
                    'unique_observed_fruits': 30_000,
                    'snapshot_rows': 50_000,
                    'merge_sources': 40_000,
                },
            }), encoding='utf-8')
            (analysis_dir / 'lifecycle_by_level.csv').write_text(
                'level,fruits,eventual_merge_probability_resolved\n'
                '1,100,0.95\n2,80,nan\n', encoding='utf-8'
            )

            result = scan_analysis_datasets(project_root)

            self.assertTrue(result['available'])
            self.assertEqual(len(result['datasets']), 1)
            dataset = result['datasets'][0]
            self.assertEqual(dataset['kind'], 'merge_potential')
            self.assertEqual(dataset['metadata']['episodes'], 20_000)
            self.assertEqual(dataset['metadata']['checkpoint'], 'final.pt')
            table = dataset['tables'][0]
            self.assertEqual(table['id'], 'lifecycle_by_level')
            self.assertEqual(table['rows'][0]['level'], 1)
            self.assertEqual(table['rows'][0]['fruits'], 100)
            self.assertIsNone(table['rows'][1]['eventual_merge_probability_resolved'])

    def test_cloud_instance_registration_is_parsed_only_from_current_section(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cloud.md'
            path.write_text(
                '# 云服务器\n\n## 当前可用实例\n\n'
                '| 实例 | SSH 连接 | 密码 |\n| --- | --- | --- |\n'
                '| 5号 | `ssh -p 30021 root@current.example` | `new-pass` |\n\n'
                '## 历史已弃用实例\n\n'
                '| 4号 | `ssh -p 20432 root@old.example` | `old-pass` |\n',
                encoding='utf-8',
            )

            config = _read_cloud_ssh_config(path)

            self.assertEqual(config.port, 30021)
            self.assertEqual(config.target, 'root@current.example')
            self.assertEqual(config.password, 'new-pass')

    def test_merge_potential_manifest_is_normalized_for_dashboard(self):
        with TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            run_dir = project_root / 'runs' / 'analysis' / 'pilot'
            (run_dir / 'analysis').mkdir(parents=True)
            (run_dir / 'analysis' / 'analysis_manifest.json').write_text(
                '{}', encoding='utf-8'
            )
            (run_dir / 'manifest.json').write_text(json.dumps({
                'purpose': 'merge_potential_t_merge_collection',
                'status': 'running',
                'created_at_utc': '2099-01-01T00:00:00+00:00',
                'updated_at_utc': '2099-01-01T00:00:10+00:00',
                'checkpoint': '/cloud/checkpoints/final.pt',
                'checkpoint_sha256': 'abc123',
                'simulator_config': {
                    'physics_fps': 30,
                    'drop_fast_forward': False,
                },
                'parameters': {
                    'episodes': 20_000,
                    'parallel_envs': 4096,
                    'max_drops': 0,
                },
                'completed_episodes': 5_000,
                'transitions': 2_000_000,
                'elapsed_seconds': 100.0,
                'env_steps_per_second': 20_000.0,
                'peak_cuda_allocated_bytes': 8 * 1024 ** 3,
                'flushed_rows': {
                    'snapshots': 123_456,
                    'merge_sources': 78_901,
                    'episodes': 5_000,
                },
            }), encoding='utf-8')

            result = scan_merge_potential_runs(project_root)

            self.assertTrue(result['available'])
            current = result['current']
            self.assertEqual(current['run_dir'], 'runs/analysis/pilot')
            self.assertEqual(current['progress_fraction'], 0.25)
            self.assertEqual(current['snapshot_rows'], 123_456)
            self.assertEqual(current['physics_fps'], 30)
            self.assertFalse(current['drop_fast_forward'])
            self.assertEqual(current['checkpoint_name'], 'final.pt')
            self.assertTrue(current['analysis_ready'])

    def test_merge_potential_scan_ignores_unrelated_or_invalid_manifests(self):
        with TemporaryDirectory() as temporary:
            analysis_root = Path(temporary) / 'runs' / 'analysis'
            unrelated = analysis_root / 'unrelated'
            broken = analysis_root / 'broken'
            unrelated.mkdir(parents=True)
            broken.mkdir(parents=True)
            (unrelated / 'manifest.json').write_text(json.dumps({
                'purpose': 'different_job',
            }), encoding='utf-8')
            (broken / 'manifest.json').write_text('{', encoding='utf-8')

            result = scan_merge_potential_runs(Path(temporary))

            self.assertFalse(result['available'])
            self.assertIsNone(result['current'])
            self.assertEqual(result['runs'], [])

    def test_merge_distance_training_manifest_and_history_are_normalized(self):
        with TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            run_dir = project_root / 'runs' / 'merge_distance' / 'pilot'
            run_dir.mkdir(parents=True)
            (run_dir / 'manifest.json').write_text(json.dumps({
                'purpose': 'merge_distance_predictor_training',
                'status': 'running',
                'phase': 'training',
                'created_at_utc': '2099-01-01T00:00:00+00:00',
                'updated_at_utc': '2099-01-01T00:01:00+00:00',
                'arguments': {'epochs': 12},
                'current_epoch': 2,
                'completed_epochs': 2,
                'progress_fraction': 2 / 12,
                'epoch_batch': 40,
                'train_loss': 1.25,
                'latest_validation': {
                    'nll': 1.1,
                    'lifecycle_weighted_nll': 1.05,
                    'exact_bin_accuracy': 0.42,
                    'adjacent_bin_accuracy': 0.73,
                },
                'parameter_count': 123_979,
            }), encoding='utf-8')
            (run_dir / 'metrics.jsonl').write_text(
                json.dumps({
                    'epoch': 1,
                    'train_loss': 1.5,
                    'validation': {
                        'nll': 1.3,
                        'lifecycle_weighted_nll': 1.2,
                        'exact_bin_accuracy': 0.3,
                        'adjacent_bin_accuracy': 0.6,
                    },
                }) + '\n' + json.dumps({
                    'epoch': 2,
                    'train_loss': 1.25,
                    'validation': {
                        'nll': 1.1,
                        'lifecycle_weighted_nll': 1.05,
                        'exact_bin_accuracy': 0.42,
                        'adjacent_bin_accuracy': 0.73,
                    },
                }) + '\n',
                encoding='utf-8',
            )

            result = scan_merge_distance_runs(project_root)

            self.assertTrue(result['available'])
            current = result['current']
            self.assertEqual(current['kind'], 'training')
            self.assertEqual(current['current_epoch'], 2)
            self.assertAlmostEqual(current['validation_nll'], 1.1)
            self.assertEqual(current['parameter_count'], 123_979)
            self.assertEqual(len(current['history']), 2)
            self.assertAlmostEqual(
                current['history'][-1]['validation_exact_bin_accuracy'], 0.42
            )

    def test_merge_distance_collection_and_labeling_share_one_status(self):
        with TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            run_dir = project_root / 'runs' / 'analysis' / 'pilot'
            predictor_dir = run_dir / 'predictor'
            predictor_dir.mkdir(parents=True)
            (run_dir / 'manifest.json').write_text(json.dumps({
                'purpose': 'merge_distance_predictor_collection',
                'status': 'complete',
                'created_at_utc': '2099-01-01T00:00:00+00:00',
                'updated_at_utc': '2099-01-01T00:01:00+00:00',
                'parameters': {'episodes': 100, 'parallel_envs': 32},
                'completed_episodes': 100,
                'table_rows': {'scenes': 12_000, 'merge_sources': 8_000},
            }), encoding='utf-8')
            (predictor_dir / 'dataset_manifest.json').write_text(json.dumps({
                'purpose': 'merge_distance_predictor_dataset',
                'status': 'labeling',
                'phase': 'labeling',
                'created_at_utc': '2099-01-01T00:02:00+00:00',
                'updated_at_utc': '2099-01-01T00:03:00+00:00',
                'source_scene_shards': 20,
                'completed_scene_shards': 8,
                'scene_rows': 4_800,
                'resolved_fruit_samples': 32_000,
            }), encoding='utf-8')

            result = scan_merge_distance_runs(project_root)

            self.assertEqual(len(result['runs']), 2)
            self.assertEqual(result['current']['kind'], 'labeling')
            self.assertAlmostEqual(result['current']['progress_fraction'], 0.4)
            collection = next(
                item for item in result['runs']
                if item['kind'] == 'collection'
            )
            self.assertEqual(collection['scene_rows'], 12_000)
            self.assertEqual(collection['progress_fraction'], 1.0)

    def test_scan_documents_returns_current_markdown_content(self):
        documents = scan_documents(PROJECT_ROOT)
        by_path = {document['path']: document for document in documents}

        self.assertIn('docs/README.md', by_path)
        self.assertIn('docs/model_evaluations/COMPARISON_MATRIX.md', by_path)
        self.assertTrue(by_path['docs/README.md']['content'].startswith('#'))
        self.assertEqual(
            by_path['docs/model_evaluations/COMPARISON_MATRIX.md']['category'],
            'evaluations',
        )
        self.assertTrue(all(document['search_text'] for document in documents))
        self.assertRegex(document_revision(PROJECT_ROOT), r'^\d+:\d+$')

    def test_scenario_lab_command_is_a_validated_argument_array(self):
        command, url = build_tool_command(
            'scenario_lab',
            {
                'device': 'cuda',
                'comparison': 'on',
                'comparison_preset': 'play_vs_training',
                'model_device': 'auto',
                'port': 8769,
                'reward_scale': 1.0,
                'checkpoint': '',
            },
        )

        self.assertEqual(
            command[1:3], ['tools/open_scenario_lab.py', '--host']
        )
        self.assertEqual(command[command.index('--host') + 1], '127.0.0.1')
        self.assertEqual(command[command.index('--device') + 1], 'cuda')
        self.assertIn('--comparison', command)
        self.assertEqual(
            command[command.index('--comparison-preset') + 1],
            'play_vs_training',
        )
        self.assertEqual(url, 'http://127.0.0.1:8769/')

    def test_tool_environment_includes_project_source_root(self):
        environment = _subprocess_environment()
        entries = environment['PYTHONPATH'].split(os.pathsep)
        self.assertEqual(Path(entries[0]).resolve(), (PROJECT_ROOT / 'src').resolve())
        self.assertEqual(environment['PYTHONIOENCODING'], 'utf-8')
        self.assertEqual(environment['PYTHONUTF8'], '1')

    def test_tool_command_rejects_unlisted_choices(self):
        with self.assertRaisesRegex(ValueError, 'device'):
            build_tool_command(
                'scenario_lab',
                {
                    'device': 'remote-shell',
                    'model_device': 'auto',
                    'port': 8769,
                    'reward_scale': 1.0,
                    'checkpoint': '',
                },
            )

    def test_portal_health_and_document_asset_endpoints(self):
        server = PortalServer(port=0)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f'{server.url}/api/health',
                headers={'Origin': 'http://127.0.0.1:3100'},
            )
            with urlopen(request, timeout=5) as response:
                health = json.loads(response.read().decode('utf-8'))
                self.assertEqual(
                    response.headers['Access-Control-Allow-Origin'],
                    'http://127.0.0.1:3100',
                )
            self.assertTrue(health['ok'])
            self.assertGreater(health['documents'], 0)

            with urlopen(
                f'{server.url}/api/documents/revision', timeout=5
            ) as response:
                revision = json.loads(response.read().decode('utf-8'))
            self.assertRegex(revision['revision'], r'^\d+:\d+$')

            with urlopen(
                f'{server.url}/api/file?path=docs/README.md', timeout=5
            ) as response:
                body = response.read().decode('utf-8')
            self.assertTrue(body.startswith('#'))
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
