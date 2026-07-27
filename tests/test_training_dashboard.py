"""只读训练面板的解析、实时心跳和 HTTP 安全边界测试。"""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import tools.training_dashboard as dashboard


class DashboardFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_dir = root / 'runs' / 'run_10k'
        self.monitor_dir = root / 'runs' / 'resource_monitor' / 'monitor_10k'
        self.control_dir = root / 'runs' / 'stage_control' / 'control_10k'
        self.static_dir = root / 'static'
        for path in (
            self.run_dir / 'plots',
            self.monitor_dir,
            self.control_dir,
            self.static_dir,
        ):
            path.mkdir(parents=True)

        (self.run_dir / 'config.json').write_text(
            json.dumps(
                {
                    'args': {'total_updates': 1000},
                    'argv': ['--password', 'never-expose-this'],
                }
            ),
            encoding='utf-8',
        )
        # 最后一行故意没有换行且字段不完整，模拟训练进程写到一半。
        (self.run_dir / 'metrics.csv').write_text(
            'update_step,env_steps,epsilon,loss,td_loss,structural_loss,'
            'buffer_size,updates_per_second,env_steps_per_second,'
            'actor_inference_requests,actor_inference_batches,'
            'actor_inference_mean_batch_size,actor_inference_max_batch,'
            'counterfactual_enabled,counterfactual_actual_token_ratio,'
            'counterfactual_hard_budget_respected,causal_replay_size,'
            'best_eval_score\n'
            '100,2112,0.8,2.5,2.2,0.3,2000,2.0,40,120,30,4,8,1,0.08,1,50,321\n'
            '200,3712,0.7,broken',
            encoding='utf-8',
        )
        (self.run_dir / 'episode_metrics.csv').write_text(
            'episode_index,update_step,score\n1,90,100\n2,100,140\n',
            encoding='utf-8',
        )
        (self.monitor_dir / 'system_metrics.csv').write_text(
            'timestamp,elapsed_sec,sample,cpu_count,target_cpu_percent,'
            'target_process_count,mem_total_mb,mem_used_mb,mem_available_mb,'
            'target_rss_mb,gpu_count\n'
            '2026-07-28T00:00:00Z,60,1,208,1250,1,790000,500000,290000,20000,1\n',
            encoding='utf-8',
        )
        (self.monitor_dir / 'gpu_metrics.csv').write_text(
            'timestamp,elapsed_sec,sample,index,util_gpu_percent,memory_used_mb,'
            'memory_total_mb,temperature_c,power_draw_w,power_limit_w\n'
            '2026-07-28T00:00:00Z,60,1,0,66,29000,32607,61,420,575\n',
            encoding='utf-8',
        )
        gib = 1024 ** 3
        (self.monitor_dir / 'cgroup_metrics.csv').write_text(
            'timestamp,elapsed_sec,sample,memory_max_bytes,'
            'memory_working_set_bytes,memory_effective_available_bytes\n'
            f'2026-07-28T00:00:00Z,60,1,{90 * gib},{30 * gib},{60 * gib}\n',
            encoding='utf-8',
        )
        (self.monitor_dir / 'events.jsonl').write_text(
            json.dumps(
                {
                    'timestamp': '2026-07-28T00:00:00Z',
                    'level': 'warning',
                    'event': 'gpu_memory_high',
                    'details': {
                        'gpu_memory_used_mb': 31000,
                        'password': 'never-expose-this',
                        'error': 'credential-like diagnostic',
                    },
                }
            )
            + '\n',
            encoding='utf-8',
        )
        (self.control_dir / 'train_wrapper.log').write_text(
            '[progress 进度] | phase=train 阶段=train | 120/1000 | 12.0% | '
            'env_steps=2432 投放=2432 | buffer=2300 经验池=2300 | eps=0.750 | '
            'speed=42.50 env_steps/s 投放/秒=42.50 | '
            'eta=7.3min 预计剩余=7.3分钟 | loss=2.1000\n',
            encoding='utf-8',
        )
        (self.run_dir / 'plots' / 'training_curves.png').write_bytes(
            b'\x89PNG\r\n\x1a\nfixture'
        )
        (self.run_dir / 'plots' / 'private.png').write_bytes(b'private')
        (self.static_dir / 'index.html').write_text('<h1>Dashboard</h1>', encoding='utf-8')
        (self.static_dir / 'app.js').write_text('void 0;', encoding='utf-8')
        (self.static_dir / 'styles.css').write_text('body{}', encoding='utf-8')

    def builder(self) -> dashboard.DashboardStateBuilder:
        return dashboard.DashboardStateBuilder(
            project_root=self.root,
            run_dir=self.run_dir,
            monitor_dir=self.monitor_dir,
            control_dir=self.control_dir,
            history_limit=50,
        )


class TrainingDashboardParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fixture = DashboardFixture(Path(self.temp_dir.name))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_state_uses_heartbeat_cgroup_and_ignores_partial_csv_tail(self):
        with mock.patch.object(
            dashboard,
            'read_cgroup_cpu_capacity',
            return_value=25.0,
        ):
            state = self.fixture.builder().build()

        self.assertEqual(state['status'], 'running')
        self.assertEqual(state['progress']['current_update'], 120)
        self.assertEqual(state['progress']['env_steps'], 2432)
        self.assertEqual(state['progress']['target_updates'], 1000)
        self.assertAlmostEqual(state['progress']['percent'], 12.0)
        self.assertAlmostEqual(state['progress']['eta_seconds'], 438.0)
        self.assertEqual(state['progress']['eta_source'], 'heartbeat')
        self.assertEqual(state['training']['loss'], 2.1)
        self.assertEqual(state['training']['buffer_size'], 2300)
        self.assertEqual(len(state['series']['loss']), 1)
        self.assertEqual(state['series']['loss'][0]['x'], 100.0)

        resources = state['resources']
        self.assertEqual(resources['memory_source'], 'cgroup_working_set')
        self.assertEqual(resources['cpu_count'], 25.0)
        self.assertAlmostEqual(resources['cpu_util_percent'], 50.0)
        self.assertAlmostEqual(state['series']['cpu_util_percent'][0]['y'], 50.0)
        self.assertAlmostEqual(resources['memory_total_mb'], 90 * 1024)
        self.assertAlmostEqual(resources['memory_used_mb'], 30 * 1024)
        self.assertEqual(resources['gpu_util_percent'], 66.0)
        self.assertEqual(state['actor']['requests'], 120)
        self.assertAlmostEqual(state['causal']['actual_token_ratio'], 0.08)
        self.assertEqual(state['plots'][0]['url'], '/plots/training_curves.png')

        serialized = json.dumps(state)
        self.assertNotIn('never-expose-this', serialized)
        self.assertNotIn(str(self.fixture.run_dir), serialized)
        self.assertEqual(
            state['alerts'][0]['details'],
            {'gpu_memory_used_mb': 31000},
        )

    def test_warmup_heartbeat_does_not_replace_training_target(self):
        (self.fixture.control_dir / 'train_wrapper.log').write_text(
            '[progress 进度] | phase=warmup 阶段=warmup | 2500/2500 | 100.0% | '
            'env_steps=2500 投放=2500 | buffer=2500 经验池=2500 | eps=1.000 | '
            'speed=40.00 env_steps/s 投放/秒=40.00 | '
            'eta=0.0min 预计剩余=0.0分钟 | loss=0.0000\n',
            encoding='utf-8',
        )

        state = self.fixture.builder().build()

        self.assertEqual(state['progress']['phase'], 'warmup')
        self.assertEqual(state['progress']['current_update'], 100)
        self.assertEqual(state['progress']['target_updates'], 1000)
        self.assertFalse(state['progress']['is_complete'])
        self.assertEqual(state['status'], 'running')

    def test_stale_monitor_is_not_reported_as_running(self):
        old = time.time() - dashboard.DATA_STALE_AFTER_SECONDS - 10
        for path in (
            self.fixture.run_dir / 'metrics.csv',
            self.fixture.monitor_dir / 'system_metrics.csv',
            self.fixture.control_dir / 'train_wrapper.log',
        ):
            os.utime(path, (old, old))

        state = self.fixture.builder().build()

        self.assertEqual(state['status'], 'stale')
        self.assertFalse(state['freshness']['data_fresh'])
        self.assertFalse(state['progress']['process_active'])
        self.assertTrue(state['progress']['process_active_observed'])

    def test_episode_total_is_not_limited_to_history_window(self):
        rows = ['episode_index,update_step,score']
        rows.extend(f'{index},{index},100' for index in range(1, 76))
        (self.fixture.run_dir / 'episode_metrics.csv').write_text(
            '\n'.join(rows) + '\n',
            encoding='utf-8',
        )

        state = self.fixture.builder().build()

        self.assertEqual(state['training']['episode_count'], 75)
        self.assertEqual(state['training']['episode_history_count'], 50)

    def test_train_exit_sets_completed_or_failed_status(self):
        (self.fixture.control_dir / 'train.exit').write_text('0\n', encoding='utf-8')
        self.assertEqual(self.fixture.builder().build()['status'], 'completed')

        (self.fixture.control_dir / 'train.exit').write_text('7\n', encoding='utf-8')
        state = self.fixture.builder().build()
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['progress']['exit_code'], 7)
        self.assertTrue(
            any(alert['event'] == 'training_exit_nonzero' for alert in state['alerts'])
        )

    def test_auto_discovery_supports_current_stage_control_root(self):
        run_dir, monitor_dir, control_dir = dashboard.discover_directories(
            self.fixture.root,
            None,
            None,
            None,
        )
        self.assertEqual(run_dir, self.fixture.run_dir.resolve())
        self.assertEqual(monitor_dir, self.fixture.monitor_dir.resolve())
        self.assertEqual(control_dir, self.fixture.control_dir.resolve())


class TrainingDashboardHTTPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fixture = DashboardFixture(Path(self.temp_dir.name))
        self.server = dashboard.DashboardServer(
            ('127.0.0.1', 0),
            self.fixture.builder(),
            self.fixture.static_dir,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            '127.0.0.1',
            self.server.server_address[1],
            timeout=3,
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp_dir.cleanup()

    def request(self, path: str) -> tuple[int, dict[str, str], bytes]:
        self.connection.request('GET', path)
        response = self.connection.getresponse()
        body = response.read()
        return response.status, dict(response.getheaders()), body

    def test_state_health_static_and_allowed_plot(self):
        status, headers, body = self.request('/api/state')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(json.loads(body)['progress']['current_update'], 120)

        status, _, body = self.request('/api/health')
        self.assertEqual(status, 200)
        health = json.loads(body)
        self.assertTrue(health['ok'])
        self.assertTrue(health['service_ok'])
        self.assertTrue(health['data_fresh'])

        status, _, body = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn(b'Dashboard', body)

        status, headers, body = self.request('/styles.css')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'text/css')
        self.assertIn(b'body', body)

        status, headers, body = self.request('/app.js')
        self.assertEqual(status, 200)
        self.assertIn('javascript', headers['Content-Type'])
        self.assertIn(b'void 0', body)

        status, headers, body = self.request('/plots/training_curves.png')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'image/png')
        self.assertTrue(body.startswith(b'\x89PNG'))

    def test_plot_whitelist_and_path_traversal_are_rejected(self):
        for path in (
            '/plots/private.png',
            '/plots/%2e%2e/config.json',
            '/plots/..%2fconfig.json',
            '/api/unknown',
        ):
            status, _, _ = self.request(path)
            self.assertEqual(status, 404, path)

    def test_query_string_is_not_written_to_access_log(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            status, _, _ = self.request('/?password=never-log-this')

        self.assertEqual(status, 200)
        self.assertNotIn('never-log-this', output.getvalue())
        self.assertIn('GET / 200', output.getvalue())

    def test_plot_parent_symlink_is_rejected(self):
        plot_root = self.fixture.run_dir / 'plots'
        for child in plot_root.iterdir():
            child.unlink()
        plot_root.rmdir()
        external = self.fixture.root / 'external-plots'
        external.mkdir()
        (external / 'training_curves.png').write_bytes(
            b'\x89PNG\r\n\x1a\noutside'
        )
        try:
            plot_root.symlink_to(external, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f'当前平台不能创建目录符号链接：{exc}')

        status, _, _ = self.request('/plots/training_curves.png')

        self.assertEqual(status, 404)


if __name__ == '__main__':
    unittest.main()
