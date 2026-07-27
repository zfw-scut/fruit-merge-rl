"""云端轻量训练产物同步工具测试。"""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.sync_cloud_training_artifacts as sync_tool
from tools.sync_cloud_training_artifacts import (
    ArtifactSyncError,
    COMPLETE_REQUIRED_PATHS,
    PNG_IEND,
    PNG_SIGNATURE,
    build_remote_command,
    build_ssh_command,
    download_remote_archive,
    install_downloaded_archive,
    safe_extract_archive,
    validate_remote_run_dir,
)


def _complete_artifacts() -> dict[str, bytes]:
    """构造足以通过完整阶段校验的最小产物集合。"""

    png = PNG_SIGNATURE + b'test-image-payload' + PNG_IEND
    return {
        'config.json': json.dumps({'args': {'total_updates': 10}}).encode(),
        'metrics.csv': b'update_step,loss\n10,0.5\n',
        'episode_metrics.csv': b'episode_index,score\n1,123\n',
        'attribution_warmup.json': json.dumps({'phase': 'warmup'}).encode(),
        'attribution_shutdown.json': json.dumps({'completed': True}).encode(),
        'counterfactual_shutdown.json': json.dumps(
            {'completed': True}
        ).encode(),
        'plots/training_curves.png': png,
        'plots/reward_breakdown_curves.png': png,
        'plots/structure_learning_curves.png': png,
    }


def _write_archive(
        path: Path,
        files: dict[str, bytes],
        *,
        links: tuple[tuple[str, str], ...] = ()) -> None:
    """写入测试 tar；links 用于验证链接成员会被拒绝。"""

    with tarfile.open(path, mode='w:gz') as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        for name, target in links:
            info = tarfile.TarInfo(name=name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)


class CloudTrainingArtifactSyncTest(unittest.TestCase):
    """验证同步边界、安全解包、完整模式和本地幂等安装。"""

    def test_remote_command_quotes_path_and_ssh_argv_has_no_password(self):
        """带空格路径应安全引用，命令行中永远没有密码参数。"""

        remote_dir = '/root/training runs/run-10k'
        remote_command = build_remote_command(remote_dir)
        command = build_ssh_command(
            ssh_executable='ssh',
            host='cloud.example.test',
            port=22022,
            user='trainer',
            remote_run_dir=remote_dir,
            identity_file=Path('test-key'),
            known_hosts_file=Path('known-hosts'),
        )

        self.assertIn("'/root/training runs/run-10k'", remote_command)
        self.assertEqual(command[-1], remote_command)
        self.assertIn('trainer@cloud.example.test', command)
        self.assertNotIn('password', ' '.join(command).lower())

        for invalid in (
                'relative/run',
                '/root/run/../secret',
                '/root/run\nextra-command'):
            with self.assertRaises(ArtifactSyncError):
                validate_remote_run_dir(invalid)
        with self.assertRaises(ArtifactSyncError):
            build_ssh_command(
                ssh_executable='ssh',
                host='trusted.example@evil.example',
                port=22,
                user='trainer',
                remote_run_dir='/srv/run',
            )

    def test_safe_extract_rejects_traversal_backslash_and_links(self):
        """归档不能借助 POSIX/Windows 路径或链接逃出 staging。"""

        cases = (
            ({'../escape.json': b'{}'}, ()),
            ({'..\\escape.json': b'{}'}, ()),
            ({'C:escape.json': b'{}'}, ()),
            ({'failure_C:escape.json': b'{}'}, ()),
            (
                {
                    'failure_A.json': b'{}',
                    'failure_a.json': b'{}',
                },
                (),
            ),
            ({'config.json': b'{}'}, (('plots/curve.png', '../config.json'),)),
        )
        for files, links in cases:
            with self.subTest(files=tuple(files), links=links):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    archive_path = root / 'payload.tar.gz'
                    _write_archive(archive_path, files, links=links)
                    with self.assertRaises(ArtifactSyncError):
                        safe_extract_archive(archive_path, root / 'staging')
                    self.assertFalse((root.parent / 'escape.json').exists())

    def test_complete_install_preserves_unrelated_evidence_and_writes_manifest(self):
        """完整同步应原子更新基础包，同时保留已有 readiness 等旁路证据。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / 'payload.tar.gz'
            local_dir = root / 'evidence' / '10k'
            local_dir.mkdir(parents=True)
            (local_dir / 'readiness.json').write_text(
                '{"ready": true}',
                encoding='utf-8',
            )
            (local_dir / 'metrics.csv').write_text(
                'old,metrics\n',
                encoding='utf-8',
            )
            _write_archive(archive_path, _complete_artifacts())

            timestamp = dt.datetime(
                2026,
                7,
                28,
                12,
                0,
                tzinfo=dt.timezone.utc,
            )
            manifest = install_downloaded_archive(
                archive_path,
                local_dir,
                remote_run_dir='/srv/project/runs/calibration_10k',
                require_complete=True,
                synced_at=timestamp,
            )

            self.assertTrue((local_dir / 'readiness.json').is_file())
            self.assertEqual(
                (local_dir / 'metrics.csv').read_bytes(),
                _complete_artifacts()['metrics.csv'],
            )
            for relative_path in COMPLETE_REQUIRED_PATHS:
                self.assertTrue((local_dir / relative_path).is_file())

            self.assertEqual(manifest['mode'], 'complete')
            self.assertEqual(manifest['missing_optional'], [])
            self.assertFalse(
                manifest['selection']['checkpoint_and_replay_included']
            )
            manifest_text = (local_dir / 'sync_manifest.json').read_text(
                encoding='utf-8',
            )
            self.assertNotIn('password', manifest_text.lower())
            self.assertNotIn('checkpoint.pt', manifest_text)

            # 第二次安装相同归档应得到相同文件哈希，不删除无关证据。
            second = install_downloaded_archive(
                archive_path,
                local_dir,
                remote_run_dir='/srv/project/runs/calibration_10k',
                require_complete=True,
                synced_at=timestamp,
            )
            self.assertEqual(manifest['files'], second['files'])
            self.assertTrue((local_dir / 'readiness.json').is_file())

    def test_missing_complete_artifact_does_not_overwrite_existing_files(self):
        """严格模式校验失败必须发生在目标目录被修改之前。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / 'incomplete.tar.gz'
            local_dir = root / 'evidence'
            local_dir.mkdir()
            original_metrics = b'update_step,loss\n99,9.9\n'
            (local_dir / 'metrics.csv').write_bytes(original_metrics)

            incomplete = _complete_artifacts()
            del incomplete['plots/reward_breakdown_curves.png']
            _write_archive(archive_path, incomplete)

            with self.assertRaisesRegex(
                    ArtifactSyncError,
                    'reward_breakdown_curves'):
                install_downloaded_archive(
                    archive_path,
                    local_dir,
                    remote_run_dir='/srv/project/runs/incomplete',
                    require_complete=True,
                )
            self.assertEqual(
                (local_dir / 'metrics.csv').read_bytes(),
                original_metrics,
            )
            self.assertFalse((local_dir / 'sync_manifest.json').exists())

    def test_archive_rejects_local_sidecars_and_checkpoint(self):
        """远端不能覆盖本地 readiness，也不能夹带大型训练状态。"""

        for untrusted_path in ('readiness.json', 'checkpoints/latest.pt'):
            with self.subTest(untrusted_path=untrusted_path):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    archive_path = root / 'payload.tar.gz'
                    files = _complete_artifacts()
                    files[untrusted_path] = b'not-trusted'
                    _write_archive(archive_path, files)

                    with self.assertRaisesRegex(
                            ArtifactSyncError,
                            '白名单外产物'):
                        safe_extract_archive(
                            archive_path,
                            root / 'staging',
                        )

    def test_prepare_failure_keeps_old_directory_and_manifest_together(self):
        """第 N 个本地文件写入失败也不能留下新旧混合快照。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / 'payload.tar.gz'
            local_dir = root / 'evidence'
            local_dir.mkdir()
            original_metrics = b'update_step,loss\n3,3.0\n'
            original_manifest = '{"generation": "old"}\n'
            (local_dir / 'metrics.csv').write_bytes(original_metrics)
            (local_dir / 'sync_manifest.json').write_text(
                original_manifest,
                encoding='utf-8',
            )
            _write_archive(archive_path, _complete_artifacts())

            original_copy = sync_tool._copy_staged_file
            call_count = 0

            def flaky_copy(source, replacement_root, relative_path):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError('simulated local write failure')
                return original_copy(source, replacement_root, relative_path)

            with mock.patch.object(
                    sync_tool,
                    '_copy_staged_file',
                    side_effect=flaky_copy):
                with self.assertRaisesRegex(
                        ArtifactSyncError,
                        '事务快照'):
                    install_downloaded_archive(
                        archive_path,
                        local_dir,
                        remote_run_dir='/srv/project/runs/calibration_10k',
                        require_complete=True,
                    )

            self.assertEqual(
                (local_dir / 'metrics.csv').read_bytes(),
                original_metrics,
            )
            self.assertEqual(
                (local_dir / 'sync_manifest.json').read_text(encoding='utf-8'),
                original_manifest,
            )
            self.assertFalse((local_dir / 'config.json').exists())

    def test_complete_mode_rejects_run_before_configured_update_target(self):
        """曲线已生成也不能让中途 run 冒充阶段结束的完整包。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / 'payload.tar.gz'
            files = _complete_artifacts()
            files['metrics.csv'] = b'update_step,loss\n9,0.5\n'
            _write_archive(archive_path, files)

            with self.assertRaisesRegex(
                    ArtifactSyncError,
                    'total_updates=10'):
                install_downloaded_archive(
                    archive_path,
                    root / 'evidence',
                    remote_run_dir='/srv/project/runs/calibration_10k',
                    require_complete=True,
                )

    def test_complete_mode_uses_latest_resume_config_and_rejects_failure(self):
        """同目录扩展训练应服从最新恢复配置，活动失败也不能标完整。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / 'resumed.tar.gz'
            files = _complete_artifacts()
            files['resume_config_20260728_120000_000000.json'] = json.dumps(
                {'args': {'total_updates': 20}}
            ).encode()
            _write_archive(archive_path, files)

            with self.assertRaisesRegex(ArtifactSyncError, 'total_updates=20'):
                install_downloaded_archive(
                    archive_path,
                    root / 'evidence',
                    remote_run_dir='/srv/project/runs/resumed',
                    require_complete=True,
                )

            files['metrics.csv'] = b'update_step,loss\n20,0.4\n'
            files['failure_latest.json'] = json.dumps(
                {'stage': 'shutdown'}
            ).encode()
            _write_archive(archive_path, files)
            with self.assertRaisesRegex(ArtifactSyncError, 'failure_latest'):
                install_downloaded_archive(
                    archive_path,
                    root / 'evidence',
                    remote_run_dir='/srv/project/runs/resumed',
                    require_complete=True,
                )

    def test_late_truncated_csv_does_not_replace_old_evidence(self):
        """首行正常但后段截断的 CSV 必须在目录事务开始前被拒绝。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / 'truncated.tar.gz'
            local_dir = root / 'evidence'
            local_dir.mkdir()
            original = b'episode_index,score\n1,999\n'
            (local_dir / 'episode_metrics.csv').write_bytes(original)
            files = _complete_artifacts()
            files['episode_metrics.csv'] = (
                b'episode_index,score\n'
                b'1,123\n'
                b'2,"unterminated\n'
            )
            _write_archive(archive_path, files)

            with self.assertRaisesRegex(ArtifactSyncError, 'CSV'):
                install_downloaded_archive(
                    archive_path,
                    local_dir,
                    remote_run_dir='/srv/project/runs/truncated',
                    require_complete=True,
                )
            self.assertEqual(
                (local_dir / 'episode_metrics.csv').read_bytes(),
                original,
            )

    def test_incremental_sync_prunes_stale_managed_optional_files(self):
        """远端本轮缺失的旧曲线/失败指针不能继续混在新 manifest 旁。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / 'incremental.tar.gz'
            local_dir = root / 'evidence'
            plots_dir = local_dir / 'plots'
            plots_dir.mkdir(parents=True)
            (plots_dir / 'training_curves.png').write_bytes(
                PNG_SIGNATURE + b'old' + PNG_IEND
            )
            (local_dir / 'failure_latest.json').write_text(
                '{"old": true}',
                encoding='utf-8',
            )
            (local_dir / 'readiness.json').write_text(
                '{"ready": true}',
                encoding='utf-8',
            )
            _write_archive(
                archive_path,
                {
                    'config.json': json.dumps(
                        {'args': {'total_updates': 10}}
                    ).encode(),
                    'metrics.csv': b'update_step,loss\n1,0.5\n',
                },
            )

            manifest = install_downloaded_archive(
                archive_path,
                local_dir,
                remote_run_dir='/srv/project/runs/active',
                require_complete=False,
            )

            self.assertFalse(
                (plots_dir / 'training_curves.png').exists()
            )
            self.assertFalse((local_dir / 'failure_latest.json').exists())
            self.assertTrue((local_dir / 'readiness.json').is_file())
            self.assertIn(
                'plots/training_curves.png',
                manifest['missing_optional'],
            )

    def test_download_enforces_output_limit_and_total_timeout(self):
        """异常 SSH 子进程不能无限占用磁盘或永久阻塞同步。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / 'payload.tar.gz'
            with mock.patch.object(
                    sync_tool,
                    'MAX_ARCHIVE_BYTES',
                    32):
                with self.assertRaisesRegex(
                        ArtifactSyncError,
                        '安全上限'):
                    download_remote_archive(
                        [
                            sys.executable,
                            '-c',
                            'import sys; '
                            'sys.stdout.buffer.write(b"x" * 64)',
                        ],
                        archive_path,
                        timeout_seconds=5,
                    )
            self.assertFalse(archive_path.exists())

            with self.assertRaisesRegex(ArtifactSyncError, '超时'):
                download_remote_archive(
                    [
                        sys.executable,
                        '-c',
                        'import time; time.sleep(5)',
                    ],
                    archive_path,
                    timeout_seconds=0.1,
                )
            self.assertFalse(archive_path.exists())


if __name__ == '__main__':
    unittest.main()
