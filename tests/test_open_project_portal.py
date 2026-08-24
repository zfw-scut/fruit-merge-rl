from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from tools.open_project_portal import (
    _frontend_build_revision,
    _read_frontend_pid,
    _remove_stale_frontend,
    _replace_registered_portal,
    _start_frontend,
    _write_frontend_pid,
    main,
)


class ProjectPortalLauncherTests(unittest.TestCase):
    def test_frontend_build_revision_changes_with_output(self):
        with TemporaryDirectory() as temporary:
            portal_root = Path(temporary)
            output = portal_root / 'dist' / 'server' / 'index.js'
            output.parent.mkdir(parents=True)
            output.write_text('old', encoding='utf-8')
            old_revision = _frontend_build_revision(portal_root)
            output.write_text('new-build', encoding='utf-8')
            self.assertNotEqual(
                _frontend_build_revision(portal_root),
                old_revision,
            )

    def test_frontend_pid_state_round_trip(self):
        with TemporaryDirectory() as temporary:
            state_path = Path(temporary) / 'frontend.json'
            _write_frontend_pid(state_path, pid=1234, port=3000)
            self.assertEqual(_read_frontend_pid(state_path), 1234)

    def test_unregistered_occupied_port_is_not_terminated(self):
        with TemporaryDirectory() as temporary:
            state_path = Path(temporary) / 'frontend.json'
            with (
                patch(
                    'tools.open_project_portal._port_open',
                    return_value=True,
                ),
                self.assertRaisesRegex(RuntimeError, '未登记进程'),
            ):
                _remove_stale_frontend(state_path=state_path, port=3000)

    def test_registered_stale_frontend_is_removed(self):
        with TemporaryDirectory() as temporary:
            state_path = Path(temporary) / 'frontend.json'
            _write_frontend_pid(state_path, pid=1234, port=3000)
            with (
                patch(
                    'tools.open_project_portal._port_open',
                    return_value=True,
                ),
                patch(
                    'tools.open_project_portal._terminate_process_tree'
                ) as terminate,
                patch('tools.open_project_portal._wait_port_closed'),
            ):
                _remove_stale_frontend(state_path=state_path, port=3000)
            terminate.assert_called_once_with(1234)
            self.assertFalse(state_path.exists())

    def test_registered_portal_is_replaced_but_unregistered_port_is_preserved(self):
        with TemporaryDirectory() as temporary:
            state_path = Path(temporary) / 'launcher.json'
            _write_frontend_pid(state_path, pid=4321, port=4312)
            with (
                patch('tools.open_project_portal._port_open', return_value=True),
                patch('tools.open_project_portal._terminate_process_tree') as terminate,
                patch('tools.open_project_portal._wait_port_closed'),
            ):
                replaced = _replace_registered_portal(
                    state_path=state_path, port=4312
                )
            self.assertTrue(replaced)
            terminate.assert_called_once_with(4321)
            self.assertFalse(state_path.exists())

            with (
                patch('tools.open_project_portal._port_open', return_value=True),
                patch('tools.open_project_portal._terminate_process_tree') as terminate,
            ):
                replaced = _replace_registered_portal(
                    state_path=state_path, port=4312
                )
            self.assertFalse(replaced)
            terminate.assert_not_called()

    def test_frontend_starts_direct_node_process(self):
        with TemporaryDirectory() as temporary:
            portal_root = Path(temporary)
            state_path = portal_root / 'frontend.json'
            process = MagicMock(pid=4321)
            with (
                patch(
                    'tools.open_project_portal.subprocess.Popen',
                    return_value=process,
                ) as popen,
                patch('tools.open_project_portal._wait_port'),
            ):
                result = _start_frontend(
                    node='node.exe',
                    portal_root=portal_root,
                    port=3000,
                    log_handle=None,
                    state_path=state_path,
                )
            self.assertIs(result, process)
            command = popen.call_args.args[0]
            self.assertEqual(command[0], 'node.exe')
            self.assertEqual(
                Path(command[1]),
                portal_root / 'node_modules' / 'vinext' / 'dist' / 'cli.js',
            )
            self.assertEqual(_read_frontend_pid(state_path), 4321)

    def test_existing_api_prevents_duplicate_backend(self):
        with (
            patch('tools.open_project_portal._port_open', return_value=True),
            patch(
                'tools.open_project_portal._replace_registered_portal',
                return_value=False,
            ),
            patch('tools.open_project_portal.PortalServer') as portal_server,
        ):
            main(['--no-open', '--no-cloud-telemetry'])
        portal_server.assert_not_called()


if __name__ == '__main__':
    unittest.main()
