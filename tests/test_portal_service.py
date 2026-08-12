from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Thread
import unittest
from urllib.request import Request, urlopen

from daxigua.portal.service import (
    PortalServer,
    _subprocess_environment,
    build_tool_command,
    document_revision,
    scan_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PortalServiceTests(unittest.TestCase):
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
