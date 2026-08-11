"""启动本地知识门户、工具控制API和前端服务。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daxigua.portal import PortalServer  # noqa: E402


def _port_open(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.25)
        return connection.connect_ex(('127.0.0.1', port)) == 0


def _wait_port(port: int, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return
        time.sleep(0.2)
    raise TimeoutError(f'端口 {port} 未在 {timeout:g} 秒内就绪')


def _frontend_needs_build(portal_root: Path) -> bool:
    output = portal_root / 'dist' / 'server' / 'index.js'
    if not output.exists():
        return True
    source_paths = [
        *portal_root.joinpath('app').rglob('*.tsx'),
        *portal_root.joinpath('app').rglob('*.ts'),
        *portal_root.joinpath('app').rglob('*.css'),
        portal_root / 'package.json',
        portal_root / 'package-lock.json',
        portal_root / 'vite.config.ts',
    ]
    latest_source = max(
        (path.stat().st_mtime_ns for path in source_paths if path.exists()),
        default=0,
    )
    return latest_source > output.stat().st_mtime_ns


def main(argv=None):
    parser = argparse.ArgumentParser(description='打开合成大西瓜项目知识与工具门户。')
    parser.add_argument('--api-port', type=int, default=4312)
    parser.add_argument('--web-port', type=int, default=3000)
    parser.add_argument('--backend-only', action='store_true')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args(argv)

    portal = PortalServer(port=args.api_port)
    api_thread = threading.Thread(target=portal.serve_forever, daemon=True)
    api_thread.start()
    print(f'门户控制API：{portal.url}', flush=True)

    frontend = None
    frontend_log = None
    web_url = f'http://127.0.0.1:{args.web_port}/'
    try:
        if not args.backend_only and not _port_open(args.web_port):
            npx = shutil.which('npx.cmd' if os.name == 'nt' else 'npx')
            if npx is None:
                raise RuntimeError('未找到Node.js npx，无法启动门户前端')
            portal_root = PROJECT_ROOT / 'portal'
            if not (portal_root / 'node_modules').exists():
                raise RuntimeError('门户依赖尚未安装，请先在portal目录执行npm install')
            runtime = PROJECT_ROOT / 'runs' / 'portal_processes'
            runtime.mkdir(parents=True, exist_ok=True)
            frontend_log = (runtime / 'frontend.log').open('a', encoding='utf-8')
            if _frontend_needs_build(portal_root):
                print('首次启动或前端已修改，正在构建门户……', flush=True)
                subprocess.run(
                    [npx, 'vinext', 'build'], cwd=portal_root,
                    stdout=frontend_log, stderr=subprocess.STDOUT, check=True,
                )
                frontend_log.flush()
            command = [npx, 'vinext', 'start', '--hostname', '127.0.0.1',
                       '--port', str(args.web_port)]
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            frontend = subprocess.Popen(
                command, cwd=portal_root, stdout=frontend_log,
                stderr=subprocess.STDOUT, creationflags=flags,
            )
            _wait_port(args.web_port)
        if not args.backend_only:
            print(f'项目知识门户：{web_url}', flush=True)
            if not args.no_open:
                webbrowser.open(web_url)
        while True:
            if frontend is not None and frontend.poll() is not None:
                raise RuntimeError(f'门户前端意外退出：{frontend.returncode}')
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('正在关闭项目门户……', flush=True)
    finally:
        portal.shutdown()
        if frontend is not None and frontend.poll() is None:
            frontend.terminate()
        if frontend_log is not None:
            frontend_log.close()


if __name__ == '__main__':
    main()
