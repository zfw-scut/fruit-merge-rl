"""启动本地知识门户、工具控制API和前端服务。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
CLOUD_SERVER_LOCAL = PROJECT_ROOT / 'docs' / 'CLOUD_SERVER_LOCAL.md'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daxigua.portal import PortalServer  # noqa: E402


@dataclass(frozen=True)
class CloudSshConfig:
    port: int
    target: str
    password: str


def _read_cloud_ssh_config(path=CLOUD_SERVER_LOCAL):
    """只从本机忽略文件的“当前可用实例”区段读取SSH配置。"""

    text = Path(path).read_text(encoding='utf-8')
    current = text.partition('## 当前可用实例')[2].partition('\n## ')[0]
    match = re.search(
        r'\|[^\n|]+\|\s*`ssh\s+-p\s+(\d+)\s+([^`\s]+)`\s*'
        r'\|\s*`([^`]+)`\s*\|',
        current,
    )
    if match is None:
        raise ValueError('当前可用实例中没有可解析的SSH登记')
    return CloudSshConfig(
        port=int(match.group(1)),
        target=match.group(2),
        password=match.group(3),
    )


def _prepare_askpass(runtime: Path, password: str):
    if os.name == 'nt':
        path = runtime / 'cloud_telemetry_askpass.cmd'
        path.write_text(f'@echo off\necho {password}\n', encoding='utf-8')
    else:
        path = runtime / 'cloud_telemetry_askpass.sh'
        path.write_text(f'#!/bin/sh\nprintf %s\\n {password!r}\n', encoding='utf-8')
        path.chmod(0o700)
    return path


def _start_cloud_tunnel(
        config: CloudSshConfig,
        *,
        local_port: int,
        askpass: Path,
        log_handle):
    ssh = shutil.which('ssh')
    if ssh is None:
        raise RuntimeError('未找到OpenSSH客户端，无法建立云端遥测转发')
    environment = os.environ.copy()
    environment.update({
        'SSH_ASKPASS': str(askpass),
        'SSH_ASKPASS_REQUIRE': 'force',
        'DISPLAY': 'codex',
    })
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    return subprocess.Popen(
        [
            ssh,
            '-N',
            '-L', f'127.0.0.1:{local_port}:127.0.0.1:8765',
            '-p', str(config.port),
            '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ServerAliveInterval=10',
            '-o', 'ServerAliveCountMax=2',
            '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=accept-new',
            config.target,
        ],
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
        creationflags=flags,
    )


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
    parser.add_argument(
        '--cloud-telemetry',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='自动读取本机实例登记并维护8765 SSH遥测隧道。',
    )
    args = parser.parse_args(argv)

    runtime = PROJECT_ROOT / 'runs' / 'portal_processes'
    runtime.mkdir(parents=True, exist_ok=True)
    tunnel_config = None
    tunnel = None
    tunnel_log = None
    askpass = None
    tunnel_retry_at = 0.0
    if args.cloud_telemetry and not _port_open(8765):
        try:
            tunnel_config = _read_cloud_ssh_config()
            askpass = _prepare_askpass(runtime, tunnel_config.password)
            tunnel_log = (runtime / 'cloud_telemetry_tunnel.log').open(
                'a', encoding='utf-8'
            )
            tunnel = _start_cloud_tunnel(
                tunnel_config,
                local_port=8765,
                askpass=askpass,
                log_handle=tunnel_log,
            )
            print(
                f'云端训练遥测：正在连接 {tunnel_config.target} '
                f'（本地127.0.0.1:8765）',
                flush=True,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f'云端训练遥测未自动连接：{error}', flush=True)
    elif args.cloud_telemetry:
        print('云端训练遥测：本地8765端口已有数据源', flush=True)

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
            if (
                    tunnel_config is not None
                    and askpass is not None
                    and tunnel_log is not None
                    and (tunnel is None or tunnel.poll() is not None)
                    and time.monotonic() >= tunnel_retry_at):
                tunnel = _start_cloud_tunnel(
                    tunnel_config,
                    local_port=8765,
                    askpass=askpass,
                    log_handle=tunnel_log,
                )
                tunnel_retry_at = time.monotonic() + 5.0
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('正在关闭项目门户……', flush=True)
    finally:
        portal.shutdown()
        if frontend is not None and frontend.poll() is None:
            frontend.terminate()
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=3)
            except subprocess.TimeoutExpired:
                tunnel.kill()
        if tunnel_log is not None:
            tunnel_log.close()
        if askpass is not None:
            try:
                askpass.unlink()
            except OSError:
                pass
        if frontend_log is not None:
            frontend_log.close()


if __name__ == '__main__':
    main()
