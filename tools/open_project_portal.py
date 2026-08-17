"""启动本地知识门户、工具控制API和前端服务。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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


def _wait_port_closed(port: int, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_open(port):
            return
        time.sleep(0.2)
    raise TimeoutError(f'端口 {port} 未在 {timeout:g} 秒内释放')


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


def _frontend_build_revision(portal_root: Path) -> str | None:
    """返回当前生产构建的轻量身份。

    运行中的 vinext 不会因 dist 被替换而自动热更新，因此启动器
    必须比较“启动时构建”与“磁盘当前构建”。
    """

    output = portal_root / 'dist' / 'server' / 'index.js'
    if not output.exists():
        return None
    stat = output.stat()
    return f'{stat.st_mtime_ns}:{stat.st_size}'


def _read_frontend_pid(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        pid = int(payload['pid'])
        return pid if pid > 0 else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_frontend_pid(path: Path, *, pid: int, port: int):
    path.write_text(
        json.dumps({'pid': int(pid), 'port': int(port)}, ensure_ascii=False),
        encoding='utf-8',
    )


def _process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _terminate_process_tree(pid: int):
    """终止启动器自己登记的前端进程树。"""

    if not _process_alive(pid):
        return
    if os.name == 'nt':
        subprocess.run(
            ['taskkill', '/PID', str(pid), '/T', '/F'],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    os.kill(pid, 15)


def _remove_stale_frontend(*, state_path: Path, port: int):
    """清理异常退出的旧启动器遗留的前端。"""

    if not _port_open(port):
        state_path.unlink(missing_ok=True)
        return
    pid = _read_frontend_pid(state_path)
    if pid is None:
        raise RuntimeError(
            f'端口 {port} 已被未登记进程占用，为避免误停其他服务未自动终止'
        )
    _terminate_process_tree(pid)
    _wait_port_closed(port)
    state_path.unlink(missing_ok=True)


def _start_frontend(
        *, node: str, portal_root: Path, port: int, log_handle,
        state_path: Path) -> subprocess.Popen:
    cli = portal_root / 'node_modules' / 'vinext' / 'dist' / 'cli.js'
    command = [
        node, str(cli), 'start', '--hostname', '127.0.0.1',
        '--port', str(port),
    ]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    process = subprocess.Popen(
        command,
        cwd=portal_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=flags,
        start_new_session=os.name != 'nt',
    )
    _write_frontend_pid(state_path, pid=process.pid, port=port)
    try:
        _wait_port(port)
    except Exception:
        _terminate_process_tree(process.pid)
        state_path.unlink(missing_ok=True)
        raise
    return process


def _stop_frontend(
        process: subprocess.Popen | None,
        *, state_path: Path,
        port: int,
        raise_on_timeout: bool = True) -> bool:
    if process is not None and process.poll() is None:
        _terminate_process_tree(process.pid)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    if _port_open(port):
        try:
            _wait_port_closed(port)
        except TimeoutError:
            if raise_on_timeout:
                raise
            return False
    state_path.unlink(missing_ok=True)
    return True


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
    web_url = f'http://127.0.0.1:{args.web_port}/'
    if _port_open(args.api_port):
        print(
            f'项目门户已在运行（API 127.0.0.1:{args.api_port}），'
            '不再启动重复后端。',
            flush=True,
        )
        if not args.backend_only and not args.no_open:
            webbrowser.open(web_url)
        return

    frontend_state = runtime / 'frontend.json'
    if not args.backend_only:
        _remove_stale_frontend(
            state_path=frontend_state,
            port=args.web_port,
        )

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
    npx = None
    node = None
    portal_root = PROJECT_ROOT / 'portal'
    served_build_revision = None
    frontend_check_at = 0.0
    try:
        if not args.backend_only:
            npx = shutil.which('npx.cmd' if os.name == 'nt' else 'npx')
            if npx is None:
                raise RuntimeError('未找到Node.js npx，无法启动门户前端')
            node = shutil.which('node.exe' if os.name == 'nt' else 'node')
            if node is None:
                raise RuntimeError('未找到Node.js，无法启动门户前端')
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
            served_build_revision = _frontend_build_revision(portal_root)
            frontend = _start_frontend(
                node=node,
                portal_root=portal_root,
                port=args.web_port,
                log_handle=frontend_log,
                state_path=frontend_state,
            )
        if not args.backend_only:
            print(f'项目知识门户：{web_url}', flush=True)
            if not args.no_open:
                webbrowser.open(web_url)
        while True:
            if frontend is not None and frontend.poll() is not None:
                print(
                    f'门户前端已退出（{frontend.returncode}），正在自动重启。',
                    flush=True,
                )
                frontend = _start_frontend(
                    node=node,
                    portal_root=portal_root,
                    port=args.web_port,
                    log_handle=frontend_log,
                    state_path=frontend_state,
                )
                served_build_revision = _frontend_build_revision(portal_root)
            if (
                    frontend is not None
                    and time.monotonic() >= frontend_check_at):
                frontend_check_at = time.monotonic() + 1.0
                try:
                    if _frontend_needs_build(portal_root):
                        print(
                            '检测到门户前端已修改，正在后台构建新版本……',
                            flush=True,
                        )
                        subprocess.run(
                            [npx, 'vinext', 'build'],
                            cwd=portal_root,
                            stdout=frontend_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                        frontend_log.flush()
                    current_revision = _frontend_build_revision(portal_root)
                    if current_revision != served_build_revision:
                        print(
                            '门户新构建已就绪，正在替换过时前端。',
                            flush=True,
                        )
                        _stop_frontend(
                            frontend,
                            state_path=frontend_state,
                            port=args.web_port,
                        )
                        frontend = _start_frontend(
                            node=node,
                            portal_root=portal_root,
                            port=args.web_port,
                            log_handle=frontend_log,
                            state_path=frontend_state,
                        )
                        served_build_revision = current_revision
                except subprocess.CalledProcessError:
                    print(
                        '门户新版本构建失败，已保留当前可用前端；'
                        '详情见 runs/portal_processes/frontend.log。',
                        flush=True,
                    )
                    frontend_check_at = time.monotonic() + 5.0
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
        if frontend is not None:
            _stop_frontend(
                frontend,
                state_path=frontend_state,
                port=args.web_port,
                raise_on_timeout=False,
            )
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
