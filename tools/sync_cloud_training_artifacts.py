#!/usr/bin/env python3
"""从 SSH 云端训练目录同步轻量分析产物。

训练目录里的 checkpoint 和 ReplayBuffer 通常达到 GiB 级，不适合为了查看曲线而
整体下载。本工具只在远端选择固定白名单内的配置、指标、归因/恢复/故障 JSON 和
两张标准训练图，通过一条 SSH 连接把它们打成只读 tar 流，再在本地经过路径、类型、
大小和内容校验后以目录事务写入目标目录。

认证完全交给系统 OpenSSH：可以使用 ssh-agent、密钥、交互式密码提示，或调用方
预先配置的 ``SSH_ASKPASS``。本工具没有 password 参数，也不会读取、打印或写入
任何密码。同步失败或完整性检查失败时，已有本地证据不会被覆盖。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


# 增量同步只要求训练开始后必然存在的两个文件。episode 与曲线可能还没到第一次
# flush/plot 时点，因此默认把它们作为 optional，并在 manifest 中明确记录缺失。
BASE_REQUIRED_PATHS = (
    'config.json',
    'metrics.csv',
)
COMPLETE_REQUIRED_PATHS = (
    *BASE_REQUIRED_PATHS,
    'episode_metrics.csv',
    'attribution_warmup.json',
    'attribution_shutdown.json',
    'counterfactual_shutdown.json',
    'plots/training_curves.png',
    'plots/reward_breakdown_curves.png',
)
EXPECTED_OPTIONAL_PATHS = tuple(
    path for path in COMPLETE_REQUIRED_PATHS if path not in BASE_REQUIRED_PATHS
)

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 16 * 1024 * 1024
MAX_STDERR_BYTES = 1024 * 1024
MAX_MEMBER_COUNT = 1024
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
PNG_IEND = b'\x00\x00\x00\x00IEND\xaeB\x60\x82'
DEFAULT_REMOTE_PYTHON_CANDIDATES = (
    'python3',
    'python',
    '/root/miniconda3/envs/python-torch/bin/python',
    '/root/miniconda/envs/python-torch/bin/python',
    '/opt/conda/envs/python-torch/bin/python',
)
ALLOWED_TOP_LEVEL_EXACT = frozenset(
    {
        'config.json',
        'metrics.csv',
        'episode_metrics.csv',
        'attribution_warmup.json',
        'attribution_shutdown.json',
        'counterfactual_shutdown.json',
    }
)
ALLOWED_TOP_LEVEL_JSON_PREFIXES = (
    'attribution_resume_warmup_',
    'failure_',
    'resume_',
    'resume_config_',
)
ALLOWED_PLOT_NAMES = frozenset(
    {
        'training_curves.png',
        'reward_breakdown_curves.png',
    }
)
WINDOWS_RESERVED_STEMS = frozenset(
    {
        'CON',
        'PRN',
        'AUX',
        'NUL',
        *(f'COM{index}' for index in range(1, 10)),
        *(f'LPT{index}' for index in range(1, 10)),
    }
)

# 远端脚本只依赖 Python 标准库。它不创建远端临时文件，只把经过白名单和大小限制
# 的普通文件写到 stdout 的 gzip tar 流；错误信息单独写入 stderr。
REMOTE_ARCHIVE_SCRIPT = r'''
import pathlib
import sys
import tarfile

root = pathlib.Path(sys.argv[1])
max_file_bytes = int(sys.argv[2])
max_total_bytes = int(sys.argv[3])
allowed_exact = frozenset(sys.argv[4].split(","))
allowed_json_prefixes = tuple(sys.argv[5].split(","))
allowed_plots = frozenset(sys.argv[6].split(","))

if not root.is_dir():
    print(f"remote run directory does not exist: {root}", file=sys.stderr)
    raise SystemExit(21)

selected = []
for path in sorted(root.iterdir(), key=lambda item: item.name):
    if path.is_file() and not path.is_symlink():
        if (
            path.name in allowed_exact
            or (
                path.suffix.lower() == ".json"
                and path.name.startswith(allowed_json_prefixes)
            )
        ):
            selected.append(path)

plots_dir = root / "plots"
if plots_dir.is_dir() and not plots_dir.is_symlink():
    for path in sorted(plots_dir.iterdir(), key=lambda item: item.name):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.name in allowed_plots
        ):
            selected.append(path)

total_bytes = 0
for path in selected:
    size = path.stat().st_size
    if size > max_file_bytes:
        print(f"selected artifact exceeds per-file limit: {path}", file=sys.stderr)
        raise SystemExit(22)
    total_bytes += size

if total_bytes > max_total_bytes:
    print(
        f"selected artifacts exceed total limit: {total_bytes} bytes",
        file=sys.stderr,
    )
    raise SystemExit(23)

with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    for path in selected:
        archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
'''


class ArtifactSyncError(RuntimeError):
    """表示远端下载、产物校验或本地安装失败。"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 SSH 来源、本地目标和完整性模式。"""

    parser = argparse.ArgumentParser(
        description=(
            '通过 SSH 同步训练 run 的标准轻量配置、指标、归因和曲线；'
            '不会下载 checkpoint、ReplayBuffer 或日志。'
        ),
    )
    parser.add_argument('--host', required=True, help='SSH 主机名或 IP。')
    parser.add_argument('--port', type=int, default=22, help='SSH 端口，默认 22。')
    parser.add_argument('--user', required=True, help='SSH 用户名。')
    parser.add_argument(
        '--remote-run-dir',
        required=True,
        help='云端训练 run 的绝对 POSIX 路径。',
    )
    parser.add_argument(
        '--local-dir',
        type=Path,
        required=True,
        help='本地证据目录，例如 runs/cloud_evidence/10k。',
    )
    parser.add_argument(
        '--require-complete',
        action='store_true',
        help=(
            '要求 config、两份 CSV、warmup、两个 shutdown 和两张标准曲线'
            '全部存在，并达到配置目标 update；适合阶段结束归档。'
        ),
    )
    parser.add_argument(
        '--identity-file',
        type=Path,
        help='可选 OpenSSH 私钥路径；未提供时使用 ssh-agent/默认密钥/密码提示。',
    )
    parser.add_argument(
        '--known-hosts-file',
        type=Path,
        help='可选 known_hosts 文件；仍保留 OpenSSH 的严格主机指纹校验。',
    )
    parser.add_argument(
        '--ssh-executable',
        default='ssh',
        help='OpenSSH ssh 可执行文件，默认从 PATH 查找 ssh。',
    )
    parser.add_argument(
        '--remote-python',
        action='append',
        help=(
            '远端 Python 可执行文件；可重复传入。未提供时自动尝试 python3、'
            'python 和常见 python-torch Conda 路径。'
        ),
    )
    parser.add_argument(
        '--connect-timeout',
        type=int,
        default=15,
        help='SSH 建连超时秒数，默认 15。',
    )
    parser.add_argument(
        '--transfer-timeout',
        type=int,
        default=120,
        help='整个轻量归档传输的超时秒数，默认 120。',
    )
    return parser.parse_args(argv)


def _reject_control_characters(value: str, *, label: str) -> None:
    """拒绝会改变远端 shell 命令边界的控制字符。"""

    if not value or any(character in value for character in ('\x00', '\r', '\n')):
        raise ArtifactSyncError(f'{label} 不能为空或包含 NUL/换行。')


def validate_remote_run_dir(remote_run_dir: str) -> str:
    """校验并返回规范化的绝对 POSIX run 路径。

    路径最终仍会经过 ``shlex.quote``，这里额外拒绝 ``..``，避免操作者误把同步
    范围指到预期 run 之外。
    """

    _reject_control_characters(remote_run_dir, label='remote run dir')
    path = PurePosixPath(remote_run_dir)
    if not path.is_absolute():
        raise ArtifactSyncError('remote run dir 必须是绝对 POSIX 路径。')
    if '..' in path.parts:
        raise ArtifactSyncError('remote run dir 不允许包含 .. 路径段。')
    return path.as_posix()


def _validate_ssh_identity(user: str, host: str, port: int) -> str:
    """构造 ``user@host``，同时避免把额外参数注入 SSH 目标。"""

    _reject_control_characters(user, label='SSH user')
    _reject_control_characters(host, label='SSH host')
    if any(character.isspace() for character in user + host):
        raise ArtifactSyncError('SSH user/host 不允许包含空白字符。')
    if (
        user.startswith('-')
        or host.startswith('-')
        or '@' in user
        or '@' in host
    ):
        raise ArtifactSyncError('SSH user/host 格式无效。')
    if not 1 <= int(port) <= 65535:
        raise ArtifactSyncError('SSH port 必须位于 1..65535。')
    return f'{user}@{host}'


def build_remote_command(
        remote_run_dir: str,
        *,
        remote_python_candidates: Sequence[str] | None = None) -> str:
    """构建被 OpenSSH 交给远端 shell 的单条安全命令。

    云训练机器的非交互 SSH ``PATH`` 经常不包含 Conda。这里按顺序探测解释器，
    找到后用 ``exec`` 直接运行；候选和 Python argv 都分别经过 shell quote。
    """

    normalized_dir = validate_remote_run_dir(remote_run_dir)
    candidates = tuple(
        remote_python_candidates or DEFAULT_REMOTE_PYTHON_CANDIDATES
    )
    if not candidates:
        raise ArtifactSyncError('至少需要一个远端 Python 候选。')
    for candidate in candidates:
        _reject_control_characters(candidate, label='remote Python')

    python_arguments = (
        '-c',
        REMOTE_ARCHIVE_SCRIPT,
        normalized_dir,
        str(MAX_FILE_BYTES),
        str(MAX_TOTAL_BYTES),
        ','.join(sorted(ALLOWED_TOP_LEVEL_EXACT)),
        ','.join(ALLOWED_TOP_LEVEL_JSON_PREFIXES),
        ','.join(sorted(ALLOWED_PLOT_NAMES)),
    )
    quoted_candidates = ' '.join(
        shlex.quote(candidate) for candidate in candidates
    )
    quoted_arguments = ' '.join(
        shlex.quote(argument) for argument in python_arguments
    )
    probe = shlex.quote(
        'import pathlib, sys, tarfile; '
        'raise SystemExit(0 if sys.version_info >= (3, 8) else 1)'
    )
    return (
        f'for candidate in {quoted_candidates}; do '
        'if command -v "$candidate" >/dev/null 2>&1 '
        f'&& "$candidate" -c {probe} </dev/null >/dev/null 2>&1; then '
        f'exec "$candidate" {quoted_arguments}; '
        'fi; '
        'done; '
        'echo "no usable remote Python interpreter found" >&2; '
        'exit 127'
    )


def build_ssh_command(
        *,
        ssh_executable: str,
        host: str,
        port: int,
        user: str,
        remote_run_dir: str,
        connect_timeout: int = 15,
        identity_file: Path | None = None,
        known_hosts_file: Path | None = None,
        remote_python_candidates: Sequence[str] | None = None) -> list[str]:
    """生成 OpenSSH argv；密码和其它凭据永远不进入参数列表。"""

    target = _validate_ssh_identity(user, host, port)
    if connect_timeout <= 0:
        raise ArtifactSyncError('connect timeout 必须大于 0。')

    command = [
        ssh_executable,
        '-T',
        '-o',
        f'ConnectTimeout={connect_timeout}',
        '-o',
        'ConnectionAttempts=1',
        '-o',
        'ServerAliveInterval=30',
        '-o',
        'ServerAliveCountMax=3',
        '-p',
        str(port),
    ]
    if identity_file is not None:
        command.extend(('-i', str(identity_file)))
    if known_hosts_file is not None:
        command.extend(('-o', f'UserKnownHostsFile={known_hosts_file}'))
    command.extend(
        (
            target,
            build_remote_command(
                remote_run_dir,
                remote_python_candidates=remote_python_candidates,
            ),
        )
    )
    return command


def download_remote_archive(
        command: Sequence[str],
        archive_path: Path,
        *,
        timeout_seconds: int = 120,
        environment: dict[str, str] | None = None) -> None:
    """执行只读 SSH，并限制耗时、归档和错误输出的最大体积。"""

    if timeout_seconds <= 0:
        raise ArtifactSyncError('transfer timeout 必须大于 0。')
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = archive_path.with_name(f'{archive_path.name}.stderr')
    process: subprocess.Popen[bytes] | None = None
    stop_reason: str | None = None
    try:
        with (
                archive_path.open('wb') as archive_file,
                stderr_path.open('wb') as stderr_file):
            process = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=archive_file,
                stderr=stderr_file,
                env=environment,
            )
            started_at = time.monotonic()
            while process.poll() is None:
                if (
                    archive_path.stat().st_size > MAX_ARCHIVE_BYTES
                    or stderr_path.stat().st_size > MAX_STDERR_BYTES
                ):
                    stop_reason = 'SSH 输出超过轻量同步安全上限。'
                    process.kill()
                    break
                if time.monotonic() - started_at > timeout_seconds:
                    stop_reason = (
                        f'SSH 传输超过 {timeout_seconds} 秒超时。'
                    )
                    process.kill()
                    break
                time.sleep(0.05)
            returncode = process.wait()
    except OSError as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        archive_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
        raise ArtifactSyncError(f'无法启动 OpenSSH：{exc}') from exc
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        archive_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
        raise

    archive_size = (
        archive_path.stat().st_size if archive_path.is_file() else 0
    )
    stderr_size = (
        stderr_path.stat().st_size if stderr_path.is_file() else 0
    )
    if (
        archive_size > MAX_ARCHIVE_BYTES
        or stderr_size > MAX_STDERR_BYTES
    ):
        stop_reason = 'SSH 输出超过轻量同步安全上限。'

    detail = ''
    if stderr_path.is_file() and stderr_size:
        with stderr_path.open('rb') as stderr_file:
            stderr_file.seek(max(0, stderr_size - 2000))
            detail = stderr_file.read().decode(
                'utf-8',
                errors='replace',
            ).strip()
    stderr_path.unlink(missing_ok=True)

    if stop_reason is not None:
        archive_path.unlink(missing_ok=True)
        if detail:
            stop_reason = f'{stop_reason} 远端末尾输出：{detail}'
        raise ArtifactSyncError(stop_reason)
    if returncode != 0:
        archive_path.unlink(missing_ok=True)
        detail = detail or '远端未返回错误文本'
        raise ArtifactSyncError(
            f'SSH 同步失败（exit={returncode}）：{detail}'
        )
    if archive_size == 0:
        archive_path.unlink(missing_ok=True)
        raise ArtifactSyncError('SSH 成功退出，但没有收到训练产物归档。')


def is_allowed_artifact_path(relative_path: PurePosixPath) -> bool:
    """判断归档成员是否属于轻量分析白名单。"""

    parts = relative_path.parts
    if len(parts) == 1:
        name = relative_path.name
        return (
            name in ALLOWED_TOP_LEVEL_EXACT
            or (
                relative_path.suffix.lower() == '.json'
                and name.startswith(ALLOWED_TOP_LEVEL_JSON_PREFIXES)
            )
        )
    if len(parts) == 2 and parts[0] == 'plots':
        return relative_path.name in ALLOWED_PLOT_NAMES
    return False


def _validated_member_path(member: tarfile.TarInfo) -> PurePosixPath:
    """拒绝目录穿越、链接、设备文件和白名单外成员。"""

    relative_path = PurePosixPath(member.name)
    windows_unsafe = any(
        (
            '\\' in part
            or ':' in part
            or part.endswith((' ', '.'))
            or part.split('.', 1)[0].upper() in WINDOWS_RESERVED_STEMS
        )
        for part in relative_path.parts
    )
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {'', '.', '..'} for part in relative_path.parts)
        # POSIX tar 允许反斜杠出现在文件名中，但 Windows Path 会把它再次解释为
        # 目录分隔符；若不拒绝，``..\escape.json`` 会绕过上面的 POSIX 检查。
        or windows_unsafe
    ):
        raise ArtifactSyncError(f'归档包含不安全路径：{member.name!r}')
    if not member.isfile():
        raise ArtifactSyncError(f'归档只允许普通文件：{member.name!r}')
    if not is_allowed_artifact_path(relative_path):
        raise ArtifactSyncError(f'归档包含白名单外产物：{member.name!r}')
    if member.size < 0 or member.size > MAX_FILE_BYTES:
        raise ArtifactSyncError(f'归档成员大小越界：{member.name!r}')
    return relative_path


def safe_extract_archive(archive_path: Path, staging_dir: Path) -> list[Path]:
    """把归档安全解到临时目录，并返回相对文件路径。"""

    staging_dir.mkdir(parents=True, exist_ok=False)
    staging_root = staging_dir.resolve()
    extracted: list[Path] = []
    total_bytes = 0
    seen: set[str] = set()
    seen_casefold: set[str] = set()

    try:
        with tarfile.open(archive_path, mode='r:gz') as archive:
            members = archive.getmembers()
            if len(members) > MAX_MEMBER_COUNT:
                raise ArtifactSyncError('归档成员数量超过安全上限。')
            for member in members:
                relative_posix = _validated_member_path(member)
                relative_key = relative_posix.as_posix()
                if relative_key in seen:
                    raise ArtifactSyncError(f'归档包含重复成员：{relative_key}')
                casefold_key = relative_key.casefold()
                if casefold_key in seen_casefold:
                    raise ArtifactSyncError(
                        f'归档包含 Windows 大小写冲突成员：{relative_key}'
                    )
                seen.add(relative_key)
                seen_casefold.add(casefold_key)

                total_bytes += member.size
                if total_bytes > MAX_TOTAL_BYTES:
                    raise ArtifactSyncError('归档解压后总体积超过安全上限。')

                source = archive.extractfile(member)
                if source is None:
                    raise ArtifactSyncError(f'无法读取归档成员：{relative_key}')
                relative_local = Path(*relative_posix.parts)
                destination = staging_dir / relative_local
                if not destination.resolve().is_relative_to(staging_root):
                    raise ArtifactSyncError(
                        f'归档成员逃出 staging：{relative_key}'
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
                if destination.stat().st_size != member.size:
                    raise ArtifactSyncError(f'归档成员长度不完整：{relative_key}')
                extracted.append(relative_local)
    except (tarfile.TarError, OSError) as exc:
        if isinstance(exc, ArtifactSyncError):
            raise
        raise ArtifactSyncError(f'无法安全解包远端产物：{exc}') from exc

    if not extracted:
        raise ArtifactSyncError('远端 run 中没有可同步的 CSV、JSON 或 PNG。')
    return sorted(extracted, key=lambda path: path.as_posix())


def _validate_json(path: Path) -> None:
    """验证 JSON 没有在训练写入中途被截断。"""

    try:
        with path.open('r', encoding='utf-8') as file:
            json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactSyncError(f'JSON 无法解析：{path.name}: {exc}') from exc


def _validate_csv(path: Path, *, require_data_row: bool) -> None:
    """严格遍历 CSV 到 EOF，并校验表头、数据行和每行列数。"""

    try:
        with path.open('r', encoding='utf-8-sig', newline='') as file:
            reader = csv.reader(file, strict=True)
            header = next(reader, None)
            if not header or not any(cell.strip() for cell in header):
                raise ArtifactSyncError(f'CSV 缺少有效表头：{path.name}')
            row_count = 0
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ArtifactSyncError(
                        f'CSV 第 {line_number} 行列数不匹配：{path.name}'
                    )
                row_count += 1
            if require_data_row and row_count == 0:
                raise ArtifactSyncError(f'CSV 没有数据行：{path.name}')
    except (OSError, UnicodeError, csv.Error) as exc:
        if isinstance(exc, ArtifactSyncError):
            raise
        raise ArtifactSyncError(f'CSV 无法解析：{path.name}: {exc}') from exc


def _validate_png(path: Path) -> None:
    """验证 PNG 签名和最终 IEND，避免同步到正在覆盖中的半张曲线图。"""

    try:
        size = path.stat().st_size
        if size < len(PNG_SIGNATURE) + len(PNG_IEND):
            raise ArtifactSyncError(f'PNG 文件过短：{path.name}')
        with path.open('rb') as file:
            signature = file.read(len(PNG_SIGNATURE))
            file.seek(-len(PNG_IEND), os.SEEK_END)
            ending = file.read(len(PNG_IEND))
    except OSError as exc:
        raise ArtifactSyncError(f'PNG 无法读取：{path.name}: {exc}') from exc
    if signature != PNG_SIGNATURE or ending != PNG_IEND:
        raise ArtifactSyncError(f'PNG 签名或 IEND 不完整：{path.name}')


def _validate_complete_training_progress(staging_dir: Path) -> None:
    """确认严格包已经记录到配置目标 update。

    shutdown 文件的存在由 ``COMPLETE_REQUIRED_PATHS`` 保证；这里再核对 config
    目标与 metrics 最大 update，避免训练中途已经画过图时被误称为阶段结束包。
    这仍只是产物完整性检查，不能替代 readiness 的模型/replay 恢复门禁。
    """

    if (staging_dir / 'failure_latest.json').exists():
        raise ArtifactSyncError(
            '完整模式拒绝仍含 failure_latest.json 的活动失败 run。'
        )

    resume_configs = sorted(staging_dir.glob('resume_config_*.json'))
    effective_config_path = (
        resume_configs[-1]
        if resume_configs
        else staging_dir / 'config.json'
    )
    try:
        config = json.loads(
            effective_config_path.read_text(encoding='utf-8')
        )
        configured_updates = config['args']['total_updates']
        if (
            isinstance(configured_updates, bool)
            or int(configured_updates) <= 0
        ):
            raise ValueError('total_updates must be a positive integer')
        configured_updates = int(configured_updates)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError,
            ValueError) as exc:
        raise ArtifactSyncError(
            '完整模式无法从有效配置读取 args.total_updates：'
            f'{effective_config_path.name}'
        ) from exc

    maximum_update: int | None = None
    try:
        with (staging_dir / 'metrics.csv').open(
                'r',
                encoding='utf-8-sig',
                newline='') as file:
            for row in csv.DictReader(file):
                update = int(row['update_step'])
                maximum_update = (
                    update
                    if maximum_update is None
                    else max(maximum_update, update)
                )
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError,
            ValueError) as exc:
        raise ArtifactSyncError(
            '完整模式无法从 metrics.csv 读取有效 update_step。'
        ) from exc

    if maximum_update is None or maximum_update < configured_updates:
        raise ArtifactSyncError(
            '完整模式要求训练达到配置目标：'
            f'metrics max update={maximum_update}, '
            f'{effective_config_path.name} total_updates={configured_updates}'
        )


def validate_staged_artifacts(
        staging_dir: Path,
        relative_paths: Iterable[Path],
        *,
        require_complete: bool) -> list[str]:
    """校验标准文件及所有同步成员，返回缺失的 optional 路径。"""

    path_map = {
        relative_path.as_posix(): staging_dir / relative_path
        for relative_path in relative_paths
    }
    required = (
        COMPLETE_REQUIRED_PATHS if require_complete else BASE_REQUIRED_PATHS
    )
    missing_required = [path for path in required if path not in path_map]
    if missing_required:
        raise ArtifactSyncError(
            '远端 run 缺少必要产物：' + ', '.join(missing_required)
        )

    for relative_path, absolute_path in path_map.items():
        suffix = absolute_path.suffix.lower()
        if suffix == '.json':
            _validate_json(absolute_path)
        elif suffix == '.csv':
            _validate_csv(
                absolute_path,
                require_data_row=require_complete
                and relative_path in {'metrics.csv', 'episode_metrics.csv'},
            )
        elif suffix == '.png':
            _validate_png(absolute_path)

    if require_complete:
        _validate_complete_training_progress(staging_dir)

    return [
        path for path in EXPECTED_OPTIONAL_PATHS if path not in path_map
    ]


def _sha256_file(path: Path) -> str:
    """流式计算文件哈希，供同步清单和后续审计使用。"""

    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入本地同步清单。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.sync-',
        suffix='.tmp',
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _copy_staged_file(
        source: Path,
        replacement_root: Path,
        relative_path: Path) -> None:
    """把文件写入尚不可见的替换目录，并拒绝已有父目录链接。"""

    parent = replacement_root
    for part in relative_path.parent.parts:
        parent = parent / part
        if parent.is_symlink():
            raise ArtifactSyncError(
                f'本地目标含不安全目录链接：{relative_path.as_posix()}'
            )
        if parent.exists() and not parent.is_dir():
            raise ArtifactSyncError(
                f'本地目标父路径不是目录：{relative_path.as_posix()}'
            )
        parent.mkdir(exist_ok=True)

    destination = replacement_root / relative_path
    if destination.exists() and destination.is_dir():
        raise ArtifactSyncError(
            f'本地目标与同步文件类型冲突：{relative_path.as_posix()}'
        )
    if destination.is_symlink():
        destination.unlink()
    shutil.copy2(source, destination)


def _remove_managed_artifacts(replacement_root: Path) -> None:
    """移除上一轮由本工具管理的文件，避免 optional 缺失时残留旧证据。"""

    for path in tuple(replacement_root.iterdir()):
        relative_path = PurePosixPath(path.name)
        if is_allowed_artifact_path(relative_path):
            if path.is_dir() and not path.is_symlink():
                raise ArtifactSyncError(
                    f'本地受管文件被目录占用：{path.name}'
                )
            path.unlink(missing_ok=True)

    plots_dir = replacement_root / 'plots'
    if plots_dir.is_symlink():
        raise ArtifactSyncError('本地 plots 不能是符号链接。')
    if plots_dir.exists():
        if not plots_dir.is_dir():
            raise ArtifactSyncError('本地 plots 必须是普通目录。')
        for plot_name in ALLOWED_PLOT_NAMES:
            path = plots_dir / plot_name
            if path.is_dir() and not path.is_symlink():
                raise ArtifactSyncError(
                    f'本地受管曲线被目录占用：plots/{plot_name}'
                )
            path.unlink(missing_ok=True)

    manifest_path = replacement_root / 'sync_manifest.json'
    if manifest_path.is_dir() and not manifest_path.is_symlink():
        raise ArtifactSyncError('本地 sync_manifest.json 被目录占用。')
    manifest_path.unlink(missing_ok=True)


def _swap_replacement_directory(
        replacement_dir: Path,
        local_dir: Path) -> None:
    """以目录为事务单位安装完整新快照，失败时恢复旧目录。"""

    backup_dir: Path | None = None
    if local_dir.exists():
        backup_dir = local_dir.with_name(
            f'.{local_dir.name}.sync-backup-{uuid.uuid4().hex}'
        )
        try:
            os.replace(local_dir, backup_dir)
        except OSError as exc:
            raise ArtifactSyncError(f'无法备份本地证据目录：{exc}') from exc

    try:
        os.replace(replacement_dir, local_dir)
    except OSError as exc:
        if backup_dir is not None and backup_dir.exists():
            try:
                os.replace(backup_dir, local_dir)
            except OSError as rollback_exc:
                raise ArtifactSyncError(
                    '安装新证据目录失败，且旧目录自动恢复失败：'
                    f'install={exc}; rollback={rollback_exc}; '
                    f'backup={backup_dir}'
                ) from rollback_exc
        raise ArtifactSyncError(
            f'安装新证据目录失败，旧目录已恢复：{exc}'
        ) from exc

    if backup_dir is not None:
        # 目标目录和 manifest 已经作为同一个快照可见。旧备份清理属于 best effort，
        # Windows 杀毒软件短暂占用文件时不能把成功同步反报为失败。
        try:
            shutil.rmtree(backup_dir)
        except OSError:
            pass


def install_downloaded_archive(
        archive_path: Path,
        local_dir: Path,
        *,
        remote_run_dir: str,
        require_complete: bool,
        synced_at: dt.datetime | None = None) -> dict[str, Any]:
    """校验并安装已下载归档，成功后返回同时落盘的 manifest。"""

    normalized_remote_dir = validate_remote_run_dir(remote_run_dir)
    if local_dir.is_symlink():
        raise ArtifactSyncError('local dir 不能是符号链接。')
    local_dir = local_dir.resolve()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.exists() and not local_dir.is_dir():
        raise ArtifactSyncError('local dir 必须是普通目录，不能是文件或符号链接。')

    replacement_dir = local_dir.with_name(
        f'.{local_dir.name}.sync-replacement-{uuid.uuid4().hex}'
    )
    try:
        with tempfile.TemporaryDirectory(
                prefix='fruit-merge-artifact-stage-',
                dir=local_dir.parent) as temporary_dir:
            staging_dir = Path(temporary_dir) / 'payload'
            relative_paths = safe_extract_archive(archive_path, staging_dir)
            missing_optional = validate_staged_artifacts(
                staging_dir,
                relative_paths,
                require_complete=require_complete,
            )

            files = []
            for relative_path in relative_paths:
                source = staging_dir / relative_path
                files.append(
                    {
                        'path': relative_path.as_posix(),
                        'size_bytes': source.stat().st_size,
                        'sha256': _sha256_file(source),
                    }
                )

            timestamp = synced_at or dt.datetime.now(dt.timezone.utc)
            manifest: dict[str, Any] = {
                'schema_version': 1,
                'synced_at': timestamp.isoformat(),
                'source': {
                    'transport': 'ssh',
                    'remote_run_dir': normalized_remote_dir,
                },
                'mode': 'complete' if require_complete else 'incremental',
                'selection': {
                    'top_level_exact': sorted(ALLOWED_TOP_LEVEL_EXACT),
                    'top_level_json_prefixes': list(
                        ALLOWED_TOP_LEVEL_JSON_PREFIXES
                    ),
                    'plot_files': sorted(ALLOWED_PLOT_NAMES),
                    'checkpoint_and_replay_included': False,
                },
                'file_count': len(files),
                'total_size_bytes': sum(
                    item['size_bytes'] for item in files
                ),
                'missing_optional': missing_optional,
                'files': files,
            }
            try:
                # replacement 必须直接创建在目标父目录下。Windows 的
                # TemporaryDirectory 使用受保护 ACL；把其子目录 rename 为目标会
                # 让普通工作区身份失去访问权。普通 sibling mkdir 会继承父目录 ACL。
                replacement_dir.mkdir()
                if local_dir.exists():
                    shutil.copytree(
                        local_dir,
                        replacement_dir,
                        symlinks=True,
                        dirs_exist_ok=True,
                    )
                _remove_managed_artifacts(replacement_dir)
                for relative_path in relative_paths:
                    _copy_staged_file(
                        staging_dir / relative_path,
                        replacement_dir,
                        relative_path,
                    )
                _atomic_write_json(
                    replacement_dir / 'sync_manifest.json',
                    manifest,
                )
            except ArtifactSyncError:
                raise
            except OSError as exc:
                raise ArtifactSyncError(
                    f'无法准备本地事务快照：{exc}'
                ) from exc
            _swap_replacement_directory(replacement_dir, local_dir)
    finally:
        if replacement_dir.exists():
            try:
                shutil.rmtree(replacement_dir)
            except OSError:
                pass
    return manifest


def sync_remote_artifacts(
        *,
        host: str,
        port: int,
        user: str,
        remote_run_dir: str,
        local_dir: Path,
        require_complete: bool,
        ssh_executable: str = 'ssh',
        connect_timeout: int = 15,
        transfer_timeout: int = 120,
        identity_file: Path | None = None,
        known_hosts_file: Path | None = None,
        remote_python_candidates: Sequence[str] | None = None,
        environment: dict[str, str] | None = None) -> dict[str, Any]:
    """完成远端下载、安全解包、内容校验和本地原子安装。"""

    executable = shutil.which(ssh_executable)
    if executable is None:
        candidate = Path(ssh_executable)
        if not candidate.is_file():
            raise ArtifactSyncError(f'找不到 OpenSSH ssh：{ssh_executable}')
        executable = str(candidate)

    command = build_ssh_command(
        ssh_executable=executable,
        host=host,
        port=port,
        user=user,
        remote_run_dir=remote_run_dir,
        connect_timeout=connect_timeout,
        identity_file=identity_file,
        known_hosts_file=known_hosts_file,
        remote_python_candidates=remote_python_candidates,
    )
    with tempfile.TemporaryDirectory(prefix='fruit-merge-ssh-sync-') as temp_dir:
        archive_path = Path(temp_dir) / 'artifacts.tar.gz'
        download_remote_archive(
            command,
            archive_path,
            timeout_seconds=transfer_timeout,
            environment=environment,
        )
        return install_downloaded_archive(
            archive_path,
            local_dir,
            remote_run_dir=remote_run_dir,
            require_complete=require_complete,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    args = parse_args(argv)
    try:
        manifest = sync_remote_artifacts(
            host=args.host,
            port=args.port,
            user=args.user,
            remote_run_dir=args.remote_run_dir,
            local_dir=args.local_dir,
            require_complete=args.require_complete,
            ssh_executable=args.ssh_executable,
            connect_timeout=args.connect_timeout,
            transfer_timeout=args.transfer_timeout,
            identity_file=args.identity_file,
            known_hosts_file=args.known_hosts_file,
            remote_python_candidates=args.remote_python,
        )
    except ArtifactSyncError as exc:
        print(f'[sync-error] {exc}', file=sys.stderr)
        return 2

    print(
        '[sync-ok] '
        f'files={manifest["file_count"]} '
        f'bytes={manifest["total_size_bytes"]} '
        f'mode={manifest["mode"]} '
        f'local={args.local_dir.resolve()}',
    )
    missing = manifest['missing_optional']
    if missing:
        print('[sync-warning] optional missing: ' + ', '.join(missing))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
