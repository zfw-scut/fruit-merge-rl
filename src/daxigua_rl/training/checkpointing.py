"""可恢复训练所需的 checkpoint 基础设施。

本模块刻意不依赖具体训练循环。调用方负责提供模型、优化器和计数器等
``training_state``；这里负责：

* 以同目录临时文件和 :func:`os.replace` 原子写入 checkpoint；
* 捕获并恢复 Python、PyTorch CPU 和全部 CUDA 随机数生成器；
* 生成稳定、版本化的 run manifest 和配置指纹；
* 在恢复训练前拒绝会改变训练语义的配置漂移；
* 对显式暴露 ``state_dict`` / manifest 接口的可选组件保存状态。

``torch.load`` 会执行 pickle 反序列化，因此只能加载可信 checkpoint。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shutil
import tempfile
from argparse import Namespace
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch


CHECKPOINT_SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA_VERSION = 1
RNG_STATE_SCHEMA_VERSION = 1

# 这些字段只控制训练长度、观测频率或产物路径，不改变一次参数更新的语义。
# 调用方仍可显式传入自己的白名单；字段路径支持 ``section.field``。
DEFAULT_RESUME_MUTABLE_FIELDS = frozenset(
    {
        'config',
        'checkpoint_keep_last',
        'eval_interval',
        'log_interval',
        'plot_interval',
        'progress_interval',
        'resume',
        'run_dir',
        'save_interval',
        'total_updates',
        'overwrite_run_dir',
    }
)


class CheckpointError(RuntimeError):
    """checkpoint 数据无效或无法恢复。"""


class ResumeConfigMismatchError(CheckpointError):
    """当前配置与 checkpoint 的训练语义不兼容。"""

    def __init__(
            self,
            *,
            expected_fingerprint,
            actual_fingerprint,
            changed_fields):
        self.expected_fingerprint = str(expected_fingerprint)
        self.actual_fingerprint = str(actual_fingerprint)
        self.changed_fields = tuple(changed_fields)
        changed = ', '.join(self.changed_fields) or '<unknown>'
        super().__init__(
            'resume config fingerprint mismatch: '
            f'expected={self.expected_fingerprint} '
            f'actual={self.actual_fingerprint}; '
            f'changed_fields={changed}'
        )


def _config_mapping(config):
    """把常见配置容器转换成普通 mapping。"""

    if isinstance(config, Mapping):
        return dict(config)
    if isinstance(config, Namespace):
        return vars(config)
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    raise TypeError(
        'config must be a mapping, argparse.Namespace, or dataclass instance'
    )


def _canonicalize_json(value, *, path='$'):
    """转换成无歧义、可排序的 JSON 数据。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'non-finite float at {path}')
        return value
    if isinstance(value, Enum):
        return _canonicalize_json(value.value, path=path)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize_json(asdict(value), path=path)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f'JSON object key at {path} must be str, '
                    f'got {type(key).__name__}'
                )
            result[key] = _canonicalize_json(item, path=f'{path}.{key}')
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize_json(item, path=f'{path}[{index}]')
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        normalized = [
            _canonicalize_json(item, path=f'{path}[]')
            for item in value
        ]
        return sorted(normalized, key=_canonical_json_sort_key)
    raise TypeError(
        f'value at {path} is not canonically JSON serializable: '
        f'{type(value).__name__}'
    )


def _canonical_json_sort_key(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    )


def canonical_json(value):
    """返回稳定 JSON；相同语义不受 dict/set 插入顺序影响。"""

    normalized = _canonicalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    )


def _normalize_mutable_fields(mutable_fields):
    if mutable_fields is None:
        mutable_fields = DEFAULT_RESUME_MUTABLE_FIELDS
    normalized = []
    for field in mutable_fields:
        if not isinstance(field, str):
            raise TypeError('mutable field paths must be strings')
        field = field.strip()
        if not field or any(not part for part in field.split('.')):
            raise ValueError(f'invalid mutable field path: {field!r}')
        normalized.append(field)
    return tuple(sorted(set(normalized)))


def _without_mutable_fields(config, mutable_fields):
    """复制 canonical config，并删除允许在 resume 时变化的字段。"""

    normalized = json.loads(canonical_json(_config_mapping(config)))
    for field in _normalize_mutable_fields(mutable_fields):
        parts = field.split('.')
        cursor = normalized
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                cursor = None
                break
            cursor = cursor[part]
        if isinstance(cursor, dict):
            cursor.pop(parts[-1], None)
    return normalized


def canonical_config_json(
        config,
        *,
        mutable_fields=DEFAULT_RESUME_MUTABLE_FIELDS):
    """返回剔除 resume 可变字段后的 canonical JSON。"""

    return canonical_json(_without_mutable_fields(config, mutable_fields))


def config_fingerprint(
        config,
        *,
        mutable_fields=DEFAULT_RESUME_MUTABLE_FIELDS):
    """计算训练语义配置的稳定 SHA-256 指纹。"""

    serialized = canonical_config_json(
        config,
        mutable_fields=mutable_fields,
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _runtime_manifest():
    cuda_available = bool(torch.cuda.is_available())
    return {
        'cuda_available': cuda_available,
        'cuda_device_count': (
            int(torch.cuda.device_count()) if cuda_available else 0
        ),
        'cuda_runtime': torch.version.cuda,
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': str(torch.__version__),
    }


@dataclass(frozen=True)
class RunManifest:
    """版本化 run manifest 数据契约。"""

    schema_version: int
    created_at_utc: str
    config_fingerprint: str
    resume_mutable_fields: tuple[str, ...]
    config: dict[str, Any]
    runtime: dict[str, Any]
    metadata: dict[str, Any]

    def __post_init__(self):
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise CheckpointError(
                'unsupported run manifest schema version: '
                f'{self.schema_version!r}'
            )

        try:
            parsed_time = datetime.fromisoformat(
                self.created_at_utc.replace('Z', '+00:00')
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise CheckpointError('invalid created_at_utc') from exc
        if parsed_time.tzinfo is None:
            raise CheckpointError('created_at_utc must include a timezone')

        fields = _normalize_mutable_fields(self.resume_mutable_fields)
        config = _canonicalize_json(
            _config_mapping(self.config),
            path='$.config',
        )
        runtime = _canonicalize_json(
            _config_mapping(self.runtime),
            path='$.runtime',
        )
        metadata = _canonicalize_json(
            _config_mapping(self.metadata),
            path='$.metadata',
        )
        actual = config_fingerprint(config, mutable_fields=fields)
        if actual != self.config_fingerprint:
            raise CheckpointError(
                'run manifest config fingerprint does not match its config'
            )

        object.__setattr__(self, 'resume_mutable_fields', fields)
        object.__setattr__(self, 'config', config)
        object.__setattr__(self, 'runtime', runtime)
        object.__setattr__(self, 'metadata', metadata)

    def to_dict(self):
        """转换成可写入 JSON 或 torch checkpoint 的普通数据。"""

        return {
            'schema_version': self.schema_version,
            'created_at_utc': self.created_at_utc,
            'config_fingerprint': self.config_fingerprint,
            'resume_mutable_fields': list(self.resume_mutable_fields),
            'config': json.loads(canonical_json(self.config)),
            'runtime': json.loads(canonical_json(self.runtime)),
            'metadata': json.loads(canonical_json(self.metadata)),
        }

    @classmethod
    def from_dict(cls, value):
        """从 manifest mapping 恢复并完整校验数据契约。"""

        if not isinstance(value, Mapping):
            raise CheckpointError('run manifest must be a mapping')
        required = {
            'schema_version',
            'created_at_utc',
            'config_fingerprint',
            'resume_mutable_fields',
            'config',
            'runtime',
            'metadata',
        }
        missing = required.difference(value)
        if missing:
            raise CheckpointError(
                f'run manifest is missing fields: {sorted(missing)!r}'
            )
        return cls(
            schema_version=value['schema_version'],
            created_at_utc=value['created_at_utc'],
            config_fingerprint=value['config_fingerprint'],
            resume_mutable_fields=tuple(value['resume_mutable_fields']),
            config=dict(value['config']),
            runtime=dict(value['runtime']),
            metadata=dict(value['metadata']),
        )


def create_run_manifest(
        config,
        *,
        mutable_fields=DEFAULT_RESUME_MUTABLE_FIELDS,
        metadata=None,
        created_at_utc=None):
    """基于完整配置创建 manifest；完整配置用于解释和后续差异诊断。"""

    fields = _normalize_mutable_fields(mutable_fields)
    normalized_config = _canonicalize_json(_config_mapping(config))
    if metadata is None:
        metadata = {}
    if created_at_utc is None:
        created_at_utc = (
            datetime.now(timezone.utc)
            .isoformat(timespec='seconds')
            .replace('+00:00', 'Z')
        )
    return RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        created_at_utc=created_at_utc,
        config_fingerprint=config_fingerprint(
            normalized_config,
            mutable_fields=fields,
        ),
        resume_mutable_fields=fields,
        config=normalized_config,
        runtime=_runtime_manifest(),
        metadata=_canonicalize_json(_config_mapping(metadata)),
    )


def _changed_paths(expected, actual, *, prefix=''):
    if isinstance(expected, dict) and isinstance(actual, dict):
        paths = []
        for key in sorted(set(expected).union(actual)):
            path = f'{prefix}.{key}' if prefix else key
            if key not in expected or key not in actual:
                paths.append(path)
            else:
                paths.extend(
                    _changed_paths(expected[key], actual[key], prefix=path)
                )
        return paths
    if isinstance(expected, list) and isinstance(actual, list):
        paths = []
        for index in range(max(len(expected), len(actual))):
            path = f'{prefix}[{index}]'
            if index >= len(expected) or index >= len(actual):
                paths.append(path)
            else:
                paths.extend(
                    _changed_paths(
                        expected[index],
                        actual[index],
                        prefix=path,
                    )
                )
        return paths
    return [] if expected == actual else [prefix or '<root>']


def validate_resume_config(
        manifest,
        current_config,
        *,
        mutable_fields=None):
    """校验 resume 配置并返回实际指纹。

    ``mutable_fields=None`` 使用 manifest 创建时记录的白名单。显式传入时，
    会对 manifest 内的原始配置和当前配置使用同一新白名单重新计算，而不是
    盲目信任旧指纹。
    """

    if not isinstance(manifest, RunManifest):
        manifest = RunManifest.from_dict(manifest)
    fields = (
        manifest.resume_mutable_fields
        if mutable_fields is None
        else _normalize_mutable_fields(mutable_fields)
    )
    expected_config = _without_mutable_fields(manifest.config, fields)
    actual_config = _without_mutable_fields(current_config, fields)
    expected = hashlib.sha256(
        canonical_json(expected_config).encode('utf-8')
    ).hexdigest()
    actual = hashlib.sha256(
        canonical_json(actual_config).encode('utf-8')
    ).hexdigest()
    if actual != expected:
        raise ResumeConfigMismatchError(
            expected_fingerprint=expected,
            actual_fingerprint=actual,
            changed_fields=_changed_paths(expected_config, actual_config),
        )
    return actual


def capture_rng_state():
    """捕获 Python、torch CPU 以及每块 CUDA 设备的 RNG 状态。"""

    cuda_states = ()
    if torch.cuda.is_available():
        cuda_states = tuple(
            state.detach().cpu().clone()
            for state in torch.cuda.get_rng_state_all()
        )
    return {
        'schema_version': RNG_STATE_SCHEMA_VERSION,
        'python': random.getstate(),
        'torch_cpu': torch.get_rng_state().detach().cpu().clone(),
        'torch_cuda': cuda_states,
        'cuda_device_count': len(cuda_states),
    }


def _validated_rng_state(state):
    if not isinstance(state, Mapping):
        raise CheckpointError('RNG state must be a mapping')
    if state.get('schema_version') != RNG_STATE_SCHEMA_VERSION:
        raise CheckpointError(
            f'unsupported RNG state schema version: '
            f'{state.get("schema_version")!r}'
        )
    cpu_state = state.get('torch_cpu')
    if not isinstance(cpu_state, torch.Tensor):
        raise CheckpointError('torch_cpu RNG state must be a tensor')
    cuda_states = state.get('torch_cuda')
    if not isinstance(cuda_states, (list, tuple)):
        raise CheckpointError('torch_cuda RNG states must be a sequence')
    if any(not isinstance(item, torch.Tensor) for item in cuda_states):
        raise CheckpointError('every CUDA RNG state must be a tensor')
    declared_count = state.get('cuda_device_count')
    if declared_count != len(cuda_states):
        raise CheckpointError(
            'cuda_device_count does not match saved CUDA RNG states'
        )
    return (
        state.get('python'),
        cpu_state.detach().cpu().clone(),
        tuple(item.detach().cpu().clone() for item in cuda_states),
    )


def validate_rng_state(state):
    """只读校验 RNG checkpoint 结构和 Python/CPU 状态可恢复性。"""

    python_state, cpu_state, cuda_states = _validated_rng_state(state)
    try:
        probe_python = random.Random()
        probe_python.setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise CheckpointError('invalid Python RNG state') from exc
    if (
            cpu_state.device.type != 'cpu'
            or cpu_state.dtype != torch.uint8
            or cpu_state.ndim != 1
            or cpu_state.numel() <= 0):
        raise CheckpointError(
            'torch_cpu RNG state must be a non-empty CPU uint8 vector'
        )
    try:
        probe_torch = torch.Generator(device='cpu')
        probe_torch.set_state(cpu_state)
    except (TypeError, RuntimeError) as exc:
        raise CheckpointError('invalid torch CPU RNG state') from exc
    for index, cuda_state in enumerate(cuda_states):
        if (
                cuda_state.device.type != 'cpu'
                or cuda_state.dtype != torch.uint8
                or cuda_state.ndim != 1
                or cuda_state.numel() <= 0):
            raise CheckpointError(
                f'torch_cuda[{index}] RNG state must be a non-empty '
                'CPU uint8 vector'
            )
    return {
        'schema_version': RNG_STATE_SCHEMA_VERSION,
        'python_state_valid': True,
        'torch_cpu_state_bytes': int(cpu_state.numel()),
        'cuda_device_count': len(cuda_states),
        'torch_cuda_state_bytes': [
            int(item.numel())
            for item in cuda_states
        ],
    }


def restore_rng_state(state, *, strict_cuda=True):
    """恢复 RNG 状态。

    若 checkpoint 包含 CUDA 状态，默认要求当前 CUDA 设备数完全一致，避免
    表面恢复成功、实际随机序列已经分叉。``strict_cuda=False`` 时，硬件不兼容
    会跳过 CUDA 恢复，但仍恢复 Python 和 CPU 状态。
    """

    python_state, cpu_state, cuda_states = _validated_rng_state(state)
    restore_cuda = bool(cuda_states)
    if restore_cuda:
        current_count = (
            int(torch.cuda.device_count())
            if torch.cuda.is_available()
            else 0
        )
        if current_count != len(cuda_states):
            if strict_cuda:
                raise CheckpointError(
                    'CUDA RNG state count mismatch: '
                    f'checkpoint={len(cuda_states)} current={current_count}'
                )
            restore_cuda = False

    try:
        random.setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise CheckpointError('invalid Python RNG state') from exc
    torch.set_rng_state(cpu_state)
    if restore_cuda:
        torch.cuda.set_rng_state_all(list(cuda_states))


def _component_manifest(component):
    provider = getattr(component, 'checkpoint_manifest', None)
    if provider is None:
        provider = getattr(component, 'manifest', None)
    if provider is None:
        return None
    value = provider() if callable(provider) else provider
    return _canonicalize_json(value, path='$.component_manifest')


def capture_optional_component(component):
    """捕获一个可选组件公开的状态/manifest；无接口时返回 ``None``。"""

    state_provider = getattr(component, 'checkpoint_state_dict', None)
    protocol = 'checkpoint_state_dict'
    if not callable(state_provider):
        state_provider = getattr(component, 'state_dict', None)
        protocol = 'state_dict'

    manifest = _component_manifest(component)
    if not callable(state_provider) and manifest is None:
        return None
    return {
        'state_protocol': protocol if callable(state_provider) else None,
        'state': state_provider() if callable(state_provider) else None,
        'manifest': manifest,
    }


def capture_optional_components(components):
    """捕获 mapping 中所有具备显式 checkpoint API 的组件。"""

    if components is None:
        return {}
    if not isinstance(components, Mapping):
        raise TypeError('components must be a mapping')
    snapshots = {}
    for name, component in components.items():
        if not isinstance(name, str) or not name:
            raise TypeError('component names must be non-empty strings')
        snapshot = capture_optional_component(component)
        if snapshot is not None:
            snapshots[name] = snapshot
    return snapshots


def restore_optional_components(
        snapshots,
        components,
        *,
        strict=True):
    """把通用组件状态交还给它们公开的 loader，并返回已恢复的名称。"""

    if not snapshots:
        return ()
    if not isinstance(snapshots, Mapping):
        raise CheckpointError('component snapshots must be a mapping')
    if not isinstance(components, Mapping):
        raise TypeError('components must be a mapping')

    restored = []
    for name, snapshot in snapshots.items():
        if name not in components:
            if strict:
                raise CheckpointError(
                    f'missing component required by checkpoint: {name}'
                )
            continue
        if not isinstance(snapshot, Mapping):
            raise CheckpointError(
                f'component snapshot {name!r} must be a mapping'
            )

        component = components[name]
        protocol = snapshot.get('state_protocol')
        state = snapshot.get('state')
        validator = getattr(
            component,
            'validate_checkpoint_manifest',
            None,
        )
        if callable(validator) and snapshot.get('manifest') is not None:
            # 先验证兼容性，再把状态写进组件，避免失败时留下半恢复对象。
            validator(snapshot['manifest'])

        if protocol == 'checkpoint_state_dict':
            loader = getattr(component, 'load_checkpoint_state_dict', None)
        elif protocol == 'state_dict':
            loader = getattr(component, 'load_state_dict', None)
        elif protocol is None:
            loader = None
        else:
            raise CheckpointError(
                f'unsupported component state protocol: {protocol!r}'
            )

        if protocol is not None:
            if not callable(loader):
                if strict:
                    raise CheckpointError(
                        f'component {name!r} does not support {protocol!r} '
                        'restore'
                    )
                continue
            loader(state)

        restored.append(name)
    return tuple(restored)


def build_training_checkpoint(
        *,
        training_state,
        config,
        manifest=None,
        mutable_fields=DEFAULT_RESUME_MUTABLE_FIELDS,
        metadata=None,
        components=None):
    """构建只含普通容器、tensor 和调用方状态的版本化 checkpoint。"""

    if not isinstance(training_state, Mapping):
        raise TypeError('training_state must be a mapping')
    if manifest is None:
        manifest = create_run_manifest(
            config,
            mutable_fields=mutable_fields,
            metadata=metadata,
        )
    elif not isinstance(manifest, RunManifest):
        manifest = RunManifest.from_dict(manifest)
    validate_resume_config(manifest, config)
    return {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'run_manifest': manifest.to_dict(),
        'training_state': dict(training_state),
        'rng_state': capture_rng_state(),
        'component_snapshots': capture_optional_components(components),
    }


def atomic_torch_save(value, path):
    """在目标同目录写临时文件，成功后用 ``os.replace`` 原子替换。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.',
        suffix='.tmp',
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        # 直接复用 mkstemp 的可写句柄；Windows 不允许对只读句柄 fsync。
        with os.fdopen(descriptor, 'wb') as file:
            torch.save(value, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def atomic_clone_file(source, destination):
    """原子物化同卷文件，优先硬链接，失败时退化为一次字节复制。

    大型训练 checkpoint 会同时保留 ``latest``、周期 ``step`` 和可选 ``best``。
    它们在同一个保存点内容完全相同，因此先把 payload 只序列化到 ``latest``，
    再用本函数创建其它稳定名字，可避免重复的 pickle/torch 序列化。硬链接仍保留
    正确的版本语义：下一次原子替换 ``latest`` 时，旧 step/best 继续指向旧 inode。
    """

    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        return destination, 'same_file'
    if not source.is_file():
        raise FileNotFoundError(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.',
        suffix='.tmp',
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        # os.link 要求目标不存在；mkstemp 只用于安全预留随机名字。
        temporary_path.unlink()
        try:
            os.link(source, temporary_path)
            method = 'hardlink'
        except OSError:
            with (
                    source.open('rb') as source_file,
                    temporary_path.open('xb') as destination_file):
                shutil.copyfileobj(
                    source_file,
                    destination_file,
                    length=8 * 1024 * 1024,
                )
                destination_file.flush()
                os.fsync(destination_file.fileno())
            method = 'copy'
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination, method


def save_training_checkpoint(
        path,
        *,
        training_state,
        config,
        manifest=None,
        mutable_fields=DEFAULT_RESUME_MUTABLE_FIELDS,
        metadata=None,
        components=None):
    """构建并原子保存训练 checkpoint，返回保存路径。"""

    payload = build_training_checkpoint(
        training_state=training_state,
        config=config,
        manifest=manifest,
        mutable_fields=mutable_fields,
        metadata=metadata,
        components=components,
    )
    return atomic_torch_save(payload, path)


def _load_trusted_torch_file(path, map_location):
    """兼容仍未提供 ``weights_only`` 参数的较旧 PyTorch。"""

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError as exc:
        if 'weights_only' not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


def validate_training_checkpoint(payload):
    """校验顶层版本与必要字段，并返回已解析的 run manifest。"""

    if not isinstance(payload, Mapping):
        raise CheckpointError('training checkpoint must be a mapping')
    if payload.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            'unsupported training checkpoint schema version: '
            f'{payload.get("schema_version")!r}'
        )
    required = {
        'run_manifest',
        'training_state',
        'rng_state',
        'component_snapshots',
    }
    missing = required.difference(payload)
    if missing:
        raise CheckpointError(
            f'training checkpoint is missing fields: {sorted(missing)!r}'
        )
    if not isinstance(payload['training_state'], Mapping):
        raise CheckpointError('training_state must be a mapping')
    _validated_rng_state(payload['rng_state'])
    if not isinstance(payload['component_snapshots'], Mapping):
        raise CheckpointError('component_snapshots must be a mapping')
    return RunManifest.from_dict(payload['run_manifest'])


def extract_inference_checkpoint(payload):
    """统一读取版本化训练 checkpoint 和早期扁平 checkpoint。

    返回 ``(training_args, online_model_state)``。新格式先执行完整 schema 与
    manifest 校验；旧格式仅作为历史模型观看/对比的兼容入口。
    """

    if not isinstance(payload, Mapping):
        raise CheckpointError('checkpoint must be a mapping')
    if payload.get('schema_version') == CHECKPOINT_SCHEMA_VERSION:
        manifest = validate_training_checkpoint(payload)
        state = payload['training_state']
        if 'online_model' not in state:
            raise CheckpointError(
                'training checkpoint has no online_model state'
            )
        return dict(manifest.config), state['online_model']
    args = payload.get('args')
    if not isinstance(args, Mapping) or 'online_model' not in payload:
        raise CheckpointError(
            'unsupported legacy inference checkpoint'
        )
    return dict(args), payload['online_model']


def load_training_checkpoint(
        path,
        *,
        current_config=None,
        mutable_fields=None,
        map_location='cpu',
        restore_rng=False,
        components=None,
        strict_cuda_rng=True,
        strict_components=True):
    """加载、校验并可选恢复 RNG/组件，返回原始 checkpoint mapping。"""

    payload = _load_trusted_torch_file(path, map_location)
    manifest = validate_training_checkpoint(payload)
    if current_config is not None:
        validate_resume_config(
            manifest,
            current_config,
            mutable_fields=mutable_fields,
        )
    if components is not None:
        restore_optional_components(
            payload['component_snapshots'],
            components,
            strict=strict_components,
        )
    if restore_rng:
        restore_rng_state(
            payload['rng_state'],
            strict_cuda=strict_cuda_rng,
        )
    return payload
