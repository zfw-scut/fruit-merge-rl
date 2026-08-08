"""训练事实、派生监督与可插拔数据出口的稳定契约。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
from typing import Mapping, Protocol

import torch

from daxigua.simulator import (
    BatchDecisionSidecar,
    BatchDropResult,
    BatchMergeEvents,
    BatchPhysicsResult,
)

from .observations import TensorState
from .replay import ReplayStateReference


DECISION_FACT_FORMAT_VERSION = 1
DERIVED_SUPERVISION_FORMAT_VERSION = 1
FACT_PRODUCER_VERSION = 'decision-fact-collector-v1'


def _vector(name, value, batch_size, *, device=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name} must be torch.Tensor')
    if value.shape != (batch_size,):
        raise ValueError(f'{name} must have shape ({batch_size},)')
    if device is not None and value.device != device:
        raise ValueError(f'{name} must use device {device}')


def _state_payload(state):
    return {
        name: getattr(state, name).detach()
        for name in state.__dataclass_fields__
        if name != 'physics_fps'
    }


def _dataclass_tensor_payload(value):
    return {
        name: (
            _dataclass_tensor_payload(item)
            if hasattr(item, '__dataclass_fields__')
            else item.detach() if isinstance(item, torch.Tensor) else item
        )
        for name in value.__dataclass_fields__
        for item in (getattr(value, name),)
    }


def _tree_nbytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tree_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tree_nbytes(item) for item in value)
    return 0


def _tree_to_cpu(value, *, stream=None):
    if isinstance(value, torch.Tensor):
        source = value.detach()
        if source.device.type == 'cpu':
            return source.clone()
        destination = torch.empty_like(
            source, device='cpu', pin_memory=True
        )
        destination.copy_(source, non_blocking=True)
        return destination
    if isinstance(value, Mapping):
        return {
            key: _tree_to_cpu(item, stream=stream)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item, stream=stream) for item in value)
    if isinstance(value, list):
        return [_tree_to_cpu(item, stream=stream) for item in value]
    return value


def _filter_records(value, mask, original_count):
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == original_count:
            return value[mask]
        return value
    if isinstance(value, Mapping):
        return {
            key: _filter_records(item, mask, original_count)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _filter_records(item, mask, original_count) for item in value
        )
    if isinstance(value, list):
        return [
            _filter_records(item, mask, original_count) for item in value
        ]
    return value


def _concat_record_trees(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0)
    if isinstance(first, Mapping):
        return {
            key: _concat_record_trees([value[key] for value in values])
            for key in first
        }
    if isinstance(first, tuple):
        return tuple(
            _concat_record_trees([value[index] for value in values])
            for index in range(len(first))
        )
    if isinstance(first, list):
        return [
            _concat_record_trees([value[index] for value in values])
            for index in range(len(first))
        ]
    if any(value != first for value in values[1:]):
        return list(values)
    return first


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ActionSelectionBatch:
    """一次 actor 前向产生的完整动作选择事实。"""

    actions: torch.Tensor
    greedy_actions: torch.Tensor
    explore_mask: torch.Tensor
    q_values: torch.Tensor
    uncertainty: torch.Tensor | None = None
    active_learning_mask: torch.Tensor | None = None

    def __post_init__(self):
        if not isinstance(self.q_values, torch.Tensor) or self.q_values.ndim != 2:
            raise ValueError('q_values must be a two-dimensional Tensor')
        if not self.q_values.is_floating_point():
            raise TypeError('q_values must use a floating dtype')
        batch_size = int(self.q_values.shape[0])
        for name in ('actions', 'greedy_actions', 'explore_mask'):
            _vector(
                name,
                getattr(self, name),
                batch_size,
                device=self.q_values.device,
            )
        if self.actions.is_floating_point() or self.actions.dtype == torch.bool:
            raise TypeError('actions must use an integer dtype')
        if (
                self.greedy_actions.is_floating_point()
                or self.greedy_actions.dtype == torch.bool):
            raise TypeError('greedy_actions must use an integer dtype')
        if self.explore_mask.dtype != torch.bool:
            raise TypeError('explore_mask must use bool dtype')
        if self.uncertainty is not None:
            if self.uncertainty.shape != self.q_values.shape:
                raise ValueError('uncertainty shape must match q_values')
            if self.uncertainty.device != self.q_values.device:
                raise ValueError('uncertainty and q_values must share device')
        if self.active_learning_mask is not None:
            _vector(
                'active_learning_mask',
                self.active_learning_mask,
                batch_size,
                device=self.q_values.device,
            )
            if self.active_learning_mask.dtype != torch.bool:
                raise TypeError('active_learning_mask must use bool dtype')

    @property
    def batch_size(self):
        return int(self.q_values.shape[0])


@dataclass(frozen=True, slots=True)
class DecisionSelectionBatch:
    """选择器返回的固定上限设备端候选。"""

    rows: torch.Tensor
    valid_mask: torch.Tensor
    priorities: torch.Tensor
    reason_bits: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.rows, torch.Tensor) or self.rows.ndim != 1:
            raise ValueError('rows must be a one-dimensional Tensor')
        if self.rows.is_floating_point() or self.rows.dtype == torch.bool:
            raise TypeError('rows must use an integer dtype')
        count = int(self.rows.numel())
        for name in ('valid_mask', 'priorities', 'reason_bits'):
            _vector(
                name,
                getattr(self, name),
                count,
                device=self.rows.device,
            )
        if self.valid_mask.dtype != torch.bool:
            raise TypeError('valid_mask must use bool dtype')
        if not self.priorities.is_floating_point():
            raise TypeError('priorities must use a floating dtype')
        if self.reason_bits.is_floating_point() or self.reason_bits.dtype == torch.bool:
            raise TypeError('reason_bits must use an integer dtype')

    @classmethod
    def empty(cls, count, *, device):
        count = int(count)
        if count <= 0:
            raise ValueError('selection capacity must be positive')
        return cls(
            rows=torch.zeros(count, dtype=torch.int64, device=device),
            valid_mask=torch.zeros(count, dtype=torch.bool, device=device),
            priorities=torch.zeros(count, dtype=torch.float32, device=device),
            reason_bits=torch.zeros(count, dtype=torch.int64, device=device),
        )

    @property
    def capacity(self):
        return int(self.rows.numel())


@dataclass(frozen=True, slots=True)
class DecisionFactBatch:
    """可在 GPU 留存或异步归档的一批不可变事实记录。"""

    run_id: str
    producer_version: str
    information_scope: str
    decision_ids: torch.Tensor
    episode_ids: torch.Tensor
    segment_ids: torch.Tensor
    plan_ids: torch.Tensor
    environment_rows: torch.Tensor
    replay_reference: ReplayStateReference
    policy_versions: torch.Tensor
    priorities: torch.Tensor
    reason_bits: torch.Tensor
    valid_mask: torch.Tensor
    action_selection: ActionSelectionBatch
    rewards: torch.Tensor
    stages: torch.Tensor
    current: TensorState
    next_state: TensorState
    pre_sidecar: BatchDecisionSidecar
    drop: BatchDropResult
    physics: BatchPhysicsResult

    def __post_init__(self):
        if not self.run_id.strip():
            raise ValueError('run_id must not be empty')
        batch_size = int(self.decision_ids.numel())
        device = self.decision_ids.device
        vectors = (
            'episode_ids',
            'segment_ids',
            'plan_ids',
            'environment_rows',
            'policy_versions',
            'priorities',
            'reason_bits',
            'valid_mask',
            'rewards',
            'stages',
        )
        _vector('decision_ids', self.decision_ids, batch_size)
        for name in vectors:
            _vector(name, getattr(self, name), batch_size, device=device)
        if self.valid_mask.dtype != torch.bool:
            raise TypeError('valid_mask must use bool dtype')
        if self.replay_reference.count != batch_size:
            raise ValueError('replay reference batch size does not match facts')
        if self.action_selection.batch_size != batch_size:
            raise ValueError('action selection batch size does not match facts')
        if self.current.batch_size != batch_size:
            raise ValueError('current state batch size does not match facts')
        if self.next_state.batch_size != batch_size:
            raise ValueError('next state batch size does not match facts')
        if self.pre_sidecar.batch_size != batch_size:
            raise ValueError('sidecar batch size does not match facts')

    @property
    def batch_size(self):
        return int(self.decision_ids.numel())

    @property
    def device(self):
        return self.decision_ids.device

    @property
    def memory_bytes(self):
        return _tree_nbytes(self.to_payload())

    def to_payload(self):
        return {
            'format_version': DECISION_FACT_FORMAT_VERSION,
            'run_id': self.run_id,
            'producer_version': self.producer_version,
            'information_scope': self.information_scope,
            'physics_fps': float(self.current.physics_fps),
            'identity': {
                'decision_ids': self.decision_ids.detach(),
                'episode_ids': self.episode_ids.detach(),
                'segment_ids': self.segment_ids.detach(),
                'plan_ids': self.plan_ids.detach(),
                'environment_rows': self.environment_rows.detach(),
                'replay_indices': self.replay_reference.indices.detach(),
                'replay_generations': (
                    self.replay_reference.generations.detach()
                ),
                'policy_versions': self.policy_versions.detach(),
            },
            'selection': {
                'priorities': self.priorities.detach(),
                'reason_bits': self.reason_bits.detach(),
                'valid_mask': self.valid_mask.detach(),
            },
            'action': {
                'actions': self.action_selection.actions.detach(),
                'greedy_actions': (
                    self.action_selection.greedy_actions.detach()
                ),
                'explore_mask': self.action_selection.explore_mask.detach(),
                'q_values': self.action_selection.q_values.detach(),
                'uncertainty': (
                    None
                    if self.action_selection.uncertainty is None
                    else self.action_selection.uncertainty.detach()
                ),
                'active_learning_mask': (
                    None
                    if self.action_selection.active_learning_mask is None
                    else self.action_selection.active_learning_mask.detach()
                ),
            },
            'outcome': {
                'rewards': self.rewards.detach(),
                'stages': self.stages.detach(),
                'drop': _dataclass_tensor_payload(self.drop),
                'physics': _dataclass_tensor_payload(self.physics),
            },
            'current': _state_payload(self.current),
            'next_state': _state_payload(self.next_state),
            'pre_sidecar': _dataclass_tensor_payload(self.pre_sidecar),
        }


@dataclass(frozen=True, slots=True)
class DerivedSupervisionBatch:
    """未来反事实、辅助目标、规划或课程任务的通用追加信封。"""

    task_type: str
    producer_version: str
    information_scope: str
    decision_ids: torch.Tensor
    segment_ids: torch.Tensor
    plan_ids: torch.Tensor
    policy_versions: torch.Tensor
    valid_mask: torch.Tensor
    confidence: torch.Tensor
    payload: Mapping[str, object]
    status: str = 'available'
    format_version: int = DERIVED_SUPERVISION_FORMAT_VERSION

    def __post_init__(self):
        for name in (
                'task_type', 'producer_version', 'information_scope', 'status'):
            if not getattr(self, name).strip():
                raise ValueError(f'{name} must not be empty')
        batch_size = int(self.decision_ids.numel())
        device = self.decision_ids.device
        _vector('decision_ids', self.decision_ids, batch_size)
        for name in (
                'segment_ids', 'plan_ids', 'policy_versions',
                'valid_mask', 'confidence'):
            _vector(name, getattr(self, name), batch_size, device=device)
        if self.valid_mask.dtype != torch.bool:
            raise TypeError('valid_mask must use bool dtype')
        if not self.confidence.is_floating_point():
            raise TypeError('confidence must use a floating dtype')
        for name, value in self.payload.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.shape[0] != batch_size:
                    raise ValueError(
                        f'payload {name} first dimension must match batch'
                    )
                if value.device != device:
                    raise ValueError(f'payload {name} must use device {device}')


class DecisionFactSink(Protocol):
    def submit(self, batch: DecisionFactBatch) -> bool:
        ...

    def flush(self):
        ...

    def close(self):
        ...

    def metrics(self) -> Mapping[str, int | float | bool]:
        ...


class GpuDecisionBuffer:
    """有界、无 CPU 往返的事实批次出口，供未来设备端消费者使用。"""

    def __init__(self, capacity):
        if int(capacity) <= 0:
            raise ValueError('GPU decision capacity must be positive')
        self.capacity = int(capacity)
        self._batches = deque()
        self._records = 0
        self._evicted_records = 0
        self._lock = threading.Lock()

    def submit(self, batch):
        if not isinstance(batch, DecisionFactBatch):
            raise TypeError('batch must be DecisionFactBatch')
        count = batch.batch_size
        if count > self.capacity:
            self._evicted_records += count
            return False
        with self._lock:
            while self._records + count > self.capacity and self._batches:
                removed = self._batches.popleft()
                self._records -= removed.batch_size
                self._evicted_records += removed.batch_size
            self._batches.append(batch)
            self._records += count
        return True

    def pop(self):
        with self._lock:
            if not self._batches:
                return None
            batch = self._batches.popleft()
            self._records -= batch.batch_size
            return batch

    def flush(self):
        return None

    def close(self):
        return None

    def metrics(self):
        with self._lock:
            return {
                'capacity': self.capacity,
                'records': self._records,
                'batches': len(self._batches),
                'evicted_records': self._evicted_records,
            }


@dataclass(slots=True)
class _ArchiveJob:
    payload: dict
    cuda_event: object | None
    source_owner: object | None


class AsyncDecisionArchive:
    """有界异步事实归档；训练线程从不执行文件写入。"""

    def __init__(
            self,
            output_dir,
            *,
            shard_records=1024,
            queue_size=8,
            max_storage_bytes=0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_records = int(shard_records)
        self.max_storage_bytes = int(max_storage_bytes)
        if self.shard_records <= 0 or int(queue_size) <= 0:
            raise ValueError('archive shard and queue sizes must be positive')
        if self.max_storage_bytes < 0:
            raise ValueError('max_storage_bytes cannot be negative')
        self._queue = queue.Queue(maxsize=int(queue_size))
        self._slots = threading.BoundedSemaphore(int(queue_size))
        self._pending_payloads = []
        self._pending_records = 0
        self._closed = False
        self._storage_full = False
        self._error = None
        self._lock = threading.Lock()
        self._submitted_batches = 0
        self._dropped_batches = 0
        self._written_records = 0
        self._written_bytes = 0
        self._shards = []
        self._next_shard = 0
        self._copy_stream = None
        self._copy_device = None
        self._load_manifest()
        self._thread = threading.Thread(
            target=self._run,
            name='daxigua-decision-archive',
            daemon=True,
        )
        self._thread.start()

    @property
    def manifest_path(self):
        return self.output_dir / 'manifest.json'

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return
        payload = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        self._shards = list(payload.get('shards', ()))
        self._written_records = sum(
            int(item.get('records', 0)) for item in self._shards
        )
        self._written_bytes = sum(
            int(item.get('bytes', 0)) for item in self._shards
        )
        self._next_shard = len(self._shards)

    def _cpu_payload(self, batch):
        payload = batch.to_payload()
        if batch.device.type != 'cuda':
            return _tree_to_cpu(payload), None
        if self._copy_stream is None:
            self._copy_device = batch.device
            self._copy_stream = torch.cuda.Stream(device=batch.device)
        if batch.device != self._copy_device:
            raise ValueError('one archive cannot mix CUDA devices')
        current_stream = torch.cuda.current_stream(batch.device)
        self._copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._copy_stream):
            cpu_payload = _tree_to_cpu(payload, stream=self._copy_stream)
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        return cpu_payload, event

    def submit(self, batch):
        if self._closed:
            raise RuntimeError('decision archive is closed')
        if self._error is not None or self._storage_full:
            self._dropped_batches += 1
            return False
        if not self._slots.acquire(blocking=False):
            self._dropped_batches += 1
            return False
        try:
            payload, event = self._cpu_payload(batch)
            self._queue.put_nowait(_ArchiveJob(
                payload,
                event,
                batch if event is not None else None,
            ))
            self._submitted_batches += 1
            return True
        except BaseException:
            self._slots.release()
            raise

    def _write_manifest(self):
        payload = {
            'format_version': DECISION_FACT_FORMAT_VERSION,
            'written_records': self._written_records,
            'written_bytes': self._written_bytes,
            'storage_full': self._storage_full,
            'shards': self._shards,
        }
        temporary = self.manifest_path.with_suffix('.json.tmp')
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        os.replace(temporary, self.manifest_path)

    def _write_shard(self):
        with self._lock:
            if not self._pending_payloads:
                return
            records = _concat_record_trees(self._pending_payloads)
            count = self._pending_records
            self._pending_payloads = []
            self._pending_records = 0
        estimated = _tree_nbytes(records)
        if (
                self.max_storage_bytes > 0
                and self._written_bytes + estimated > self.max_storage_bytes):
            self._storage_full = True
            self._write_manifest()
            return
        name = f'decision_facts_{self._next_shard:06d}.pt'
        path = self.output_dir / name
        temporary = path.with_suffix('.pt.tmp')
        torch.save({
            'format_version': DECISION_FACT_FORMAT_VERSION,
            'record_count': count,
            'records': records,
        }, temporary)
        os.replace(temporary, path)
        size = path.stat().st_size
        self._shards.append({
            'path': name,
            'records': count,
            'bytes': size,
            'sha256': _sha256(path),
        })
        self._next_shard += 1
        self._written_records += count
        self._written_bytes += size
        if (
                self.max_storage_bytes > 0
                and self._written_bytes >= self.max_storage_bytes):
            self._storage_full = True
        self._write_manifest()

    def _consume(self, job):
        if job.cuda_event is not None:
            job.cuda_event.synchronize()
        payload = job.payload
        valid = payload['selection']['valid_mask'].to(torch.bool)
        original_count = int(valid.numel())
        count = int(valid.sum().item())
        if count <= 0:
            return
        filtered = _filter_records(payload, valid, original_count)
        with self._lock:
            self._pending_payloads.append(filtered)
            self._pending_records += count
            should_write = self._pending_records >= self.shard_records
        if should_write:
            self._write_shard()

    def _run(self):
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                if self._error is None and not self._storage_full:
                    self._consume(job)
            except BaseException as error:
                self._error = error
            finally:
                self._queue.task_done()
                if job is not None:
                    self._slots.release()

    def flush(self):
        self._queue.join()
        if self._error is not None:
            raise RuntimeError('decision archive writer failed') from self._error
        self._write_shard()

    def close(self):
        if self._closed:
            return
        failure = None
        try:
            self.flush()
        except BaseException as error:
            failure = error
        finally:
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        if failure is not None:
            raise failure
        if self._error is not None:
            raise RuntimeError('decision archive writer failed') from self._error

    def metrics(self):
        return {
            'submitted_batches': self._submitted_batches,
            'dropped_batches': self._dropped_batches,
            'queue_depth': self._queue.qsize(),
            'written_records': self._written_records,
            'written_bytes': self._written_bytes,
            'storage_full': self._storage_full,
            'writer_failed': self._error is not None,
        }


class CompositeDecisionSink:
    """将同一事实批次广播到多个相互独立的可选出口。"""

    def __init__(self, sinks=()):
        self.sinks = tuple(sinks)

    def submit(self, batch):
        accepted = False
        for sink in self.sinks:
            accepted = bool(sink.submit(batch)) or accepted
        return accepted

    def flush(self):
        for sink in self.sinks:
            sink.flush()

    def close(self):
        errors = []
        for sink in self.sinks:
            try:
                sink.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError('one or more decision sinks failed') from errors[0]

    def metrics(self):
        result = {}
        for index, sink in enumerate(self.sinks):
            prefix = f'{type(sink).__name__.lower()}_{index}'
            result.update({
                f'{prefix}_{name}': value
                for name, value in sink.metrics().items()
            })
        return result
