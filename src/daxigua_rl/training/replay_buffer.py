"""DQN 使用的固定容量经验回放池。

ReplayBuffer 的职责保持单一：

1. 保存 rollout 过程中产生的 `TensorTransition`。
2. 容量满后丢弃最旧经验。
3. 训练时随机采样一批经验。

为支持大规模训练，本实现提供两种模式：

- 默认内存模式：行为接近普通环形 buffer，适合测试和小训练。
- 热内存 + 冷磁盘模式：最近经验保留在内存，较旧经验按段写到磁盘，
  采样时从热数据和冷缓存中混合抽样，降低常驻内存压力。

它仍然不负责构图、动作选择、reward 计算或 DQN 参数更新。
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

from .tensor_transition import TensorTransition


class ReplayBuffer:
    """固定容量 Replay Buffer。

    默认构造 `ReplayBuffer(capacity=...)` 会启用纯内存环形存储，兼容旧测试和
    小规模训练。训练脚本如果传入 `cold_dir` 且 `hot_capacity < capacity`，
    则启用热内存 + 冷磁盘分层。
    """

    def __init__(
            self,
            capacity=100_000,
            seed=None,
            hot_capacity=None,
            cold_dir=None,
            segment_size=1024,
            cold_cache_size=4096,
            cold_sample_ratio=0.25,
            cold_cache_refresh_interval=500):
        """创建经验回放池。

        参数：
        - `capacity`: 最大保存多少条 transition。
        - `seed`: buffer 自己的随机种子，用于复现 sample 结果。
        - `hot_capacity`: 常驻内存的最新 transition 数量；不传则等于总容量。
        - `cold_dir`: 冷数据磁盘目录；传入后启用分层存储。
        - `segment_size`: 冷数据每多少条 transition 写一个段文件。
        - `cold_cache_size`: 采样时最多缓存多少条冷数据到内存。
        - `cold_sample_ratio`: 每个 batch 期望从冷数据中采样的比例。
        - `cold_cache_refresh_interval`: 每多少次 sample 刷新一次冷缓存。
        """

        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be positive')

        hot_capacity = capacity if hot_capacity is None else int(hot_capacity)
        if hot_capacity <= 0:
            raise ValueError('hot_capacity must be positive')

        self.capacity = capacity
        self.hot_capacity = min(hot_capacity, capacity)
        self.segment_size = int(segment_size)
        self.cold_cache_size = int(cold_cache_size)
        self.cold_sample_ratio = float(cold_sample_ratio)
        self.cold_cache_refresh_interval = int(cold_cache_refresh_interval)

        if self.segment_size <= 0:
            raise ValueError('segment_size must be positive')
        if self.cold_cache_size < 0:
            raise ValueError('cold_cache_size must be >= 0')
        if self.cold_sample_ratio < 0.0 or self.cold_sample_ratio > 1.0:
            raise ValueError('cold_sample_ratio must be in [0, 1]')
        if self.cold_cache_refresh_interval <= 0:
            raise ValueError('cold_cache_refresh_interval must be positive')

        # 独立随机源避免和环境随机数、模型初始化随机数互相影响。
        self._rng = random.Random(seed)

        # 不传 cold_dir 时保持老的纯内存环形模式，避免测试和小实验被磁盘 I/O 影响。
        self._disk_enabled = cold_dir is not None and self.hot_capacity < self.capacity

        # 纯内存模式使用环形写入，避免容量较大时频繁 pop(0)。
        self._items = []
        self._next_index = 0

        # 分层模式使用时间顺序列表管理热数据。hot_capacity 通常远小于总容量，
        # 老数据溢出时写到磁盘，常驻内存压力受 hot_capacity 控制。
        self._hot_items = deque()
        self._pending_cold = deque()
        self._segments = []
        self._segment_item_count = 0
        self._cold_cache = []
        self._sample_calls = 0
        self._next_segment_index = 0

        self.cold_dir = Path(cold_dir) if cold_dir is not None else None
        if self._disk_enabled:
            self.cold_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        """返回当前已保存的 transition 数量。"""

        if not self._disk_enabled:
            return len(self._items)

        return len(self._hot_items) + len(self._pending_cold) + self._segment_item_count

    @property
    def is_full(self):
        """ReplayBuffer 是否已经达到最大容量。"""

        return len(self) == self.capacity

    @property
    def remaining_capacity(self):
        """距离满容量还剩多少位置。"""

        return max(0, self.capacity - len(self))

    @property
    def storage_stats(self):
        """返回分层存储状态，供 profiling 日志和调试查看。"""

        if not self._disk_enabled:
            return {
                'mode': 'memory',
                'hot_count': len(self._items),
                'pending_cold_count': 0,
                'cold_segment_count': 0,
                'cold_count': 0,
                'cold_cache_count': 0,
            }

        return {
            'mode': 'hybrid',
            'hot_count': len(self._hot_items),
            'pending_cold_count': len(self._pending_cold),
            'cold_segment_count': len(self._segments),
            'cold_count': len(self._pending_cold) + self._segment_item_count,
            'cold_cache_count': len(self._cold_cache),
        }

    def clear(self):
        """清空所有经验，并重置写入位置。"""

        self._items.clear()
        self._next_index = 0

        self._hot_items.clear()
        self._pending_cold.clear()
        self._cold_cache.clear()
        self._segment_item_count = 0
        self._sample_calls = 0

        for segment in self._segments:
            try:
                Path(segment['path']).unlink()
            except FileNotFoundError:
                pass
        self._segments.clear()

    def push(self, transition):
        """写入一条经验对象。"""

        if not isinstance(transition, TensorTransition):
            raise TypeError(f'transition must be TensorTransition, got {type(transition)!r}')

        if self._disk_enabled:
            self._push_hybrid(transition)
        else:
            self._push_memory(transition)

    def extend(self, transitions):
        """批量写入 transition，并返回实际写入数量。"""

        count = 0
        for transition in transitions:
            self.push(transition)
            count += 1
        return count

    def sample(self, batch_size):
        """随机无放回采样一批 transition。"""

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if batch_size > len(self):
            raise ValueError(
                f'cannot sample batch_size={batch_size} from replay buffer with {len(self)} items'
            )

        if not self._disk_enabled:
            return tuple(self._rng.sample(self._items, batch_size))

        self._sample_calls += 1
        cold_available = len(self._pending_cold) + self._segment_item_count
        hot_available = len(self._hot_items)

        cold_count = 0
        if cold_available > 0 and self.cold_sample_ratio > 0.0:
            cold_count = min(cold_available, int(round(batch_size * self.cold_sample_ratio)))

        hot_count = batch_size - cold_count
        if hot_count > hot_available:
            cold_count += hot_count - hot_available
            hot_count = hot_available
        if cold_count > cold_available:
            hot_count += cold_count - cold_available
            cold_count = cold_available

        hot_samples = self._sample_hot(hot_count) if hot_count else []
        cold_samples = self._sample_cold(cold_count) if cold_count else []
        batch = list(hot_samples) + list(cold_samples)
        self._rng.shuffle(batch)
        return tuple(batch)

    def is_ready(self, batch_size):
        """当前经验数量是否足够采样一个 batch。"""

        return len(self) >= int(batch_size)

    def to_tuple(self):
        """按时间顺序返回当前保存的所有 transition。

        这个方法主要用于调试和测试。分层模式下会读取所有冷数据段，正式训练
        不应在大 buffer 上调用它。
        """

        if not self._disk_enabled:
            if len(self._items) < self.capacity:
                return tuple(self._items)

            # buffer 满了以后，`_next_index` 指向下一次要覆盖的位置，
            # 也就是当前最旧 transition 的位置。
            return tuple(self._items[self._next_index:] + self._items[:self._next_index])

        cold_items = []
        for segment in self._segments:
            cold_items.extend(self._load_segment(segment))
        cold_items.extend(self._pending_cold)
        return tuple(cold_items + list(self._hot_items))

    def flush(self):
        """把当前 pending 冷数据写入磁盘段。

        训练中 pending 数量最多为 `segment_size - 1`，通常不必频繁 flush。
        这个方法主要用于长跑结束后释放一点内存或调试冷数据文件。
        """

        if self._disk_enabled and self._pending_cold:
            self._write_pending_segment()

    def _push_memory(self, transition):
        """纯内存模式下按环形 buffer 写入。"""

        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._next_index] = transition

        self._next_index = (self._next_index + 1) % self.capacity

    def _push_hybrid(self, transition):
        """分层模式下写入热内存，并把溢出的旧数据转入冷层。"""

        self._hot_items.append(transition)
        if len(self._hot_items) > self.hot_capacity:
            oldest_hot = self._hot_items.popleft()
            self._pending_cold.append(oldest_hot)

        if len(self._pending_cold) >= self.segment_size:
            self._write_pending_segment()

        self._enforce_capacity()

    def _enforce_capacity(self):
        """确保分层模式总容量不超过 `capacity`。"""

        while len(self) > self.capacity:
            if self._segments:
                segment = self._segments.pop(0)
                self._segment_item_count -= int(segment['count'])
                self._remove_segment_file(segment)
                self._cold_cache.clear()
            elif self._pending_cold:
                self._pending_cold.popleft()
            elif self._hot_items:
                self._hot_items.popleft()
            else:
                break

    def _write_pending_segment(self):
        """把 pending 冷数据压缩成一个磁盘段文件。"""

        transitions = tuple(self._pending_cold)
        if not transitions:
            return

        path = self.cold_dir / f'segment_{self._next_segment_index:08d}.pt'
        self._next_segment_index += 1

        payload = self._pack_segment(transitions)
        import torch

        torch.save(payload, path)
        self._segments.append({'path': str(path), 'count': len(transitions)})
        self._segment_item_count += len(transitions)
        self._pending_cold.clear()

    def _pack_segment(self, transitions):
        """把 transition 段打包，并在段内去重共享的 GraphTensor 对象。"""

        graph_indices = {}
        graphs = []
        records = []

        def add_graph(graph):
            if graph is None:
                return None

            # next_graph 缓存会让相邻 transition 共享同一个 GraphTensor 对象。
            # 这里按对象 id 做段内去重，避免同一状态在冷存储里保存两份。
            graph_key = id(graph)
            if graph_key not in graph_indices:
                graph_indices[graph_key] = len(graphs)
                graphs.append(graph)
            return graph_indices[graph_key]

        for transition in transitions:
            records.append((
                add_graph(transition.graph),
                int(transition.action_offset),
                float(transition.reward),
                add_graph(transition.next_graph),
                bool(transition.terminated),
                bool(transition.truncated),
            ))

        return {
            'version': 1,
            'graphs': tuple(graphs),
            'records': tuple(records),
        }

    def _sample_hot(self, count):
        """从热数据 deque 中采样少量 transition。"""

        if count <= 0:
            return ()
        indices = self._rng.sample(range(len(self._hot_items)), count)
        return [self._hot_items[index] for index in indices]

    def _load_segment(self, segment):
        """读取一个冷数据段，并还原为 TensorTransition 列表。"""

        import torch

        payload = torch.load(segment['path'], map_location='cpu', weights_only=False)
        graphs = payload['graphs']
        transitions = []
        for graph_index, action_offset, reward, next_graph_index, terminated, truncated in payload['records']:
            next_graph = None if next_graph_index is None else graphs[next_graph_index]
            transitions.append(TensorTransition(
                graph=graphs[graph_index],
                action_offset=action_offset,
                reward=reward,
                next_graph=next_graph,
                terminated=terminated,
                truncated=truncated,
            ))
        return transitions

    def _sample_cold(self, count):
        """从 pending 冷数据和冷缓存中随机采样。"""

        if count <= 0:
            return ()

        if self._should_refresh_cold_cache(count):
            self._refresh_cold_cache()

        pool = list(self._pending_cold) + list(self._cold_cache)
        if len(pool) < count:
            # 极端情况下缓存还不够，就按需补读更多段，保证 sample 语义正确。
            for segment in self._shuffled_segments():
                pool.extend(self._load_segment(segment))
                if len(pool) >= count:
                    break

        if count > len(pool):
            raise ValueError(
                f'cannot sample {count} cold items from cold pool with {len(pool)} items'
            )
        return tuple(self._rng.sample(pool, count))

    def _should_refresh_cold_cache(self, requested_count):
        """判断是否需要刷新冷数据采样缓存。"""

        if self._segment_item_count <= 0:
            return False
        if len(self._cold_cache) < requested_count:
            return True
        return self._sample_calls % self.cold_cache_refresh_interval == 1

    def _refresh_cold_cache(self):
        """从随机冷数据段中装载一批 transition 作为采样缓存。"""

        if self.cold_cache_size == 0 or not self._segments:
            self._cold_cache = []
            return

        loaded = []
        for segment in self._shuffled_segments():
            loaded.extend(self._load_segment(segment))
            if len(loaded) >= self.cold_cache_size:
                break

        if len(loaded) > self.cold_cache_size:
            loaded = self._rng.sample(loaded, self.cold_cache_size)
        self._cold_cache = loaded

    def _shuffled_segments(self):
        """返回随机顺序的冷数据段列表。"""

        segments = list(self._segments)
        self._rng.shuffle(segments)
        return segments

    def _remove_segment_file(self, segment):
        """删除一个冷数据段文件。"""

        try:
            Path(segment['path']).unlink()
        except FileNotFoundError:
            pass
