"""GPU 全场圆障碍加权 Voronoi 图原型。

该模块把每个活动水果视为圆形障碍物，并把桶的左壁、右壁和底部视为
边界障碍物。水果站点使用到圆表面的有符号距离 ``||p-c||-r``，边界站点
使用到内壁的距离。顶部只裁剪计算域，不作为封闭障碍物。

第一版以规则采样距离场恢复图边和交汇点。规则采样仅是 GPU 上构造完整
图的数值手段，不会作为模型节点或场景实验室的可视化节点输出。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Lock
import time

import torch

from .config import SimulatorConfig


@dataclass(frozen=True, slots=True)
class WeightedVoronoiGraph:
    """单个环境的紧凑 Voronoi 图，所有张量均保留在构建设备上。"""

    edge_start: torch.Tensor
    edge_end: torch.Tensor
    edge_clearance: torch.Tensor
    edge_owners: torch.Tensor
    vertex_position: torch.Tensor
    vertex_clearance: torch.Tensor
    vertex_owners: torch.Tensor
    visible_site_samples: torch.Tensor
    free_sample_count: torch.Tensor
    sample_count: int


class WeightedVoronoiGraphBuilder:
    """在 CPU 或 CUDA 上批量构造完整的圆障碍加权 Voronoi 图。"""

    BOUNDARY_NAMES = ('left_wall', 'right_wall', 'floor')

    def __init__(
            self,
            config: SimulatorConfig,
            *,
            device='cpu',
            sample_spacing=4.0,
            point_chunk_size=65536):
        self.config = config
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available for Voronoi construction')
        self.sample_spacing = float(sample_spacing)
        if not math.isfinite(self.sample_spacing) or self.sample_spacing <= 0:
            raise ValueError('sample_spacing must be positive and finite')
        self.point_chunk_size = int(point_chunk_size)
        if self.point_chunk_size <= 0:
            raise ValueError('point_chunk_size must be positive')

        self.left = float(config.wall_width)
        self.right = float(config.board_width - config.wall_width)
        self.top = 0.0
        self.floor = float(config.board_height - config.wall_width)
        self._x = self._axis(self.left, self.right)
        self._y = self._axis(self.top, self.floor)
        grid_y, grid_x = torch.meshgrid(self._y, self._x, indexing='ij')
        self._grid_points = torch.stack((grid_x, grid_y), dim=-1)
        self._points = self._grid_points.reshape(-1, 2)

        center_x = (self._x[:-1] + self._x[1:]) * 0.5
        center_y = (self._y[:-1] + self._y[1:]) * 0.5
        cell_y, cell_x = torch.meshgrid(center_y, center_x, indexing='ij')
        self._cell_centers = torch.stack((cell_x, cell_y), dim=-1)

        top_mid = torch.stack((cell_x, grid_y[:-1, :-1]), dim=-1)
        right_mid = torch.stack((grid_x[:-1, 1:], cell_y), dim=-1)
        bottom_mid = torch.stack((cell_x, grid_y[1:, :-1]), dim=-1)
        left_mid = torch.stack((grid_x[:-1, :-1], cell_y), dim=-1)
        self._side_positions = torch.stack(
            (top_mid, right_mid, bottom_mid, left_mid), dim=-2
        )

    def _axis(self, start, end):
        intervals = max(1, math.ceil((end - start) / self.sample_spacing))
        return torch.linspace(
            start,
            end,
            intervals + 1,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def bounds(self):
        return {
            'left': self.left,
            'right': self.right,
            'top': self.top,
            'floor': self.floor,
        }

    @property
    def raster_shape(self):
        return int(self._y.numel()), int(self._x.numel())

    def _nearest_sites(self, points, positions, radii, active):
        batch_size, fruit_count = positions.shape[:2]
        site_work = max(1, batch_size * max(1, fruit_count))
        chunk_size = min(
            self.point_chunk_size,
            max(1, 8_388_608 // site_work),
        )
        owner_chunks = []
        clearance_chunks = []
        for start in range(0, points.shape[0], chunk_size):
            chunk = points[start:start + chunk_size]
            if fruit_count:
                delta = chunk[None, :, None, :] - positions[:, None, :, :]
                fruit_distance = torch.sqrt(
                    delta.square().sum(dim=-1).clamp_min_(1e-12)
                ) - radii[:, None, :]
                fruit_distance = fruit_distance.masked_fill(
                    ~active[:, None, :], torch.inf
                )
            else:
                fruit_distance = torch.empty(
                    (batch_size, chunk.shape[0], 0),
                    dtype=torch.float32,
                    device=self.device,
                )
            boundary_distance = torch.stack((
                chunk[:, 0] - self.left,
                self.right - chunk[:, 0],
                self.floor - chunk[:, 1],
            ), dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
            distances = torch.cat((fruit_distance, boundary_distance), dim=-1)
            nearest, owners = torch.min(distances, dim=-1)
            owner_chunks.append(owners)
            clearance_chunks.append(nearest)
        return torch.cat(owner_chunks, dim=1), torch.cat(
            clearance_chunks, dim=1
        )

    def build(self, positions, radii, active=None):
        """构造一批图。

        ``positions`` 为 ``[B,N,2]``，``radii`` 和 ``active`` 为 ``[B,N]``。
        不进行站点裁剪；每个 active 水果及三条封闭桶边界都会参与最近站点
        竞争。
        """

        positions = torch.as_tensor(
            positions, dtype=torch.float32, device=self.device
        )
        radii = torch.as_tensor(radii, dtype=torch.float32, device=self.device)
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError('positions must have shape [B, N, 2]')
        if radii.shape != positions.shape[:2]:
            raise ValueError('radii must have shape [B, N]')
        if active is None:
            active = torch.ones_like(radii, dtype=torch.bool)
        else:
            active = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        if active.shape != radii.shape:
            raise ValueError('active must have shape [B, N]')
        if torch.any(radii[active] <= 0):
            raise ValueError('active fruit radii must be positive')

        batch_size, fruit_count = positions.shape[:2]
        height, width = self.raster_shape
        owners, clearance = self._nearest_sites(
            self._points, positions, radii, active
        )
        _center_owners, center_clearance = self._nearest_sites(
            self._cell_centers.reshape(-1, 2), positions, radii, active
        )
        owners = owners.reshape(batch_size, height, width)
        clearance = clearance.reshape(batch_size, height, width)
        center_clearance = center_clearance.reshape(
            batch_size, height - 1, width - 1
        )

        graphs = []
        site_count = fruit_count + len(self.BOUNDARY_NAMES)
        tolerance = self.sample_spacing * 0.55
        pair_indices = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        for batch_index in range(batch_size):
            owner = owners[batch_index]
            clear = clearance[batch_index]
            tl, tr = owner[:-1, :-1], owner[:-1, 1:]
            br, bl = owner[1:, 1:], owner[1:, :-1]
            ctl, ctr = clear[:-1, :-1], clear[:-1, 1:]
            cbr, cbl = clear[1:, 1:], clear[1:, :-1]

            side_owner_pairs = torch.stack((
                torch.stack((tl, tr), dim=-1),
                torch.stack((tr, br), dim=-1),
                torch.stack((bl, br), dim=-1),
                torch.stack((tl, bl), dim=-1),
            ), dim=-2).sort(dim=-1).values
            side_clearance = torch.stack((
                (ctl + ctr) * 0.5,
                (ctr + cbr) * 0.5,
                (cbl + cbr) * 0.5,
                (ctl + cbl) * 0.5,
            ), dim=-1)
            side_crossing = (
                side_owner_pairs[..., 0] != side_owner_pairs[..., 1]
            ) & (side_clearance >= -tolerance)
            crossing_count = side_crossing.sum(dim=-1)

            sorted_corners = torch.stack((tl, tr, br, bl), dim=-1).sort(
                dim=-1
            ).values
            unique_count = 1 + (
                sorted_corners[..., 1:] != sorted_corners[..., :-1]
            ).sum(dim=-1)
            direct_cell = (unique_count == 2) & (crossing_count == 2)
            center_clear = center_clearance[batch_index]
            hub_cell = (
                (crossing_count >= 2)
                & ~direct_cell
                & (center_clear >= -tolerance)
            )

            edge_starts = []
            edge_ends = []
            edge_clearances = []
            edge_owners = []
            for first, second in pair_indices:
                mask = (
                    direct_cell
                    & side_crossing[..., first]
                    & side_crossing[..., second]
                )
                edge_starts.append(self._side_positions[..., first, :][mask])
                edge_ends.append(self._side_positions[..., second, :][mask])
                edge_clearances.append(torch.minimum(
                    side_clearance[..., first][mask],
                    side_clearance[..., second][mask],
                ).clamp_min_(0.0))
                edge_owners.append(side_owner_pairs[..., first, :][mask])
            for side in range(4):
                mask = hub_cell & side_crossing[..., side]
                edge_starts.append(self._cell_centers[mask])
                edge_ends.append(self._side_positions[..., side, :][mask])
                edge_clearances.append(torch.minimum(
                    center_clear[mask], side_clearance[..., side][mask]
                ).clamp_min_(0.0))
                edge_owners.append(side_owner_pairs[..., side, :][mask])

            edge_start = torch.cat(edge_starts, dim=0)
            edge_end = torch.cat(edge_ends, dim=0)
            edge_clearance = torch.cat(edge_clearances, dim=0)
            edge_owner = torch.cat(edge_owners, dim=0)

            vertex_mask = (
                (unique_count >= 3)
                & (crossing_count >= 2)
                & (center_clear >= -tolerance)
            )
            vertex_position = self._cell_centers[vertex_mask]
            vertex_clearance = center_clear[vertex_mask].clamp_min_(0.0)
            vertex_owners = sorted_corners[vertex_mask]

            flat_owner = owner.reshape(-1)
            flat_free = clear.reshape(-1) >= 0.0
            visible_site_samples = torch.bincount(
                flat_owner[flat_free], minlength=site_count
            )
            graphs.append(WeightedVoronoiGraph(
                edge_start=edge_start,
                edge_end=edge_end,
                edge_clearance=edge_clearance,
                edge_owners=edge_owner,
                vertex_position=vertex_position,
                vertex_clearance=vertex_clearance,
                vertex_owners=vertex_owners,
                visible_site_samples=visible_site_samples,
                free_sample_count=flat_free.sum(),
                sample_count=int(flat_free.numel()),
            ))
        return tuple(graphs)


class ScenarioVoronoiEvaluator:
    """场景实验室使用的线程安全、单场景缓存的 Voronoi 适配器。"""

    def __init__(self, config, *, device='cpu', sample_spacing=4.0):
        self.device = torch.device(device)
        self.builder = WeightedVoronoiGraphBuilder(
            config,
            device=self.device,
            sample_spacing=sample_spacing,
        )
        self._lock = Lock()
        self._cache_key = None
        self._cache_payload = None

    @staticmethod
    def _key(scene):
        return tuple(
            (
                fruit['id'], fruit['level'], fruit['physics_radius'],
                fruit['x'], fruit['y'],
            )
            for fruit in scene['fruits']
        )

    def evaluate(self, raw_scene):
        # 延迟导入避免场景服务和本模块形成导入环。
        from .scenario_lab_service import validate_scenario

        scene = validate_scenario(raw_scene)
        cache_key = self._key(scene)
        with self._lock:
            if cache_key == self._cache_key and self._cache_payload is not None:
                return {**self._cache_payload, 'cache_hit': True}
            fruits = scene['fruits']
            positions = torch.tensor(
                [[(fruit['x'], fruit['y']) for fruit in fruits]],
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, len(fruits), 2)
            radii = torch.tensor(
                [[fruit['physics_radius'] for fruit in fruits]],
                dtype=torch.float32,
                device=self.device,
            )
            active = torch.ones_like(radii, dtype=torch.bool)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            graph = self.builder.build(positions, radii, active)[0]
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            compute_ms = (time.perf_counter() - started) * 1000.0
            payload = self._payload(scene, graph, compute_ms)
            self._cache_key = cache_key
            self._cache_payload = payload
            return payload

    def _payload(self, scene, graph, compute_ms):
        fruit_count = len(scene['fruits'])
        visible = graph.visible_site_samples.detach().cpu().tolist()
        sites = []
        for index, fruit in enumerate(scene['fruits']):
            sites.append({
                'index': index,
                'kind': 'fruit',
                'fruit_id': fruit['id'],
                'level': fruit['level'],
                'x': round(float(fruit['x']), 4),
                'y': round(float(fruit['y']), 4),
                'radius': round(float(fruit['physics_radius']), 4),
                'visible_samples': int(visible[index]),
            })
        for offset, name in enumerate(self.builder.BOUNDARY_NAMES):
            sites.append({
                'index': fruit_count + offset,
                'kind': name,
                'visible_samples': int(visible[fruit_count + offset]),
            })

        starts = graph.edge_start.detach().cpu().tolist()
        ends = graph.edge_end.detach().cpu().tolist()
        clearances = graph.edge_clearance.detach().cpu().tolist()
        owners = graph.edge_owners.detach().cpu().tolist()
        edges = [{
            'x1': round(float(start[0]), 3),
            'y1': round(float(start[1]), 3),
            'x2': round(float(end[0]), 3),
            'y2': round(float(end[1]), 3),
            'clearance': round(float(clearance), 3),
            'owners': [int(owner[0]), int(owner[1])],
        } for start, end, clearance, owner in zip(
            starts, ends, clearances, owners
        )]

        vertex_positions = graph.vertex_position.detach().cpu().tolist()
        vertex_clearances = graph.vertex_clearance.detach().cpu().tolist()
        vertex_owners = graph.vertex_owners.detach().cpu().tolist()
        vertices = []
        for position, clearance, owner_row in zip(
                vertex_positions, vertex_clearances, vertex_owners):
            vertices.append({
                'x': round(float(position[0]), 3),
                'y': round(float(position[1]), 3),
                'clearance': round(float(clearance), 3),
                'owners': sorted({int(owner) for owner in owner_row}),
            })

        edge_values = [edge['clearance'] for edge in edges]
        return {
            'format_version': 1,
            'algorithm': 'full_disk_weighted_voronoi_raster_v1',
            'sampled': True,
            'sample_spacing': self.builder.sample_spacing,
            'raster_shape': list(self.builder.raster_shape),
            'device': str(self.device),
            'compute_ms': round(compute_ms, 3),
            'cache_hit': False,
            'bounds': self.builder.bounds,
            'sites': sites,
            'edges': edges,
            'vertices': vertices,
            'stats': {
                'fruit_site_count': fruit_count,
                'boundary_site_count': len(self.builder.BOUNDARY_NAMES),
                'visible_fruit_site_count': sum(
                    site['visible_samples'] > 0 for site in sites[:fruit_count]
                ),
                'edge_count': len(edges),
                'vertex_count': len(vertices),
                'free_sample_count': int(graph.free_sample_count.item()),
                'sample_count': graph.sample_count,
                'free_sample_ratio': round(
                    int(graph.free_sample_count.item())
                    / max(1, graph.sample_count), 6
                ),
                'min_edge_clearance': min(edge_values, default=0.0),
                'max_edge_clearance': max(edge_values, default=0.0),
            },
        }
