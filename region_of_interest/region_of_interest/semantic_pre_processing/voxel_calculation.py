import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import Type, Optional

@dataclass
class VoxelCalculationConfig:
    _target: Type = field(default=None)  # optional, not needed at runtime
    target_voxels: int = 10000            # desired total cells
    max_voxels: int = 2_000_000          # safety cap (tune as needed)
    snap: Optional[int] = None           # e.g., 8 or 16 to snap Dx,Dy,Dz

class VoxelCalculation:
    def __init__(self, bbox: o3d.geometry.AxisAlignedBoundingBox, config: VoxelCalculationConfig):
        self.bbox = bbox
        self.config = config

    def _dims_from_target(self, Lx: float, Ly: float, Lz: float):
        V = float(Lx * Ly * Lz)
        N = max(1, int(self.config.target_voxels))
        s = (V / N) ** (1.0 / 3.0)              # cubic spacing
        Dx, Dy, Dz = np.ceil([Lx/s, Ly/s, Lz/s]).astype(int)

        # optional snapping to nicer multiples
        if self.config.snap:
            snap = int(self.config.snap)
            Dx = int(np.ceil(Dx / snap) * snap)
            Dy = int(np.ceil(Dy / snap) * snap)
            Dz = int(np.ceil(Dz / snap) * snap)

        # enforce safety cap by inflating s uniformly
        total = int(Dx) * int(Dy) * int(Dz)
        if total > self.config.max_voxels:
            scale = (total / self.config.max_voxels) ** (1.0 / 3.0)
            s *= scale
            Dx, Dy, Dz = np.ceil([Lx/s, Ly/s, Lz/s]).astype(int)
            total = int(Dx) * int(Dy) * int(Dz)

        return s, int(Dx), int(Dy), int(Dz)

    def calculate_voxelgrid(self) -> o3d.geometry.VoxelGrid:
        extent = self.bbox.get_extent().astype(float)
        Lx, Ly, Lz = map(float, extent)
        s, Dx, Dy, Dz = self._dims_from_target(Lx, Ly, Lz)

        # Open3D expects the min corner as origin for create_dense
        origin = self.bbox.get_min_bound().astype(float)

        return o3d.geometry.VoxelGrid.create_dense(
            origin=origin,
            voxel_size=float(s),
            width=float(Lx),
            height=float(Ly),
            depth=float(Lz),
            color=np.array([0.0, 0.0, 0.0], dtype=float),
        )
