import numpy as np
import open3d as o3d
from typing import List, Tuple
import torch

def aabbs_to_tensor(bboxes: List[o3d.geometry.AxisAlignedBoundingBox]) -> torch.Tensor:
    """Convierte AABBs a un tensor [N,6]: (minx, miny, minz, maxx, maxy, maxz)."""
    arr = np.asarray([
        [*b.get_min_bound(), *b.get_max_bound()]
        for b in bboxes
    ], dtype=np.float32)
    return torch.from_numpy(arr)

def tensor_to_aabbs(bounds: torch.Tensor) -> List[o3d.geometry.AxisAlignedBoundingBox]:
    """Reconstruye AABBs desde un tensor [N,6]."""
    arr = bounds.detach().cpu().numpy()
    out = []
    for i in range(arr.shape[0]):
        mn = arr[i, :3]
        mx = arr[i, 3:6]
        out.append(o3d.geometry.AxisAlignedBoundingBox(mn, mx))
    return out

def _aabb_minmax(aabb: o3d.geometry.AxisAlignedBoundingBox) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(aabb.get_min_bound(), dtype=float), np.asarray(aabb.get_max_bound(), dtype=float)

def _overlap(minA, maxA, minB, maxB, eps: float) -> bool:
    # Collide if intervals overlap on all three axes (touch counts as overlap if eps >= 0)
    return np.all(minA <= maxB + eps) and np.all(minB <= maxA + eps)

def _union_minmax(minA, maxA, minB, maxB):
    return np.minimum(minA, minB), np.maximum(maxA, maxB)

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True

def merge_colliding_aabbs(
    bboxes: List[o3d.geometry.AxisAlignedBoundingBox],
    epsilon: float = 0.0,
    method: str = "sweep",   # "sweep" (recommended) or "naive"
) -> List[o3d.geometry.AxisAlignedBoundingBox]:
    """
    Merge all AABBs that collide (overlap/touch) into union boxes.
    Collision is transitive: if A overlaps B and B overlaps C, A∪B∪C is one merged box.

    Args:
        bboxes: list of Open3D AABBs.
        epsilon: inflate boxes by this amount during overlap tests (useful to merge near-touching boxes).
        method: "sweep" uses a sweep line on x to reduce comparisons; "naive" compares all pairs.

    Returns:
        List of merged Open3D AABBs.
    """
    n = len(bboxes)
    if n == 0:
        return []
    if n == 1:
        return [bboxes[0]]

    mins = np.empty((n, 3), dtype=float)
    maxs = np.empty((n, 3), dtype=float)
    for i, b in enumerate(bboxes):
        mins[i], maxs[i] = _aabb_minmax(b)

    dsu = DSU(n)

    if method == "naive":
        for i in range(n):
            for j in range(i + 1, n):
                if _overlap(mins[i], maxs[i], mins[j], maxs[j], epsilon):
                    dsu.union(i, j)
    else:
        # Sweep-line on X: sort by min_x, keep active boxes whose max_x >= current min_x - eps
        order = np.argsort(mins[:, 0])
        active = []  # list of indices currently overlapping in X
        import bisect

        # We’ll maintain active as a list of (max_x, idx), sorted by max_x, to pop those that end before current min_x - eps
        active_sorted_maxx = []  # parallel list of max_x for binary search

        for k in order:
            minx_k = mins[k, 0]
            # Drop from active all with max_x < minx_k - eps
            cutoff = minx_k - epsilon
            pos = bisect.bisect_left(active_sorted_maxx, cutoff)
            if pos > 0:
                active = active[pos:]
                active_sorted_maxx = active_sorted_maxx[pos:]

            # Compare against remaining active candidates (they overlap in X by construction)
            for maxx_i, i in active:
                if _overlap(mins[i], maxs[i], mins[k], maxs[k], epsilon):
                    dsu.union(i, k)

            # Insert current box into active (keep sorted by max_x)
            insert_pos = bisect.bisect_left(active_sorted_maxx, maxs[k, 0])
            active_sorted_maxx.insert(insert_pos, maxs[k, 0])
            active.insert(insert_pos, (maxs[k, 0], k))

    # Collect components
    comp_map = {}
    for i in range(n):
        root = dsu.find(i)
        comp_map.setdefault(root, []).append(i)

    # Build merged AABBs
    merged = []
    for comp_indices in comp_map.values():
        cmn = mins[comp_indices].min(axis=0)
        cmx = maxs[comp_indices].max(axis=0)
        merged.append(o3d.geometry.AxisAlignedBoundingBox(cmn, cmx))
    return merged

# --- helpers ---------------------------------------------------------------

def aabb_volume(aabb: o3d.geometry.AxisAlignedBoundingBox) -> float:
    ex = np.asarray(aabb.get_extent(), dtype=float)
    return float(np.prod(np.maximum(ex, 0.0)))

def pick_global_spacing_for_target(bboxes, target_points_for_largest: int = 200) -> float:
    """Pick one spacing s so the largest AABB gets ~target_points_for_largest points."""
    vols = [aabb_volume(b) for b in bboxes]
    vmax = max(vols) if vols else 0.0
    if vmax <= 0:
        raise ValueError("All AABBs have zero volume.")
    s = (vmax / float(target_points_for_largest)) ** (1.0 / 3.0)
    return float(s)

def grid_points_in_aabb(aabb: o3d.geometry.AxisAlignedBoundingBox, spacing: float,
                        center_on_voxels: bool = True, include_max: bool = False,
                        dtype=np.float32) -> np.ndarray:
    mn = np.asarray(aabb.get_min_bound(), dtype=float)
    mx = np.asarray(aabb.get_max_bound(), dtype=float)
    ext = mx - mn
    # number of steps per axis (>=1)
    nx = max(1, int(np.floor(ext[0] / spacing)) + (1 if include_max and ext[0] >= spacing else 0))
    ny = max(1, int(np.floor(ext[1] / spacing)) + (1 if include_max and ext[1] >= spacing else 0))
    nz = max(1, int(np.floor(ext[2] / spacing)) + (1 if include_max and ext[2] >= spacing else 0))

    if center_on_voxels:
        xs = (mn[0] + spacing * 0.5) + spacing * np.arange(nx, dtype=dtype)
        ys = (mn[1] + spacing * 0.5) + spacing * np.arange(ny, dtype=dtype)
        zs = (mn[2] + spacing * 0.5) + spacing * np.arange(nz, dtype=dtype)
        xs = xs[xs < mx[0]]; ys = ys[ys < mx[1]]; zs = zs[zs < mx[2]]
    else:
        xs = mn[0] + spacing * np.arange(nx, dtype=dtype)
        ys = mn[1] + spacing * np.arange(ny, dtype=dtype)
        zs = mn[2] + spacing * np.arange(nz, dtype=dtype)
        xs = xs[xs <= mx[0] + 1e-9]; ys = ys[ys <= mx[1] + 1e-9]; zs = zs[zs <= mx[2] + 1e-9]

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(dtype, copy=False)

# --- main: constant-density grids for all AABBs ---------------------------

def grids_for_aabbs_constant_density(bboxes, target_points_for_largest: int = 200,
                                     spacing: float | None = None):
    """
    If spacing is None, compute it from the largest AABB so that it gets ~target_points_for_largest points.
    Then use the same spacing for all boxes -> point count ~ volume, as desired.
    """
    if spacing is None:
        spacing = pick_global_spacing_for_target(bboxes, target_points_for_largest)
    out = []
    for k, b in enumerate(bboxes):
        pts = grid_points_in_aabb(b, spacing=spacing, center_on_voxels=True)
        out.append(pts)
    return spacing, out  # returns spacing and list of (Nk,3) arrays
