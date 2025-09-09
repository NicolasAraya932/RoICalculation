# Libraries
import open3d as o3d
import numpy as np

def from_voxel_to_point_cloud(voxel_grid : o3d.geometry.VoxelGrid) -> o3d.geometry.PointCloud:

    voxels = voxel_grid.get_voxels()

    voxels_bounding_points = o3d.geometry.PointCloud()
    voxels_bounding_points.points = o3d.utility.Vector3dVector()
    voxels_bounding_points.colors = o3d.utility.Vector3dVector()

    for v in voxels:
        voxels_bounding_points.points.extend(voxel_grid.get_voxel_bounding_points(v.grid_index))
        # if color: Color for each 8 bounding points of the current voxel
        if v.color is not None:
            voxels_bounding_points.colors.extend(np.asarray([v.color for _ in range(8)]))

    return voxels_bounding_points

def voxelgrid_centers_pointcloud(voxel_grid: o3d.geometry.VoxelGrid) -> o3d.geometry.PointCloud:
    voxels = voxel_grid.get_voxels()
    if not voxels:
        return o3d.geometry.PointCloud()

    idx = np.asarray([v.grid_index for v in voxels], dtype=np.int64)      # (N,3)
    col = np.asarray([v.color for v in voxels], dtype=np.float32)         # (N,3) in [0,1]

    vs = float(voxel_grid.voxel_size)
    origin = np.asarray(voxel_grid.origin, dtype=np.float32)

    centers = origin[None, :] + (idx.astype(np.float32) + 0.5) * vs       # (N,3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.colors = o3d.utility.Vector3dVector(col)
    return pcd