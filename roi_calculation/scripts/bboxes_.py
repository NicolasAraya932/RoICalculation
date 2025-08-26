import numpy as np
import open3d as o3d
from typing import List

def filter_points_by_bboxes(points: o3d.utility.Vector3dVector, bboxes: List[o3d.geometry.AxisAlignedBoundingBox]) -> o3d.utility.Vector3dVector:
    points_np = np.asarray(points)
    N = points_np.shape[0]
    points_index = []
    for bbox in bboxes:
        idx = bbox.get_point_indices_within_bounding_box(points)
        points_index.extend(idx)
    # Remove duplicates and out-of-bounds indices
    points_index = list(set(points_index))
    points_index = [i for i in points_index if i < N]
    # Inverse mask
    inverse_mask = np.ones(N, dtype=bool)
    inverse_mask[points_index] = False
    filtered_points = points_np[inverse_mask]
    return o3d.utility.Vector3dVector(filtered_points)