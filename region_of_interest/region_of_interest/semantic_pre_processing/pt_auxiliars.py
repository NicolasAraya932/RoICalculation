# Libraries
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

def bbox_edges_from_corners(corners):
    # 12 edges of a box, each as a pair of indices into the corners array
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    return np.array([[corners[i], corners[j]] for i, j in edges])

def cluster_and_filter(points: np.ndarray, eps: float, min_samples: int):
    """
    1) Run DBSCAN on `points` with (eps, min_samples).
    2) Throw away any resulting cluster whose size < min_cluster_size.
    Returns: a list of (cluster_label, indices), and a final boolean mask.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_  # -1 means noise
    unique_labels = set(labels)
    
    cluster_indices = []
    keep_mask = np.zeros(points.shape[0], dtype=bool)
    
    for lbl in unique_labels:
        if lbl == -1:
            continue  # skip DBSCAN noise
        idx = np.where(labels == lbl)[0]
        cluster_indices.append((lbl, idx))
        keep_mask[idx] = True
    
    return cluster_indices, keep_mask

def agglomerative(points: np.ndarray, n_clusters: int, distance_threshold: float = None):
    """
    Perform agglomerative clustering on the points.
    Returns a list of cluster labels and the clustered points.
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold, compute_full_tree=True)
    labels = clustering.fit_predict(points)
    
    # Create a mask for the clustered points
    unique_labels = np.unique(labels)
    clustered_points = [points[labels == label] for label in unique_labels]
    
    return unique_labels, clustered_points

def density_filter(points: np.ndarray, r: float, k: int):
    """
    Keep only points that have at least k neighbors within radius r.
    Returns a boolean mask and the filtered points.
    """
    nbrs = NearestNeighbors(radius=r, algorithm="auto").fit(points)
    idx_radius = nbrs.radius_neighbors(points, return_distance=False)
    # exclude self by subtracting 1
    neighbor_counts = np.array([len(idxs) - 1 for idxs in idx_radius])
    keep_mask = neighbor_counts >= k
    return keep_mask, points[keep_mask]


def merge_aabbs_inplace(bboxes: list[o3d.geometry.AxisAlignedBoundingBox]
                        ) -> o3d.geometry.AxisAlignedBoundingBox:
    if not bboxes:
        raise ValueError("Empty list of AABBs.")

    acc = o3d.geometry.AxisAlignedBoundingBox(
        bboxes[0].get_min_bound(), bboxes[0].get_max_bound()
    )
    for b in bboxes[1:]:
        acc += b  # expands acc to include b
    return acc