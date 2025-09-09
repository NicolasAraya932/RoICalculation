# Libraries
import open3d as o3d
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from dataclasses import dataclass, field
from pathlib import Path, PosixPath
from typing import Type, List, Tuple, Optional

@dataclass
class BboxCalculationConfig:

    _target: Type = field(default_factory=lambda: BboxCalculation)
    percentile : float = 98.0
    k : int = 7
    min_samples : int = 100   # DBSCAN’s “core point” requirement (often same as k)
    eps : Optional[float] = None  # if None, will be computed via eps_calculation()

class BboxCalculation:

    config : BboxCalculationConfig

    def __init__(self, semantic_field_pt : PosixPath, config : BboxCalculationConfig) -> None:
        self.config = config
        self.semantic_field_pt = semantic_field_pt
        self.semantic_field = torch.load(semantic_field_pt)
        self.semantic_field_points = self.semantic_field['points'].cpu().numpy()

        self.k = self.config.k
        self.min_samples = self.config.min_samples
        if self.config.eps is None:
            self.eps = self.eps_calculation()
        else:
            self.eps = self.config.eps

    def eps_calculation(self) -> float:

        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto').fit(self.semantic_field_points)
        distances, _ = nbrs.kneighbors(self.semantic_field_points)
        # distances[:, 0] is zero (distance to itself), so take dist[:,1]...dist[:,k]
        farthest_distance = distances[:, self.k] * self.k  # nearest neighbor distance

        return np.percentile(farthest_distance, self.config.percentile)

    def cluster_and_filter(self, points: np.ndarray, eps: float, min_samples: int):
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
    
    def calculate_bboxes(self) -> List[o3d.geometry.AxisAlignedBoundingBox]:

        points = self.semantic_field_points

        clusters, keep_clusters_mask = self.cluster_and_filter(
            self.semantic_field_points,
            self.eps,
            self.min_samples
            )
        print(f"Found {len(clusters)} clusters")

        bboxes : List[o3d.geometry.AxisAlignedBoundingBox] = []

        for cidx, idx in clusters:
            # idx is an array of indices into filtered_pts
            cluster_points = o3d.utility.Vector3dVector(points[idx])

            # compute bounding sphere (center, radius)
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(cluster_points)

            # Saving as points
            bboxes.append(bbox)
        
        return bboxes
