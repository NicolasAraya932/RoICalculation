from pathlib import PosixPath, Path

from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from region_of_interest.scripts.aux import find_parent_cwd
from region_of_interest.semantic_pre_processing.bboxes_calculation import BboxCalculation, BboxCalculationConfig
from region_of_interest.scripts.aabb_merge import (
    tensor_to_aabbs,
    aabbs_to_tensor
)

from region_of_interest.semantic_pre_processing.voxel_calculation import (
    VoxelCalculationConfig,
    VoxelCalculation
)

from externals import get_config

from typing import List

import open3d as o3d

import numpy as np

import torch

import yaml

import os

# GLOBAL PARAMETERS
# Paths
CWD : PosixPath = Path().resolve()
CONFIGS = get_config()

PATHS = CONFIGS['PATHS']
CONFIG_PATH : PosixPath = Path(PATHS['CONFIG_PATH'])
DATASETS_PATH : PosixPath = Path(PATHS['DATASETS_PATH'])

TORCH = PATHS['TORCH']
SEMANTIC_FIELD_PT : PosixPath = Path(TORCH['SEMANTIC_FIELD_PT'])
RADIANCE_FIELD_PT : PosixPath = Path(TORCH['RADIANCE_FIELD_PT'])

# Model Parameters
PARAMETERS = CONFIGS['PARAMETERS']
NUM_RAYS_PER_BATCH : int = PARAMETERS['NUM_RAYS_PER_BATCH']

# Bboxes
BBOXES = PATHS['BBOXES']
CANDIDATE_REGIONS : PosixPath = Path(BBOXES['CANDIDATE_REGIONS'])

def set_cwd():
    global CWD
    # Looking for all saved datasets
    data_paths = []
    for root, dirs, files in os.walk(DATASETS_PATH):
        if "transforms.json" in files:
            data_paths.append(Path(os.path.join(root, "transforms.json")).parent)

    config = yaml.load(CONFIG_PATH.read_text(), Loader=yaml.Loader)

    cwd_path = find_parent_cwd(config.data, config.experiment_name, data_paths)

    os.chdir(cwd_path)

    CWD = Path(cwd_path)

def config_initialization():
    # Initializing the config

    from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
    from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
    from nerfstudio.utils.eval_utils import eval_setup
    config, pipeline, checkpoint_path, step = eval_setup(CONFIG_PATH)

    # Checking the pipeline and datamanager
    assert isinstance(
        pipeline.datamanager,
        (VanillaDataManager, ParallelDataManager),
    )
    if isinstance(pipeline.datamanager, VanillaDataManager):
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = NUM_RAYS_PER_BATCH

    return config, pipeline, checkpoint_path, step

def calculate_voxels() -> List[np.ndarray]:
    # Setting the current working directory
    progress = Progress(
        TextColumn(":cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )

    bboxes = tensor_to_aabbs(torch.load(CANDIDATE_REGIONS))
    VoxelConfig = VoxelCalculationConfig()

    voxel_grids = []
    with progress:
        task = progress.add_task("Calculating Voxel Grids...", total=len(bboxes))
        for bbox in bboxes:
            voxel_calculation = VoxelCalculation(bbox, VoxelConfig)
            voxel_grid = voxel_calculation.calculate_voxelgrid()
            voxel_grids.append(voxel_grid)
            progress.update(task, advance=1)

    positions: List[np.ndarray] = []
    for grid in voxel_grids:
        assert isinstance(grid, o3d.geometry.VoxelGrid)
        voxels = grid.get_voxels()  # list[o3d.geometry.Voxel]

        centers = [grid.get_voxel_center_coordinate(v.grid_index) for v in voxels]  # <-- new name
        positions.append(np.asarray(centers, dtype=np.float64))                     # <-- append array

    return positions


def main(positions: List[np.ndarray]) -> None:

    from nerfstudio.cameras.rays import RayBundle
    from nerfstudio.pipelines.base_pipeline import Pipeline
    from nerfstudio.model_components.ray_samplers import Frustums, RaySamples
    set_cwd()
    config, pipeline, checkpoint_path, step = config_initialization()


    # Load the field
    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline.model, "field")
    field = pipeline.model.field
    assert hasattr(field, "density_fn")
    assert callable(field.density_fn)

    device = pipeline.device
    thr = torch.tensor(20000.0, device=device)          # make once

    kept_points = []                                   # numpy chunks back on CPU
    with torch.no_grad():
        for cloud in positions:                        # cloud: (N,3) numpy
            for point in cloud:
                pts = torch.as_tensor(point, dtype=torch.float32, device=device)
                sigma = field.density_fn(pts).squeeze(-1)  # (N,) densities

                mask = sigma > thr                         # (N,) boolean
                if mask.any():
                    kept_points.append(pts[mask].detach().cpu().numpy())

    all_pts = np.concatenate(kept_points, axis=0).astype(np.float64, copy=False) # (N_total,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)  # Vector3dVector expects float64
    o3d.io.write_point_cloud(
        "/workspace/Desktop/RoICalculation/region_of_interest/region_of_interest/points.ply",
        pcd
    )

    # #field.get_density()
    # ray_bundle, _ = pipeline.datamanager.next_train(0)
    # assert isinstance(ray_bundle, RayBundle)
    # outputs = pipeline.model(ray_bundle)

    # print([ray_bundle.__getattribute__(key).shape if isinstance(ray_bundle.__getattribute__(key), torch.Tensor) else None for key in ray_bundle.__dict__.keys()])

    # #print([ray_bundle[keys].shape for keys in ray_bundle.__dict__.keys()])
    

if __name__ == '__main__':

    positions = calculate_voxels()
    main(positions)