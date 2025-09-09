from pathlib import PosixPath, Path
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.model_components.ray_samplers import Frustums, RaySamples

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager

from nerfstudio.utils.eval_utils import eval_setup

from region_of_interest.scripts.aux import find_parent_cwd
from region_of_interest.semantic_pre_processing.bboxes_calculation import BboxCalculation, BboxCalculationConfig
from externals import get_config

from typing import List

import open3d as o3d

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

def get_bboxes(semantic_field_pt: PosixPath) -> List[o3d.geometry.AxisAlignedBoundingBox]:
    config = BboxCalculationConfig()
    bbox_calculator = BboxCalculation(semantic_field_pt, config)
    bboxes = bbox_calculator.calculate_bboxes()
    return bboxes


def main(bboxes : List[o3d.geometry.AxisAlignedBoundingBox] = None) -> None:
    # Setting the current working directory
    set_cwd()
    config, pipeline, checkpoint_path, step = config_initialization()


    # # Load the field
    # assert isinstance(pipeline, Pipeline)
    # assert hasattr(pipeline.model, "field")
    # field = pipeline.model.field
    # assert hasattr(field, "density_fn")
    # assert callable(field.density_fn)
    # # Testing the field
    # print(field.density_fn(torch.Tensor([0.004, 0.004, 0.04]).unsqueeze(0)))


    # #field.get_density()
    # ray_bundle, _ = pipeline.datamanager.next_train(0)
    # assert isinstance(ray_bundle, RayBundle)
    # outputs = pipeline.model(ray_bundle)

    # print([ray_bundle.__getattribute__(key).shape if isinstance(ray_bundle.__getattribute__(key), torch.Tensor) else None for key in ray_bundle.__dict__.keys()])

    # #print([ray_bundle[keys].shape for keys in ray_bundle.__dict__.keys()])
    

if __name__ == '__main__':

    main()