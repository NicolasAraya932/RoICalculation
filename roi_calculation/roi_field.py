"""
Copying the nerfacto field definition for RoICalculation.

The main objective is to obtain the checkpoints of the last training
of nerfacto and use them to calculate the ROI (Region of Interest)
for the next training.

Methodology:

1. Define with Semantic Is Enough the Region of Interest (ROI) for the next training.
2. Use the checkpoints from the last training to calculate the ROI.
    - The points within the ROI are defined as a uniform grid of points.
    - This grid is concatenated with the default positions of the nerfacto field.
    - The grid is used to calculate the density values.
    - Using the positions we can define the directions and locations of the rays to 
      calculate the RGB values.
3. Render a new scene and populate with more shading points and rays the defined ROI.
    - This is done by using the nerfacto field to render the scene with the new positions.
    - The new scene is rendered with more details in the ROI.
4. Use the new scene to train the nerfacto model.
5. Repeat the process until the ROI is fully defined and the model is trained to satisfaction.

Result:

A field that has renderized with more details the ROI's. This means every fruit within the ROI are
renderized with more shading points and rays, leading to a more detailed and accurate model only for
specific areas of interest.


"""

from typing import Dict, Literal, Optional, Tuple
import os

import torch
from torch import Tensor, nn

import open3d as o3d
import numpy as np
from open3d.geometry import AxisAlignedBoundingBox
from attachment.scripts.aabb_merge import tensor_to_aabbs
from roi_calculation.scripts.bboxes_ import filter_points_by_bboxes

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

from nerfstudio.fields.nerfacto_field import NerfactoField


class RoiField(NerfactoField):
    """Compound Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        num_nerf_samples_per_ray: int = 48,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__(
            aabb=aabb,
            num_images=num_images,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            geo_feat_dim=geo_feat_dim,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            num_layers_transient=num_layers_transient,
            features_per_level=features_per_level,
            hidden_dim_color=hidden_dim_color,
            hidden_dim_transient=hidden_dim_transient,
            appearance_embedding_dim=appearance_embedding_dim,
            transient_embedding_dim=transient_embedding_dim,
            use_transient_embedding=use_transient_embedding,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            average_init_density=average_init_density,
            implementation=implementation,
        )

        self.bboxes = tensor_to_aabbs(torch.load("/workspace/Desktop/RoICalculation/ground_truth/candidate_regions/Roubboxes.pt"))
        self.grids  = torch.load("/workspace/Desktop/RoICalculation/ground_truth/candidate_regions/grids.pt")

        self.num_samples_per_ray = num_nerf_samples_per_ray

        # all child modules (by name)
        for name, module in self.named_children():
            print("module:", name, module)

        # all registered buffers
        for name, buf in self.named_buffers():
            print("buffer:", name, buf)

        # all parameters
        for name, param in self.named_parameters():
            print("param:", name, param.shape)



    def get_density(self, ray_samples: RaySamples, step: int = 0) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        positions_shape = positions.shape
        positions_reshape = positions.detach().numpy().reshape(-1,3)
        pcd_positions = o3d.utility.Vector3dVector(positions_reshape)
        positions_filtered = filter_points_by_bboxes(pcd_positions, self.bboxes)
        positions_filtered = positions_filtered.reshape(positions_shape)

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)

        """
        This is where we have to add the new positions for the ROI.
        """
        # Extraction of positions
        if step > 1 and step < 500:
            print("Saving early positions... step:", step)

            os.makedirs("/workspace/Desktop/RoICalculation/early/positions/raw", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/early/positions/normalized", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/early/positions/encoded", exist_ok=True)

            torch.save(ray_samples.frustums.get_positions().cpu(), f"/workspace/Desktop/RoICalculation/early/positions/raw/positions_{step}.pt")
            torch.save(SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb).cpu(), f"/workspace/Desktop/RoICalculation/early/positions/normalized/positions_{step}.pt")
            torch.save(self.position_encoding(ray_samples.frustums.get_positions().view(-1, 3)).cpu(), f"/workspace/Desktop/RoICalculation/early/positions/encoded/positions_{step}.pt")

        if step > 29500:
            print("Saving positions... step:", step)

            os.makedirs("/workspace/Desktop/RoICalculation/positions/raw", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/positions/normalized", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/positions/encoded", exist_ok=True)

            torch.save(ray_samples.frustums.get_positions().cpu(), f"/workspace/Desktop/RoICalculation/positions/raw/positions_{step}.pt")
            torch.save(SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb).cpu(), f"/workspace/Desktop/RoICalculation/positions/normalized/positions_{step}.pt")
            torch.save(self.position_encoding(ray_samples.frustums.get_positions().view(-1, 3)).cpu(), f"/workspace/Desktop/RoICalculation/positions/encoded/positions_{step}.pt")

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]

        assert positions.numel() > 0, "positions is empty."

        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)

        assert positions_flat.numel() > 0, "positions_flat is empty."
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None, step: int = 0
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)

        if step > 1 and step < 500:
            print("Saving early directions... step:", self.step)

            os.makedirs("/workspace/Desktop/RoICalculation/early/starts/", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/early/ends/", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/early/origins/", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/early/directions/raw", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/early/directions/normalized", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/early/directions/encoded", exist_ok=True)

            torch.save(ray_samples.frustums.ends.cpu(), f"/workspace/Desktop/RoICalculation/early/starts/ray_ends_{step}.pt")
            torch.save(ray_samples.frustums.starts.cpu(), f"/workspace/Desktop/RoICalculation/early/ends/ray_starts_{step}.pt")
            torch.save(ray_samples.frustums.origins.cpu(), f"/workspace/Desktop/RoICalculation/early/origins/ray_origins_{step}.pt")
            torch.save(ray_samples.frustums.directions.cpu(), f"/workspace/Desktop/RoICalculation/early/directions/raw/directions_{step}.pt")
            torch.save(directions.cpu(), f"/workspace/Desktop/RoICalculation/early/directions/normalized/directions_{step}.pt")
            torch.save(self.direction_encoding(directions.view(-1, 3)).cpu(), f"/workspace/Desktop/RoICalculation/early/directions/encoded/directions_{step}.pt")


        if step > 29500:
            # Extraction of directions
            print("Saving directions... step:", self.step)

            os.makedirs("/workspace/Desktop/RoICalculation/starts/", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/ends/", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/origins/", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/directions/raw", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/directions/normalized", exist_ok=True)
            os.makedirs("/workspace/Desktop/RoICalculation/directions/encoded", exist_ok=True)

            torch.save(ray_samples.frustums.ends.cpu(), f"/workspace/Desktop/RoICalculation/starts/ray_ends_{step}.pt")
            torch.save(ray_samples.frustums.starts.cpu(), f"/workspace/Desktop/RoICalculation/ends/ray_starts_{step}.pt")
            torch.save(ray_samples.frustums.origins.cpu(), f"/workspace/Desktop/RoICalculation/origins/ray_origins_{step}.pt")
            torch.save(ray_samples.frustums.directions.cpu(), f"/workspace/Desktop/RoICalculation/directions/raw/directions_{step}.pt")
            torch.save(directions.cpu(), f"/workspace/Desktop/RoICalculation/directions/normalized/directions_{step}.pt")
            torch.save(self.direction_encoding(directions.view(-1, 3)).cpu(), f"/workspace/Desktop/RoICalculation/directions/encoded/directions_{step}.pt")



        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
            ]
            + (
                [embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []
            ),
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
