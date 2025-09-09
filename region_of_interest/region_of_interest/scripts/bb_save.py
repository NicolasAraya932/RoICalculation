import json
import numpy as np
import open3d as o3d

def save_aabb_json(aabb: o3d.geometry.AxisAlignedBoundingBox, path: str) -> None:
    data = {
        "type": "AABB",
        "min_bound": aabb.get_min_bound().tolist(),
        "max_bound": aabb.get_max_bound().tolist(),
    }
    json.dump(data, open(path, "w"))

def save_obb_json(obb: o3d.geometry.OrientedBoundingBox, path: str) -> None:
    data = {
        "type": "OBB",
        "center":  obb.center.tolist(),
        "extent":  obb.extent.tolist(),          # edge lengths (Lx, Ly, Lz)
        "rotation": obb.R.tolist(),              # 3x3 rotation matrix
    }
    json.dump(data, open(path, "w"))

def load_aabb_json(path:str) -> o3d.geometry.AxisAlignedBoundingBox:
    data = json.load(open(path))
    assert data["type"] == "AABB"
    return o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array(data["min_bound"]),
        max_bound=np.array(data["max_bound"])
    )

def load_obb_json(path:str) -> o3d.geometry.OrientedBoundingBox:
    data = json.load(open(path))
    assert data["type"] == "OBB"
    return o3d.geometry.OrientedBoundingBox(
        center=np.array(data["center"]),
        R=np.array(data["rotation"]),
        extent=np.array(data["extent"])
    )
