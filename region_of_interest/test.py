import viser
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/workspace/Desktop/RoICalculation/region_of_interest/region_of_interest/points.ply")


server = viser.ViserServer()

server.scene.add_point_cloud(
    name = "/point_cloud",
    points = np.asarray(pcd.points),
    point_size=0.0001,
    colors=[0, 0, 0],
)

while True:
    continue