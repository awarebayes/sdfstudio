import open3d as o3d
import json
import numpy as np
import cv2
import pycolmap

metadata_path = '/home/mscherbina/Documents/sfm_deploynmnets/office_vhq/static/storage/reconstructions/global/sdfstudio-data/meta_data.json'
colmap_path = '/home/mscherbina/Documents/sfm_deploynmnets/office_vhq/static/storage/reconstructions/global/outputs/ref/'
metadata = json.load(open(metadata_path))

def filter_points(rec: pycolmap.Reconstruction):
    bbs = rec.compute_bounding_box(0.01, 0.99)
    max_reproj_error = 6
    min_track_length = 2

    # Filter points, use original reproj error here
    points = [
        (p3D.xyz, p3D.color)
        for _, p3D in rec.points3D.items()
        if (
            (p3D.xyz >= bbs[0]).all()
            and (p3D.xyz <= bbs[1]).all()
            and p3D.error <= max_reproj_error
            and p3D.track.length() >= min_track_length
        )
    ]
    xyzs = np.array([i[0] for i in points])
    colors = np.array([i[1] for i in points]) / 255.0
    return xyzs, colors


refined = pycolmap.Reconstruction(colmap_path)
points_xyz, colors = filter_points(refined)
points_xzy = points_xyz[:, [0, 1, 2]]
swapped_point_cloud = o3d.geometry.PointCloud()
swapped_point_cloud.points = o3d.utility.Vector3dVector(points_xzy)

model = o3d.io.read_triangle_mesh('meshes/main_v2.ply')
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
model.compute_vertex_normals()

# points_xzy = points_xyz[:, [1, 2, 0]]
# points_xzy = points_xyz[:, [1, 2, 0]]
# points_xzy = points_xyz[:, [1, 0, 2]]
# points_xzy[:, 0] = -points_xzy[:, 0]
# points_xzy[:, 1] = -points_xzy[:, 1]
# points_xzy[:, 2] = -points_xzy[:, 2]

# Convert the swapped NumPy array back to an Open3D point cloud

worldtogt = np.array(metadata['worldtogt'])

transform = np.array([
    [-0.,  1.,  0.,  0.],
    [ 1.,  0., -0., -0.],
    [-0., -0., -1.,  0.],
    [ 0.,  0.,  0.,  1.]]
)

camera_poses = []
xyz = []
#for frame in metadata['frames']:
#    pose = np.array(frame['camtoworld'])
#    t = pose[:3, 3]

model.transform(transform)
model.transform(worldtogt)

o3d.visualization.draw_geometries([model, swapped_point_cloud, coordinate_frame, *camera_poses])
#o3d.visualization.draw_geometries([model, *camera_poses])
