import trimesh
import numpy as np
import os

import random
from scipy.spatial import cKDTree
import warnings
import cadquery as cq
import open3d
import skimage.transform
from PIL import Image
import torch
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points

os.environ["PYGLET_HEADLESS"] = "True"
nb_points = 8192

mesh_path = "/Users/marina/.cache/huggingface/hub/datasets--maksimko123--fusion360_test_mesh/snapshots/af9643d11bdae5512020bfba024cb4d609b893e1"
mesh_files = [f for f in os.listdir(mesh_path) if f.endswith('.stl')][:100]

pc_path = "./point_clouds"

def mesh_to_point_cloud(mesh, n_points, n_pre_points=8192):
    vertices, faces = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    vertices = vertices[ids]
    return np.asarray(vertices)


def mesh_to_image(mesh, camera_distance=-1.8, front=[1, 1, 1], width=500, height=500, img_size=256):
    
    o3d_mesh = open3d.geometry.TriangleMesh()
    o3d_mesh.vertices = open3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = open3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals() 
    
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(o3d_mesh)

    lookat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    front_array = np.array(front, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    
    eye = lookat + front_array * camera_distance
    right = np.cross(up, front_array)
    right /= np.linalg.norm(right)
    true_up = np.cross(front_array, right)
    rotation_matrix = np.column_stack((right, true_up, front_array)).T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = -rotation_matrix @ eye

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = skimage.transform.resize(
        image,
        output_shape=(img_size, img_size),
        order=2,
        anti_aliasing=True,
        preserve_range=True).astype(np.uint8)

    return Image.fromarray(image)

def save_ply(pc, pc_file):
    o3d_pc = open3d.geometry.PointCloud()
    o3d_pc.points = open3d.utility.Vector3dVector(pc)
    open3d.io.write_point_cloud(pc_file, o3d_pc)

def convert_mesh(mesh_file):
    """
    Convert a mesh file to a point cloud and an image and save them.
    """
    mesh = trimesh.load(mesh_file, force='mesh')

    if not mesh.is_watertight:
        warnings.warn(f"Mesh {mesh_file} is not watertight.")
    
    # save the point cloud to a file
    pc = mesh_to_point_cloud(mesh, nb_points)
    pc_file = os.path.join(pc_path, os.path.basename(mesh_file).replace('.stl', '.ply'))
    save_ply(pc, pc_file)

    # save the image to a file
    image = mesh_to_image(mesh)
    image_file = os.path.join(pc_path, os.path.basename(mesh_file).replace('.stl', '.png'))
    image.save(image_file)
    print(f"Processed {mesh_file}: Point cloud saved to {pc_file}, image saved to {image_file}")


if __name__ == "__main__":
    convert_mesh(os.path.join(mesh_path, mesh_files[10]))