
import trimesh
import numpy as np
import os
import random
import torch
import sys

from utils import compute_emd
nb_points = 8192

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from cadrille.rl_finetune.utils import transform_real_mesh,  compute_cd, compute_iou


def compute_metrics(pred_mesh, gt_mesh):
    return compute_cd(pred_mesh, gt_mesh), compute_iou(pred_mesh, gt_mesh), compute_emd(pred_mesh, gt_mesh, device='mps')


def add_jitter(mesh, jitter=0.01):
    noisy_mesh = trimesh.permutate.noise(mesh, magnitude=jitter)
    return noisy_mesh

def mirror_mesh_x(mesh):
    if mesh is None:
        return None
    
    point = [0, 0, 0]
    normal = [1, 0, 0]
    
    reflection_matrix = trimesh.transformations.reflection_matrix(point, normal)
    
    return mesh.apply_transform(reflection_matrix)

def rotate_mesh(mesh, angle, axis=[1, 0, 0], point=[0, 0, 0]):
    # angle in radians
    if mesh is None:
        return None
    
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis, point)
    transformed_mesh = mesh.apply_transform(rotation_matrix)

    return transformed_mesh



def run_batch_comparison(mesh_files, mesh_path, device):
    """Loads meshes, applies modifications, and computes mean metrics for the batch."""
    
    all_cd_scores = []
    all_iou_scores = []
    all_emd_scores = []

    print(f"\nStarting batch processing for {len(mesh_files)} files...")
    
    for i, filename in enumerate(mesh_files):
        full_path = os.path.join(mesh_path, filename)
        
        try:
            # --- Load and Prepare GT Mesh ---
            gt_mesh = trimesh.load(full_path, force='mesh')
            if not isinstance(gt_mesh, trimesh.Trimesh): continue
            
            gt_mesh = transform_real_mesh(gt_mesh)
            
            # --- Create a Modified Mesh ---
            modified_mesh = add_jitter(gt_mesh, jitter=0.02)
            # modified_mesh = mirror_mesh_x(gt_mesh)
            # modified_mesh = rotate_mesh(gt_mesh, angle=np.pi / 4) 
            
            if gt_mesh is None or modified_mesh is None:
                print(f"Skipping {filename} due to loading or processing error.")
                continue

            # --- Compute Metrics ---
            cd, iou, emd = compute_metrics(modified_mesh, gt_mesh)
            
            all_cd_scores.append(cd)
            all_iou_scores.append(iou)
            all_emd_scores.append(emd)
            
            print(f"  ({i+1}/{len(mesh_files)}) Processed {filename}: CD={cd:.6f}, IoU={iou:.4f}, EMD={emd:.6f}")

        except Exception as e:
            print(f"Failed to process {filename}. Error: {e}")

    # --- Calculate and Display Mean Metrics for the Batch ---
    if all_cd_scores:
        mean_cd = np.mean(all_cd_scores)
        mean_iou = np.mean(all_iou_scores)
        mean_emd = np.mean(all_emd_scores)

        print("\n--- Batch Comparison Results ---")
        print(f"Modification Type: Jitter (strength=0.02)")
        print(f"Mean Chamfer Distance (CD): {mean_cd:.6f}")
        print(f"Mean Intersection over Union (IoU): {mean_iou:.4f}")
        print(f"Mean Earth Mover's Distance (EMD): {mean_emd:.6f}")
        print("--------------------------------")
    else:
        print("\nNo meshes were successfully processed.")

device = torch.device("mps")

if __name__ == '__main__':
    mesh_path = "/Users/marina/.cache/huggingface/hub/datasets--maksimko123--fusion360_test_mesh/snapshots/af9643d11bdae5512020bfba024cb4d609b893e1"

    
    if not os.path.isdir(mesh_path):
        print(f"Error: Directory not found at {mesh_path}")
    else:
        mesh_files = [f for f in os.listdir(mesh_path) if f.lower().endswith('.stl')][:200]
        random.shuffle(mesh_files)
        
        if not mesh_files:
            print("No STL files found in the specified directory.")
        else:
            run_batch_comparison(mesh_files, mesh_path, device)