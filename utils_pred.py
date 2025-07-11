import cadquery as cq
from geomloss import SamplesLoss
from scipy.spatial import cKDTree
import trimesh
import numpy as np
import torch

import matplotlib.pyplot as plt


def compute_normals_metrics(pred_mesh, gt_mesh, tol=1, visualize=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area under the curve (AUC) of the angle distribution between the normals.
    Returns the AUC and median cosine similarity.
    """
    n_points = 8192
    tol = 2 * tol / 100  # 2 is the extent of the mesh
    print(f"tolerence: {tol:.4f}")

    gt_points, gt_face_indexes = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, pred_face_indexes = trimesh.sample.sample_surface(pred_mesh, n_points)


    distances, indices = cKDTree(gt_points).query(pred_points, k=1, distance_upper_bound=tol)
    valid_indices = indices[distances < tol]
    
    nb_invalid = n_points - len(valid_indices)
    per_invalid = nb_invalid / n_points * 100
    print(f"Number of invalid neighbors: {nb_invalid} out of {n_points} ({per_invalid:.2f}%)")
    if nb_invalid == n_points:
        return 1.0, 0.0, 100
    
    # normals of sampled points
    gt_normals = gt_mesh.face_normals[gt_face_indexes]
    pred_normals = pred_mesh.face_normals[pred_face_indexes]
    
    valid_pred_normals = pred_normals[distances < tol]
    valid_gt_normals = gt_normals[valid_indices]
    
    
    # compute cosine similarity
    cos_sim = (valid_pred_normals * valid_gt_normals).sum(axis=1)

    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    median_cos_sim = np.median(cos_sim)
    
    # distribution of angles between normals
    angles = np.arccos(cos_sim)
    angles = np.sort(angles)
    # add invalid neighbors to the end of the array with max angle (pi)
    angles = np.concatenate((angles, np.full(nb_invalid, np.pi)))

    N = len(angles)
    cdf = np.arange(1, N+1) / N

    from numpy import trapz
    x = np.concatenate(([0.0], angles, [np.pi]))
    y = np.concatenate(([0.0],   cdf,   [1.0]))
    auc_normalized = trapz(y, x) / np.pi  # Normalize by the maximum possible AUC (which is pi)

    auc_normalized = 1 - auc_normalized
    # plot the AUC
    #if auc_normalized > 0.3:
        #print(f"HIGH AUC: {auc_normalized:.2f}")
        #plot_auc(angles, cdf, title='AUC of Normal Angles', auc_value=auc_normalized)
    if visualize:
        visualize_normals(valid_pred_normals, valid_gt_normals, pred_points[distances < tol], gt_points[valid_indices])


    return auc_normalized, median_cos_sim, per_invalid

def visualize_normals(pred_normals, gt_normals, pred_points, gt_points):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect((1,1,1))

    ax.quiver(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
              pred_normals[:, 0], pred_normals[:, 1], pred_normals[:, 2],
              length=0.1, color='r', label='Predicted Normals', alpha=0.5, normalize=True)

    ax.quiver(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
              gt_normals[:, 0], gt_normals[:, 1], gt_normals[:, 2],
              length=0.1, color='b', label='Ground Truth Normals', alpha=0.5, normalize=True)

    ax.set_title('Normals Visualization')
    ax.legend()
    plt.show()