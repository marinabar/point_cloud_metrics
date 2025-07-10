import cadquery as cq
from geomloss import SamplesLoss
from scipy.spatial import cKDTree
import trimesh
import numpy as np
import torch

import matplotlib.pyplot as plt


def compute_emd(pred_mesh, gt_mesh, n_points=8192, device='cpu'):

    gt_points_np, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points_np, _ = trimesh.sample.sample_surface(pred_mesh, n_points)

    t_gt_points = torch.tensor(gt_points_np, dtype=torch.float32, device=device)
    t_pred_points = torch.tensor(pred_points_np, dtype=torch.float32, device=device)

    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    emd = loss(t_pred_points, t_gt_points).item()

    return np.sqrt(emd)

def plot_auc(angles, cdf, title='AUC Plot', auc_value=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(angles, cdf, marker='o', linestyle='-', color='b')
    plt.title(title + (f' - AUC: {auc_value:.2f}' if auc_value is not None else ''))
    plt.xlabel('Angle (radians)')
    plt.ylabel('Cumulative Distribution Function (CDF)')
    plt.grid()
    plt.xlim(0, np.pi)
    plt.ylim(0, 1)
    plt.show()

def compute_normals_metrics(pred_mesh, gt_mesh, visualize=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area under the curve (AUC) of the angle distribution between the normals.
    Returns the AUC and median cosine similarity.
    """
    n_points = 8192
    tol = 0.02 # 1% of 2, the extent of the normalized mesh
    tol = 0.01 * min(gt_mesh.extents.min(), pred_mesh.extents.min())  # 1% of the mesh extent
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


from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mesh(mesh, elev=30, azim=45):
    import plotly.graph_objects as go
    v = mesh.vertices
    f = mesh.faces

    # all unique edges
    from trimesh.geometry import faces_to_edges
    raw = faces_to_edges(f)
    e   = np.sort(raw, axis=1)
    e   = np.unique(e,   axis=0)

    fig = go.Figure([
        # filled faces
        go.Mesh3d(
            x=v[:,0], y=v[:,1], z=v[:,2],
            i=f[:,0], j=f[:,1], k=f[:,2],
            opacity=0.3,
            flatshading=True
        ),
        # wireframe edges
        go.Scatter3d(
            x=v[e].reshape(-1,3)[:,0],
            y=v[e].reshape(-1,3)[:,1],
            z=v[e].reshape(-1,3)[:,2],
            mode='lines',
            line=dict(width=1)
        ),
        # vertices
        go.Scatter3d(
            x=v[:,0], y=v[:,1], z=v[:,2],
            mode='markers',
            marker=dict(size=2, color='red')
        )
    ])
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()
