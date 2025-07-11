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

def plot_aoc(angles, cdf, title='aoc Plot', aoc_value=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(angles, cdf, marker='o', linestyle='-', color='b')
    plt.title(title + (f' - aoc: {aoc_value:.2f}' if aoc_value is not None else ''))
    plt.xlabel('Angle (radians)')
    plt.ylabel('Cumulative Distribution Function (CDF)')
    plt.grid()
    plt.xlim(0, np.pi)
    plt.ylim(0, 1)
    plt.show()

def compute_normals_metrics(pred_mesh, gt_mesh, tol=1, n_points=8192, visualize=False, visualize_aoc=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area over the curve (AOC) of the angle distribution between the normals.
    Returns the aoc and mean_cos_sim
    """
    #tol = 0.01 * max(gt_mesh.extents.max(), pred_mesh.extents.max())  # 1% of the mesh extent
    tol = pred_mesh.extents.max() * tol  / 100
    print(f"tolerence: {tol:.4f}")

    gt_points, gt_face_indexes = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, pred_face_indexes = trimesh.sample.sample_surface(pred_mesh, n_points)

    # normals of sampled points
    gt_normals = gt_mesh.face_normals[gt_face_indexes]
    pred_normals = pred_mesh.face_normals[pred_face_indexes]

    tree = cKDTree(pred_points)
    neighbors = tree.query_ball_point(gt_points, r=tol)
    # get the indices of the neighbors for each ground-truth point

    valid_pred_normals = []
    valid_gt_normals = []
    valid_gt_points = []
    valid_pred_points = []

    for i, idxs in enumerate(neighbors):
        if len(idxs) == 0:
            continue
        gn = gt_normals[i]
        pn_neighbors = pred_normals[idxs] # candidates

        valid_gt_normals.append(gn)
        dots = (pn_neighbors * gn).sum(axis=1)  # (k,)
        best_idx = np.argmax(dots)  # index of the best aligned normal

        valid_pred_normals.append(pn_neighbors[best_idx])  # (3,)

        valid_gt_points.append(gt_points[i])  # (3,)
        valid_pred_points.append(pred_points[idxs[best_idx]])  # (3,)

    valid_gt_normals = np.vstack(valid_gt_normals)
    valid_pred_normals = np.vstack(valid_pred_normals)
    valid_gt_points = np.vstack(valid_gt_points)
    valid_pred_points = np.vstack(valid_pred_points)

    nb_invalid = n_points - len(valid_pred_normals)
    per_invalid = nb_invalid / n_points * 100
    print(f"Number of points with no neighbors within tol: {nb_invalid} out of {n_points} ({per_invalid:.2f}%)")

    if nb_invalid == n_points:
        return 1.0, 0.0, 100
    
    
    # compute cosine similarity
    cos_sim = (valid_pred_normals * valid_gt_normals).sum(axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    mean_cos_sim = np.mean(cos_sim)
    
    # distribution of angles between normals
    angles = np.arccos(cos_sim)
    angles = np.sort(angles)

    # add invalid points to the end of the array with max angle (pi)
    angles = np.concatenate((angles, np.full(nb_invalid, np.pi)))

    N = len(angles)
    cdf = np.arange(1, N+1) / N

    from numpy import trapz
    x = np.concatenate(([0.0], angles, [np.pi]))
    y = np.concatenate(([0.0],   cdf,   [1.0]))
    aoc_normalized = trapz(y, x) / np.pi  # Normalize by the maximum possible aoc (which is pi)

    aoc_normalized = 1 - aoc_normalized
    # plot the aoc
    if aoc_normalized > 0.3:
        print(f"HIGH aoc: {aoc_normalized:.2f}")
        #plot_aoc(angles, cdf, title='aoc of Normal Angles', aoc_value=aoc_normalized)

    if visualize_aoc:
        plot_aoc(angles, cdf, title='aoc of Normal Angles', aoc_value=aoc_normalized)

    if visualize:
        visualize_normals(valid_pred_normals, valid_gt_normals, valid_pred_points, valid_gt_points)


    return aoc_normalized, mean_cos_sim, per_invalid

def visualize_normals(pred_normals, gt_normals, pred_points, gt_points):

    # get points only with highest value along z axis
    pred_normals = pred_normals[pred_points[:, 2].argsort()[-50:]]
    gt_normals = gt_normals[gt_points[:, 2].argsort()[-50:]]
    pred_points = pred_points[pred_points[:, 2].argsort()[-50:]]
    gt_points = gt_points[gt_points[:, 2].argsort()[-50:]]


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    sample_size = 50
    # get 50 points with lowest cosine similarity
    cos_sim = (pred_normals * gt_normals).sum(axis=1)
    indices = np.argsort(cos_sim)[:sample_size]
    pred_normals = pred_normals[indices]
    gt_normals = gt_normals[indices]
    pred_points = pred_points[indices]
    gt_points = gt_points[indices]

    ax.set_box_aspect((1,1,1))

    ax.quiver(pred_points[:sample_size, 0], pred_points[:sample_size, 1], pred_points[:sample_size, 2],
              pred_normals[:sample_size, 0], pred_normals[:sample_size, 1], pred_normals[:sample_size, 2],
              length=0.1, color='r', label='Predicted Normals', alpha=0.5, normalize=True)

    ax.quiver(gt_points[:sample_size, 0], gt_points[:sample_size, 1], gt_points[:sample_size, 2],
              gt_normals[:sample_size, 0], gt_normals[:sample_size, 1], gt_normals[:sample_size, 2],
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

def compute_iou(pred_mesh, gt_mesh):
    intersection_volume = 0
    for gt_mesh_i in gt_mesh.split():
        for pred_mesh_i in pred_mesh.split():
            intersection = gt_mesh_i.intersection(pred_mesh_i)
            volume = intersection.volume if intersection is not None else 0
            intersection_volume += volume

    gt_volume = sum(m.volume for m in gt_mesh.split())
    pred_volume = sum(m.volume for m in pred_mesh.split())
    union_volume = gt_volume + pred_volume - intersection_volume
    iou = intersection_volume / (union_volume + 1e-6)
    return iou


def compute_cd(pred_mesh, gt_mesh):
    n_points = 8192
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd