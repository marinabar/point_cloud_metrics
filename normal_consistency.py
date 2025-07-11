import os
import numpy as np
import trimesh
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from cadrille.rl_finetune.utils import transform_real_mesh,  compute_cd, compute_iou, transform_gt_mesh
from utils import compute_normals_metrics
from process_stl import mesh_to_point_cloud, save_ply
from tweaks import add_jitter, mirror_mesh_x, rotate_mesh
import pandas as pd
from tqdm import tqdm


mesh_path = "/Users/marina/.cache/huggingface/hub/datasets--maksimko123--fusion360_test_mesh/snapshots/af9643d11bdae5512020bfba024cb4d609b893e1"
deepcad_gen_path = "/Users/marina/projects/AIRI/data/gen_deepcad"
deepcad_gt_path = "/Users/marina/projects/AIRI/data/gt_deepcad"
gt_files = [f for f in os.listdir(deepcad_gt_path) if f.endswith('.stl')]

results = []
tol = 10 # 2 %
for f in tqdm(gt_files):
    # compute metrics predicted vs ground truth, and compare with noisy vs ground truth
    f_gen = f.replace('.stl', '+0.stl')
    gen_mesh = trimesh.load(os.path.join(deepcad_gen_path, f_gen))
    gt_mesh = trimesh.load(os.path.join(deepcad_gt_path, f))
    gen_mesh = transform_gt_mesh(gen_mesh)
    gt_mesh = transform_gt_mesh(gt_mesh)

    """
    modified_mesh = add_jitter(gen_mesh, jitter=0.005)
    print("computing normals metrics for noisy mesh")
    auc_noisy, median_cos_sim_noisy = compute_normals_metrics(modified_mesh, gt_mesh)
    print("computing normals metrics for generated mesh")"""
    auc_gen, median_cos_sim_gen, per_inval = compute_normals_metrics(gen_mesh, gt_mesh, tol = tol)

    results.append({
        'filename': f,
        'aoc_generated': auc_gen,
        'median_cos_sim_generated': median_cos_sim_gen,
        'percentage_samples_with_no_neigh': per_inval
    })
    #print(f"Processed: {f}")

mean_aoc = np.mean([res['aoc_generated'] for res in results])
mean_cos_sim = np.mean([res['median_cos_sim_generated'] for res in results])
mean_per_inval = np.mean([res['percentage_samples_with_no_neigh'] for res in results])

print(f"Mean AOC: {mean_aoc:.4f}, Mean Median Cosine Similarity: {mean_cos_sim:.4f}, Mean Percentage Invalid Neighbors: {mean_per_inval:.2f}%")

results_df = pd.DataFrame(results)

mean_row = pd.DataFrame([{
    'filename': 'mean',
    'aoc_generated': mean_aoc,
    'median_cos_sim_generated': mean_cos_sim,
    'percentage_samples_with_no_neigh': mean_per_inval
}])
results_df = pd.concat([results_df, mean_row], ignore_index=True)
results_df.to_csv(f'nc_results_tol_{tol}_percent.csv', index=False)
print("Results saved to .csv")
