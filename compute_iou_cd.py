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


mesh_path = "/Users/marina/.cache/huggingface/hub/datasets--maksimko123--fusion360_test_mesh/snapshots/af9643d11bdae5512020bfba024cb4d609b893e1"
deepcad_gen_path = "/Users/marina/projects/AIRI/data/gen_deepcad"
deepcad_gt_path = "/Users/marina/projects/AIRI/data/gt_deepcad"
gt_files = [f for f in os.listdir(deepcad_gt_path) if f.endswith('.stl')]

results = []
for f in gt_files:
    # compute metrics predicted vs ground truth
    f_gen = f.replace('.stl', '+0.stl')
    gen_mesh = trimesh.load(os.path.join(deepcad_gen_path, f_gen))
    gt_mesh = trimesh.load(os.path.join(deepcad_gt_path, f))
    gen_mesh = transform_gt_mesh(gen_mesh)
    gt_mesh = transform_gt_mesh(gt_mesh)
    try:
        iou = compute_iou(gen_mesh, gt_mesh)
    except Exception as e:
        print(f"Error computing IoU for {f}: {e}")
        iou = np.nan
    cd = compute_cd(gen_mesh, gt_mesh)

    results.append({
        'filename': f,
        'iou': iou,
        'cd': cd
    })
    print(f"Processed: {f}")

mean_iou = np.nanmean([res['iou'] for res in results])
mean_cd = np.mean([res['cd'] for res in results])
nb_nan_iou = np.sum(np.isnan([res['iou'] for res in results]))

print(f"Mean IoU: {mean_iou:.4f}, Mean CD: {mean_cd:.4f}, nan IoU count: {nb_nan_iou}")
print()

results_df = pd.DataFrame(results)

mean_row = pd.DataFrame([{
    'filename': 'mean',
    'iou': mean_iou,
    'cd': mean_cd
    #'emd': np.mean([res['emd'] for res in results])
}])
results_df = pd.concat([results_df, mean_row], ignore_index=True)
results_df.to_csv('emd_results.csv', index=False)
print("Results saved to emd_results.csv")