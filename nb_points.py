import os
import trimesh
import sys
from tqdm import tqdm
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from cadrille.rl_finetune.utils import transform_real_mesh,  compute_cd, compute_iou, transform_gt_mesh
from utils import compute_normals_metrics
import pandas as pd


deepcad_gen_path = "/Users/marina/projects/AIRI/data/gen_deepcad"
deepcad_gt_path = "/Users/marina/projects/AIRI/data/gt_deepcad"
gt_files = [f for f in os.listdir(deepcad_gt_path) if f.endswith('.stl')]
s = 100

gt_files = random.sample(gt_files, s)

nb_points = [2**i for i in range(10,16)]
tols = [2, 5, 10]  # in percent
results = []

for file in tqdm(gt_files):
    file_gen = file.replace('.stl', '+0.stl')
    gen_mesh = trimesh.load(os.path.join(deepcad_gen_path, file_gen))
    gt_mesh = trimesh.load(os.path.join(deepcad_gt_path, file))
    gen_mesh = transform_gt_mesh(gen_mesh)
    gt_mesh = transform_gt_mesh(gt_mesh)
    
    iou = compute_iou(gen_mesh, gt_mesh)
    for point in nb_points:
        cd = compute_cd(gen_mesh, gt_mesh, point) * 1000
        for tol in tols:
            auc_gen, mean_cos_sim, _ = compute_normals_metrics(gen_mesh, gt_mesh, tol=tol, n_points=point)

            results.append({
                'filename': file,
                'nb_points': point,
                'AOC': auc_gen,
                'Mean Cosine Similarity': mean_cos_sim,
                'CD (x1000)': cd,
                'IoU': iou,
                'tolerance': tol
            })

results_df = pd.DataFrame(results)
results_df.to_csv(f'metrics_pc.csv', index=False)