import os
import random
import pandas as pd
import trimesh
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from cadrille.rl_finetune.utils import transform_gt_mesh, compute_cd, compute_iou
from utils import compute_normals_metrics


GT_PATH     = "/Users/marina/projects/AIRI/data/gt_deepcad"
GEN_PATH    = "/Users/marina/projects/AIRI/data/gen_deepcad"
OUTPUT_CSV  = "metrics_pc_all.csv"
SAMPLE_SIZE = 100
NB_POINTS   = [2**i for i in range(10, 16)]
TOLS        = [2, 5, 10]
WORKERS     = cpu_count()

def process_file(file_name):
    rows     = []
    pred_name = file_name.replace('.stl', '+0.stl')
    pred_path = os.path.join(GEN_PATH, pred_name)
    gt_path   = os.path.join(GT_PATH, file_name)

    try:
        pred_mesh = trimesh.load(pred_path)
        gt_mesh   = trimesh.load(gt_path)
    except Exception:
        return rows

    pred_mesh = transform_gt_mesh(pred_mesh)
    gt_mesh   = transform_gt_mesh(gt_mesh)
    iou_value = compute_iou(pred_mesh, gt_mesh)

    for pts in NB_POINTS:
        cd_value = compute_cd(pred_mesh, gt_mesh, pts) * 1000
        for tol in TOLS:
            aoc, mean_cos, _ = compute_normals_metrics(
                pred_mesh, gt_mesh, tol=tol, n_points=pts
            )
            rows.append({
                'filename':               file_name,
                'nb_points':              pts,
                'tolerance':              tol,
                'AOC':                    aoc,
                'Mean Cosine Similarity': mean_cos,
                'CD (x1000)':             cd_value,
                'IoU':                    iou_value
            })
    return rows

if __name__ == '__main__':
    gt_files = [f for f in os.listdir(GT_PATH) if f.endswith('.stl')]
    if SAMPLE_SIZE:
        gt_files = random.sample(gt_files, SAMPLE_SIZE)

    all_rows = []
    with Pool(WORKERS) as pool:
        for rows in tqdm(pool.imap(process_file, gt_files), total=len(gt_files)):
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results ({len(df)} rows) to {OUTPUT_CSV}")
