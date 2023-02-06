import numpy as np
from pathlib import Path
from pol.utils.validation.scene_saver import load_scenes, count_h5_keys
import fiftyone as fo
from tqdm import tqdm
import argparse
from pol.utils.path import PathHelper
from pol.utils.cluster_mso import cluster_solutions
import json
import re
from scipy.optimize import linear_sum_assignment
import torch

g_default_label = 'thing'
g_max_cluster_count = 20

def compute_box_area(boxes):
    # boxes: Kx4
    # (x,y,w,h), (x,y) is center
    return 4 * boxes[:, 2] * boxes[:, 3]

def convert_to_aabb(boxes):
    return torch.stack([boxes[:, 0] - boxes[:, 2],
                        boxes[:, 1] - boxes[:, 3],
                        boxes[:, 0] + boxes[:, 2],
                        boxes[:, 1] + boxes[:, 3]], -1)

def compute_iou(boxes0, boxes1):
    # boxes0: Kx4, boxes1: Nx4
    if isinstance(boxes0, np.ndarray):
        boxes0 = torch.from_numpy(boxes0)
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1)
    K = boxes0.shape[0]
    N = boxes1.shape[0]

    areas0 = compute_box_area(boxes0) # K
    areas1 = compute_box_area(boxes1) # N

    aabb0 = convert_to_aabb(boxes0) # Kx4
    aabb1 = convert_to_aabb(boxes1) # Nx4

    high = torch.minimum(aabb0[:, 2:].unsqueeze(-2).expand(-1, N, -1),
                         aabb1[:, 2:].unsqueeze(-3).expand(K, -1, -1)) # KxNx2
    low = torch.maximum(aabb0[:, :2].unsqueeze(-2).expand(-1, N, -1),
                        aabb1[:, :2].unsqueeze(-3).expand(K, -1, -1)) # KxNx2
    intersection_areas = ((high[:, :, 0] - low[:, :, 0]).relu() *
                          (high[:, :, 1] - low[:, :, 1]).relu()) # KxN
    total_areas = areas0.unsqueeze(-1) + areas1.unsqueeze(0) # KxN
    if (not (total_areas + 1e-4 >= intersection_areas).all().item()):
        print(f'boxes0: {boxes0}\nboxes1: {boxes1}')
        print(f'high: {high}\nlow: {low}')
        print(f'total: {total_areas}\nintersection: {intersection_areas}')
        assert(False)
    return (intersection_areas / (total_areas - intersection_areas)).numpy()

def eval_scenes(scene_path, eval_json_file,
                overwrite=False,
                scene_range=None,
                eval_itr=-1,
                iou_threshold=0.5,
                cluster_bandwidth=0.02):
    if eval_json_file.exists() and not overwrite:
        return

    scenes = load_scenes(scene_path, scene_ids=scene_range, unroll=False)
    precision_thresholds = torch.linspace(0.001, 0.5, 50)
    witness_list = []
    for i in range(10):
        witness = torch.rand([1024, 4])
        witness_list.append(witness)

    samples = []
    num_tp = 0
    num_total_pred = 0
    num_total_gt = 0
    pbar = tqdm(enumerate(scenes))
    # pbar = tqdm(enumerate(scenes[:4]))

    from pol.utils.validation.mso_eval import MSOEvaluation, MSOSolution
    witness_results = []
    for i, scene in pbar:
        gt_boxes = torch.from_numpy(scene['gt']['boxes'][:])
        gt_box_masks = torch.from_numpy(scene['gt']['box_masks'][:])

        gt_boxes = gt_boxes[gt_box_masks, :] # Kx4
        num_total_gt += gt_boxes.shape[0]

        num_itr = count_h5_keys(scene, 'itr')
        if eval_itr < 0:
            eval_itr += num_itr
        pred_boxes = scene[f'itr_{eval_itr}']['boxes'][:]
        pred_boxes = cluster_solutions(pred_boxes,
                                       bandwidth=cluster_bandwidth, include_freq=False) # Nx4
        pred_boxes = torch.from_numpy(pred_boxes)
        if pred_boxes.shape[0] > g_max_cluster_count:
            pred_boxes = pred_boxes[:0]
        num_total_pred += pred_boxes.shape[0]

        if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
            continue

        iou = compute_iou(gt_boxes, pred_boxes) # KxN
        cost = (iou > iou_threshold).astype(int) # KxN
        row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
        num_tp += cost[row_ind, col_ind].sum()

        # Next compute witnessed metrics.
        gt_F = torch.zeros([1, gt_boxes.shape[0]], dtype=torch.float32)
        gt_S = torch.ones_like(gt_F, dtype=torch.bool)

        gt_sol = MSOSolution(gt_boxes.unsqueeze(0), gt_F, gt_S)

        pred_F = torch.zeros([1, pred_boxes.shape[0]], dtype=torch.float32)
        pred_S = torch.ones_like(pred_F, dtype=torch.bool)
        pred_sol = MSOSolution(pred_boxes.unsqueeze(0), pred_F, pred_S)

        cur_results = []
        for j in range(len(witness_list)):
            witness = witness_list[j]
            # Don't use any threshold here.
            mso_eval = MSOEvaluation(witness=witness.unsqueeze(0),
                                     inf_dist=10.0, obj_threshold=1.0,
                                     precision_thresholds=precision_thresholds,
                                     gt_solution=gt_sol, candidate_solution=pred_sol,
                                     cuda=True,
                                     quiet=True)
            mso_eval.eval()
            cur_results.append(mso_eval.avg_results)
        witness_results.append(cur_results)

        pbar.set_description("{}/{}, precision: {:.3f}, recall: {:.3f}".format(
            i, len(scenes), num_tp / num_total_pred, num_tp / num_total_gt
        ))


    # Process witness_results, which is 2D. First average over all images.
    witness_keys = witness_results[0][0].keys()
    avg_results = [{} for _ in range(len(witness_list))]
    for j in range(len(witness_list)):
        avg_result = avg_results[j]
        for key in witness_keys:
            if key == 'precision':
                avg_result[key] = torch.stack([r[j][key] for r in witness_results],
                                              -1).mean(-1)
            else:
                avg_result[key] = torch.tensor([r[j][key] for r in witness_results],
                                               dtype=torch.float32).mean()
    result_dict = {}
    # Now average over witnesses and compute mean & std.
    for key in witness_keys:
        stacked = torch.stack([r[key] for r in avg_results], -1)
        mean_result = stacked.mean(-1)
        std_result = stacked.std(-1)

        result_dict[key] = {}
        if mean_result.ndim == 0:
            result_dict[key]['mean'] = mean_result.item()
            result_dict[key]['std'] = std_result.item()
        else:
            result_dict[key]['mean'] = mean_result.tolist()
            result_dict[key]['std'] = std_result.tolist()
    result_dict['precision_thresholds'] = precision_thresholds.tolist()

    result_dict.update({
        'num_tp': int(num_tp),
        'num_total_pred': num_total_pred,
        'num_total_gt': num_total_gt,
        'match_precision': num_tp / num_total_pred,
        'match_recall': num_tp / num_total_gt,
    })

    import json
    with open(eval_json_file, 'w') as f:
        f.write(json.dumps(result_dict, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--scene_range', type=int, nargs='+', default=None)
    parser.add_argument('--cluster_bandwidth', type=float, default=0.01)
    parser.add_argument('--eval_itr', type=int, default=-1)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    args = parser.parse_args()

    problem_name = args.problem
    method_name = args.method
    path_helper = PathHelper('.')
    scene_path = path_helper.locate_scene_h5(problem_name,
                                             method_name)
    exp_name = path_helper.format_exp_name(problem_name, method_name)
    eval_dir = path_helper.app_dir / 'eval' / exp_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_json_file = eval_dir / 'eval.json'

    eval_scenes(scene_path, eval_json_file,
                overwrite=args.overwrite,
                scene_range=args.scene_range,
                eval_itr=args.eval_itr,
                cluster_bandwidth=args.cluster_bandwidth,
                iou_threshold=args.iou_threshold)

