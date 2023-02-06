import numpy as np
import torch
from pathlib import Path
from pol.utils.validation.scene_saver import load_scenes, count_h5_keys, find_max_h5_key
from pol.utils.validation.mso_eval import MSOSolution, MSOEvaluation, distance_sqr_matrix
from pol.utils.path import PathHelper
import tqdm

class Evaluation:
    def __init__(self, *,
                 root_dir='.',
                 problem,
                 method_itr):
        self.path_helper = PathHelper(root_dir)
        self.problem = problem

    def load_solution(self, method, is_gt, fast=False):
        scene_path = self.path_helper.locate_scene_h5(self.problem, method)
        scenes = load_scenes(scene_path, unroll=False)

        if method in method_itr:
            k = method_itr[method]
        else:
            k = find_max_h5_key(scenes[0], 'itr', return_itr=True)
        print('Loading result of {} at iteration {}.'.format(method, k))


        X_list = []
        loss_list = []
        S_list = []
        for i, scene in enumerate(scenes):
            tmp = scene['itr_{}'.format(k)]
            X = torch.from_numpy(tmp['reflections'][:]) # Bx4
            loss = torch.from_numpy(tmp['losses'][:]) # B
            S = torch.ones_like(loss, dtype=torch.bool)
            X_list.append(X)
            loss_list.append(loss)
            S_list.append(S)

        sol = MSOSolution(X=torch.stack(X_list, 0),
                          F=torch.stack(loss_list, 0),
                          S=torch.stack(S_list, 0))

        if is_gt:
            num_witness = 10
            witness_list = []
            for i in range(num_witness):
                witness = torch.from_numpy(scenes[i]['itr_0']['reflections'][:])
                witness = witness.unsqueeze(0).expand(len(scenes), -1, -1)
                witness_list.append(witness)
                if fast and i == 1:
                    break
            return sol, witness_list
        return sol

    def make_eval_dir(self, method):
        eval_dir = self.path_helper.app_dir / 'eval' / self.path_helper.format_exp_name(
            self.problem, method)
        if not eval_dir.exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir

def proc(*, method, obj_threshold, evaluation, witness_list,
         precision_thresholds, gt_sol, cuda):
    print('Evaluating method {} with threshold {} ...'.format(method, obj_threshold))
    sol = evaluation.load_solution(method, is_gt=False)

    results = []
    for j, witness in enumerate(witness_list):
        mso_eval = MSOEvaluation(witness=witness, inf_dist=inf_dist,
                                 obj_threshold=obj_threshold,
                                 relative_obj=False,
                                 precision_thresholds=precision_thresholds,
                                 gt_solution=gt_sol,
                                 candidate_solution=sol,
                                 cuda=cuda)
        mso_eval.eval()
        results.append(mso_eval.avg_results)

    # Average results
    final_mean_result = {}
    final_std_result = {}
    for key in results[0]:
        final_mean_result[key] = torch.stack([r[key] for r in results], -1).mean(-1)
        final_std_result[key] = torch.stack([r[key] for r in results], -1).std(-1)

    result_dict = {}
    for key in final_mean_result:
        result_dict[key] = {}
        if final_mean_result[key].ndim == 0:
            result_dict[key]['mean'] = final_mean_result[key].item()
            result_dict[key]['std'] = final_std_result[key].item()
        else:
            result_dict[key]['mean'] = final_mean_result[key].tolist()
            result_dict[key]['std'] = final_std_result[key].tolist()

    return result_dict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--candidate_methods', type=str, nargs='+', required=True)
    parser.add_argument('--gt_method', type=str, required=True)
    parser.add_argument('--mp', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    obj_thresholds = [1e-4, 1e-3, 1e-2, 1e-1]
    precision_thresholds = torch.linspace(1e-3, 2e-1, 100)
    pol_itr = 20
    gol_itr = 99
    method_itr = {
        'pol_res_dp_mot_hor10': pol_itr,
        'gol_single_res_hlr': gol_itr,
    }

    evaluation = Evaluation(root_dir='.', problem=args.problem,
                            method_itr=method_itr)

    gt_method = args.gt_method
    candidate_methods = args.candidate_methods
    gt_sol, witness_list = evaluation.load_solution(gt_method, is_gt=True)
    if args.fast:
        witness_list = [witness_list[0]]
    inf_dist = distance_sqr_matrix(witness_list[0], witness_list[0]).max().sqrt().item()
    print(f'inf_dist={inf_dist}')

    for method in candidate_methods:
        eval_dir = evaluation.make_eval_dir(method)
        json_dict = {}
        json_dict['method'] = method
        json_dict['obj_thresholds'] = obj_thresholds
        json_dict['precision_thresholds'] = precision_thresholds.tolist()
        json_dict['result_list'] = []

        if args.mp:
            from torch.multiprocessing import Pool
            def pool_fn(obj_threshold):
                return proc(method=method, obj_threshold=obj_threshold,
                            evaluation=evaluation, witness_list=witness_list,
                            precision_thresholds=precision_thresholds,
                            gt_sol=gt_sol, cuda=False)
            with Pool(processes=10) as pool:
                result_list = list(
                    tqdm.tqdm(pool.imap(pool_fn, obj_thresholds),
                              total=len(obj_thresholds)))
        else:
            result_list = []

            for obj_threshold in obj_thresholds:
                result_dict = proc(method=method, obj_threshold=obj_threshold,
                                   evaluation=evaluation, witness_list=witness_list,
                                   precision_thresholds=precision_thresholds,
                                   gt_sol=gt_sol, cuda=args.cuda)
                result_list.append(result_dict)
        json_dict['result_list'] = result_list

        import json
        with open(eval_dir / 'eval.json', 'w') as f:
            f.write(json.dumps(json_dict, indent=4))

