import numpy as np
import torch
from pathlib import Path
from pol.utils.validation.scene_saver import load_scenes, count_h5_keys
from pol.utils.validation.mso_eval import MSOSolution, MSOEvaluation, distance_sqr_matrix
from pol.utils.path import PathHelper

class Evaluation:
    def __init__(self, *,
                 root_dir='.',
                 problem,
                 theta_range,
                 method_itr):
        self.path_helper = PathHelper(root_dir)
        self.problem = problem
        self.theta_range = theta_range

    def load_solution(self, method, is_gt, fast=False):
        scene_path = self.path_helper.locate_scene_h5(self.problem, method)
        scene = load_scenes(scene_path)[0]
        if method in method_itr:
            k = method_itr[method]
        else:
            k = count_h5_keys(scene, 'itr') - 1
        tmp = scene['itr_{}'.format(k)]
        X = torch.from_numpy(tmp['X'][:])
        loss = torch.from_numpy(tmp['loss'][:])
        S = torch.from_numpy(tmp['satisfy'][:])

        if self.theta_range is not None:
            X = X[self.theta_range]
            loss = loss[self.theta_range]
            S = S[self.theta_range]

        sol = MSOSolution(X=X, F=loss, S=S)

        if is_gt:
            num_witness = count_h5_keys(scene['info'], 'witness')
            witness_list = []
            for i in range(num_witness):
                witness = torch.from_numpy(scene['info'][f'witness_{i}'][:])
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--candidate_methods', type=str, nargs='+', required=True)
    parser.add_argument('--gt_method', type=str, required=True)
    args = parser.parse_args()

    obj_thresholds = [0.05, 0.1, 0.2]
    precision_thresholds = torch.linspace(0.05, 0.4, 50)
    theta_range = range(0, 256)
    method_itr = {
        'pol_res_lot': 5,
        'gol_res_mlr': 100,
    }

    evaluation = Evaluation(root_dir='.', problem=args.problem,
                            theta_range=theta_range,
                            method_itr=method_itr)

    gt_method = args.gt_method
    candidate_methods = args.candidate_methods
    gt_sol, witness_list = evaluation.load_solution(gt_method, is_gt=True)
    inf_dist = distance_sqr_matrix(witness_list[0], witness_list[0]).max().sqrt().item()
    print(f'inf_dist={inf_dist}')

    for method in candidate_methods:
        eval_dir = evaluation.make_eval_dir(method)
        json_dict = {}
        json_dict['method'] = method
        json_dict['obj_thresholds'] = obj_thresholds
        json_dict['precision_thresholds'] = precision_thresholds.tolist()
        json_dict['result_list'] = []
        for i, obj_threshold in enumerate(obj_thresholds):
            print('Evaluating method {} with threshold {} ...'.format(method, obj_threshold))
            sol = evaluation.load_solution(method, is_gt=False)

            results = []
            for j, witness in enumerate(witness_list):
                mso_eval = MSOEvaluation(witness=witness, inf_dist=inf_dist,
                                         obj_threshold=obj_threshold,
                                        precision_thresholds=precision_thresholds,
                                        gt_solution=gt_sol,
                                        candidate_solution=sol)
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

            json_dict['result_list'].append(result_dict)

        import json
        with open(eval_dir / 'eval.json', 'w') as f:
            f.write(json.dumps(json_dict, indent=4))

