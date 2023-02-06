import torch
import pol
import numpy as np
from pathlib import Path
import copy
from pol.formulas.general import GeneralFormula
from pol.problems.analytical import AnalyticalProblem
from pol.datasets.analytical import make_analytical_loss_fn
from pol.solvers.configs.pol_config import POLConfig
from pol.solvers.configs.gol_config import GOLConfig
from pol.solvers.configs.pd_config import PDConfig
from pol.utils.argparse  import parse_config_args
from pol.datasets.analytical import \
    ProductSin, Rastrigin, ConicSection, L1Norm

def ISTA(X, t=1/2):
    # X: ...
    # ISTA(y) = argmin_y t||y||_1 + 0.5||x-y||_2^2
    cond1 = X >= t
    cond2 = torch.logical_and(X < t, X > -t)
    cond3 = X <= -t

    result = torch.where(cond1, X - t, X)
    result = torch.where(cond2, torch.zeros_like(X), result)
    result = torch.where(cond3, X + t, result)
    return result

def prepare_problems(args):
    problems = {
        'prodsin_2d': {
            'prob': ProductSin(
                dim=2,
                var_bbox=[[-8, 8], [-8, 8]]),
            'train': {'num_instance': 2**20, 'seed': 42},
            'test': {'num_instance': 256, 'seed': 9999},
        },
        'rastrigin_2d': {
            'prob': Rastrigin(
                dim=2,
                var_bbox=[[-5, 5], [-5, 5]]),
            'train': {'num_instance': 1, 'seed': 42},
            'test': {'num_instance': 1, 'seed': 42},
        },
        'rastrigin_2d_degen': {
            'prob': Rastrigin(
                dim=2,
                var_bbox=[[-5, 5], [-5, 5]]),
            'train': {'num_instance': 1, 'seed': 0},
            'test': {'num_instance': 1, 'seed': 0},
        },
        'conic': {
            'prob': ConicSection(
                var_bbox=[[-5, 5], [-5, 5]], ellipse_only=False),
            'train': {'num_instance': 2**20, 'seed': 42},
            'test': {'num_instance': 256, 'seed': 9999},
        },
    }

    for dim in [2, 4, 8, 16, 32]:
        problems['l1_norm_{}d'.format(dim)] = {
            'prob': L1Norm(var_bbox=[[-1, 1]] * dim),
            'train': {'num_instance': 2**20, 'seed': 42},
            'test': {'num_instance': 256, 'seed': 9999},
        }
    return problems


def prepare_methods(args):
    from pol.solvers.configs.common import simple_prepare_methods
    return simple_prepare_methods(args)

def prepare_conf(args, device, prob_desc, method_desc):
    conf_cls = method_desc['conf_cls']

    prob = prob_desc['prob']
    prob.change_device(device)
    var_bbox = prob.get_var_bbox()
    var_dim = var_bbox.shape[0]
    train_thetas = prob.create_parameters(**prob_desc['train'])
    test_thetas = prob.create_parameters(**prob_desc['test'])

    problem_formula = GeneralFormula(
        cls=AnalyticalProblem,
        conf={
            'device': device,
            'var_dim': var_dim,
            'fn': make_analytical_loss_fn(prob),
            'train_thetas': (train_thetas if conf_cls.is_universal() else test_thetas),
            'var_bbox': var_bbox,
        }
    )

    if isinstance(prob, L1Norm):
        gt_prox_fn = lambda X: ISTA(X, t=1/2)
        save_itrs = [0]
    else:
        gt_prox_fn = None
        save_itrs = list(range(0, 100, 5))
    params = {
        'problem_formula': problem_formula,
        'val_aux_data': {
            'test_thetas': test_thetas,
            'include_witness': True,
            'witness_batch_size': 1024,
            'num_itr': 100,
            'save_itrs': save_itrs,
            'gt_prox_fn': gt_prox_fn,
        },
        **method_desc['params']
    }
    if conf_cls.is_universal():
        params['var_dim'] = var_dim
        params['latent_dim'] = test_thetas.shape[-1]
    else if isinstance(prob, Rastrigin):
        # HACK for Rastrigin
        X = torch.linspace(-5, 5, 100)
        Y = torch.linspace(-5, 5, 100)
        grid_X, grid_Y = torch.meshgrid(X, Y, indexing='ij')
        grid = torch.stack([grid_X, grid_Y], -1) # 100x100x2
        params['val_aux_data']['loss_landscape'] = {
            'points': grid
        }

    return {'params': params, 'conf_cls': conf_cls}

if __name__ == '__main__':
    from pol.solvers.configs.common import simple_schedule_main
    simple_schedule_main(prepare_problems,
                         prepare_methods,
                         prepare_conf)
