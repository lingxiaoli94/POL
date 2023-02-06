import torch
import copy
import pol
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset
from pol.datasets.linear_regression import make_linear_regression_loss_fn
from pol.problems.supervised_learning import SupervisedLearningProblem
import pol.datasets.linear_regression
from pol.formulas.general import GeneralFormula

g_dataset_dict = {
    'toy': pol.datasets.linear_regression.create_toy1_2d_dataset(),
    '8d': pol.datasets.linear_regression.create_nd_dataset(dim=8,
                                                           num_sample=8,
                                                           seed=1234),
    '8d_uddtmd': pol.datasets.linear_regression.create_nd_dataset(dim=8,
                                                                  num_sample=4,
                                                                  seed=1234),
    '20d_uddtmd': pol.datasets.linear_regression.create_nd_dataset(dim=20,
                                                                   num_sample=10,
                                                                   seed=1234),
    '20d': pol.datasets.linear_regression.create_nd_dataset(dim=20,
                                                            num_sample=20,
                                                            seed=1234),
    '20d_noisy': pol.datasets.linear_regression.create_nd_dataset(dim=20,
                                                                  num_sample=1024,
                                                                  noise=0.2,
                                                                  seed=1234),
}


def prepare_problems(args):
    from pol.datasets.linear_regression import \
        NormRegularizer, \
        LogSumRegularizer, \
        NormMixedRegularizer, \
        MCPRegularizer
    return {
        'lasso_20d': {
            'dataset': '20d',
            'regularizer': NormRegularizer(p=1.0, eps=1e-8)
        },
        'lasso_20d_noisy': {
            'dataset': '20d_noisy',
            'regularizer': NormRegularizer(p=1.0, eps=1e-8)
        },
        'lasso_20d_uddtmd': {
            'dataset': '20d_uddtmd',
            'regularizer': NormRegularizer(p=1.0, eps=1e-8)
        },
        'quasinorm_toy': {
            'dataset': 'toy',
            'regularizer': NormRegularizer(p=0.5, eps=1e-8)
        },
        'quasinorm_8d': {
            'dataset': '8d_uddtmd',
            'regularizer': NormRegularizer(p=0.5, eps=1e-8)
        },
        'mcp_8d': {
            'dataset': '8d_uddtmd',
            'regularizer': MCPRegularizer(sigma_range=[0.5, 2])
        },
        'logsum_toy': {
            'dataset': 'toy',
            'regularizer': LogSumRegularizer(sigma=2.0)
        },
        'logsum_8d': {
            'dataset': '8d_uddtmd',
            'regularizer': LogSumRegularizer(sigma=2.0)
        },
        'mixed_8d': {
            'dataset': '8d_uddtmd',
            'regularizer': NormMixedRegularizer(p_bbox=[0.2, 1.0], eps=1e-8)
        },
        'mixed_ncvx_8d': {
            'dataset': '8d_uddtmd',
            'regularizer': NormMixedRegularizer(p_bbox=[0.2, 0.5], eps=1e-8)
        },
    }


def prepare_methods(args):
    from pol.solvers.configs.common import simple_prepare_methods
    return simple_prepare_methods(args)


def prepare_conf(args, device, prob_desc, method_desc):
    conf_cls = method_desc['conf_cls']

    regularizer = prob_desc['regularizer']
    dataset = g_dataset_dict[prob_desc['dataset']]
    var_dim = dataset[0][0].shape[0]
    train_thetas = regularizer.create_parameters(num_instance=1024, seed=42)
    test_thetas = regularizer.create_parameters(num_instance=128, seed=9999)
    # test_thetas = regularizer.create_parameters(num_instance=128, seed=9998) # for rebuttal only
    # For now assume weights are all in [-2, 2]
    var_bbox = torch.tensor([-2.0, 2.0], dtype=torch.float).unsqueeze(0).expand(var_dim, 2)  # Dx2

    problem_formula = GeneralFormula(
        cls=SupervisedLearningProblem,
        conf={
            'device': device,
            'weight_dim': var_dim,
            'loss_fn': make_linear_regression_loss_fn(regularizer),
            'dataset': dataset,
            'eval_batch_size': 32,
            'train_thetas': (train_thetas if conf_cls.is_universal() else test_thetas),
            'weight_bbox': var_bbox,
        }
    )

    params = {
        'problem_formula': problem_formula,
        'val_aux_data': {
            'test_thetas': test_thetas,
            'include_witness': True,
            'witness_batch_size': 4096,
            'prior_batch_size': 4096,
            'use_batch_eval': True,
            'save_itrs': [1000],
            # 'save_itrs': list(range(0, 200, 1))
        },
        **method_desc['params']
    }
    if conf_cls.is_universal():
        params['var_dim'] = var_dim
        params['latent_dim'] = regularizer.get_lambda_dim()

    return {'params': params, 'conf_cls': conf_cls}


if __name__ == '__main__':
    from pol.solvers.configs.common import simple_schedule_main
    simple_schedule_main(prepare_problems,
                         prepare_methods,
                         prepare_conf)
