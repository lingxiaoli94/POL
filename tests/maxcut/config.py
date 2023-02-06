import torch
import pol
import numpy as np
from pathlib import Path
import math
import copy
from pol.formulas.general import GeneralFormula
from pol.problems.maxcut import MaxCutProblem
from pol.solvers.configs.pol_config import POLConfig
from pol.solvers.configs.gol_config import GOLConfig
from pol.solvers.configs.pd_config import PDConfig
from pol.solvers.nn.mlp import ConditionalMLP
from pol.solvers.nn.residual_net import ResidualNet
from pol.utils.argparse  import parse_config_args

def generate_fixed_prior_samples(num_vertex):
    gen = torch.Generator()
    gen.manual_seed(1234)

    B = 1024
    V = num_vertex
    return 2 * np.pi * torch.rand([B, V], generator=gen)

def generate_thetas(num_instance, num_edge, seed=1235,
                    custom_thetas=None, integral_portion=0.0):
    gen = torch.Generator()
    gen.manual_seed(seed)
    thetas = torch.rand([num_instance, num_edge],
                        generator=gen)
    thetas_int = torch.randint(0, 2, [num_instance, num_edge],
                               generator=gen)
    num_integral = int(round(integral_portion * num_instance))
    if num_integral > 0:
        thetas[-num_integral:] = thetas_int[:num_integral]
    perm = torch.randperm(num_instance, generator=gen)
    thetas = thetas[perm]

    if custom_thetas is not None:
        num_custom = custom_thetas.shape[0]
        thetas = torch.cat([custom_thetas,
                            thetas[:-num_custom, :]], 0)
    return thetas

def generate_ring_graph(num_vertex):
    edges = []
    for i in range(num_vertex):
        edges.append([i, (i + 1) % num_vertex])
    return torch.tensor(edges)

def generate_complete_graph(num_vertex):
    edges = []
    for i in range(num_vertex):
        for j in range(i+1, num_vertex):
            edges.append([i, j])
    return torch.tensor(edges)


def prepare_problems(args):
    base_problems = {}
    num_test = 8192

    base_problems['ring'] = {
        'num_vertex': 5,
        'edges': generate_ring_graph(5),
        'train': {'num_instance': 2**14, 'seed': 1235},
        'test': {'num_instance': num_test, 'seed': 9999,
                 'custom_thetas': torch.ones([1, 5], dtype=torch.float32)},
    }

    base_problems['k5'] = {
        'num_vertex': 5,
        'edges': generate_complete_graph(5),
        'train': {'num_instance': 2**14, 'seed': 1235},
        'test': {'num_instance': num_test, 'seed': 9999,
                 'custom_thetas': torch.tensor([[
                     1, 0, 0, 1, 1, 0, 1, 1, 0, 1 # house graph
                 ]], dtype=torch.float32)},
    }

    gen = torch.Generator()
    gen.manual_seed(123)
    custom_thetas = torch.randint(0, 2, [num_test // 2, 28], generator=gen).float()
    base_problems['k8'] = {
        'num_vertex': 8,
        'edges': generate_complete_graph(8),
        'train': {'num_instance': 2**20, 'seed': 1235},
        'test': {'num_instance': num_test, 'seed': 9999,
                 'custom_thetas': custom_thetas},
    }

    base_problems['k8_mixed'] = {
        'num_vertex': 8,
        'edges': generate_complete_graph(8),
        'train': {'num_instance': 2**20, 'seed': 1235, 'integral_portion': 0.5},
        'test': {'num_instance': num_test, 'seed': 9999, 'integral_portion': 0.5},
    }

    base_problems['k8_int'] = {
        'num_vertex': 8,
        'edges': generate_complete_graph(8),
        'train': {'num_instance': 2**20, 'seed': 1235, 'integral_portion': 1.0},
        'test': {'num_instance': num_test, 'seed': 9999, 'integral_portion': 1.0},
    }

    problems = {}
    for rep in ['circle', 'angle']:
        for encoder_kind in ['gcn', 'linear', 'spectral', 'spectral_shared', 'adj']:
            for prob_name, prob_desc in base_problems.items():
                new_name = '{}_{}_{}'.format(prob_name, rep, encoder_kind)
                new_desc = copy.deepcopy(prob_desc)
                new_desc['representation'] = rep
                new_desc['encoder_kind'] = encoder_kind
                problems[new_name] = new_desc

    return problems

def prepare_methods(args):
    from pol.solvers.configs.common import simple_prepare_methods
    return simple_prepare_methods(args)

def prepare_conf(args, device, prob_desc, method_desc):
    conf_cls = method_desc['conf_cls']

    num_vertex = prob_desc['num_vertex']
    edges = prob_desc['edges']
    num_edge = edges.shape[0]
    fixed_prior_samples = generate_fixed_prior_samples(num_vertex)
    train_thetas = generate_thetas(num_edge=num_edge, **prob_desc['train'])
    test_thetas = generate_thetas(num_edge=num_edge, **prob_desc['test'])

    latent_dim = 512

    problem_formula = GeneralFormula(
        cls=MaxCutProblem,
        conf={
            'device': device,
            'num_vertex': num_vertex,
            'edges': edges,
            'latent_dim': latent_dim,
            'train_thetas': (train_thetas if conf_cls.is_universal() else test_thetas),
            'representation': prob_desc['representation'],
            'encoder_kind': prob_desc['encoder_kind']
        }
    )

    params = {
        'problem_formula': problem_formula,
        'val_aux_data': {
            'test_thetas': test_thetas,
            'include_witness': True,
            'witness_batch_size': 1024,
            'save_itrs': torch.arange(10).tolist() + ((torch.arange(10)+1) * 10).tolist(),
            'use_batch_eval': True
        },
        **method_desc['params']
    }
    if conf_cls.is_universal():
        params['latent_dim'] = latent_dim # test_thetas.shape[-1]
        var_dim = (num_vertex if prob_desc['representation'] == 'angle' else 2 * num_vertex)
        if prob_desc['encoder_kind'] == 'gcn':
            var_dim = 2
        params['var_dim'] = var_dim

    return {'params': params, 'conf_cls': conf_cls}

if __name__ == '__main__':
    from pol.solvers.configs.common import simple_schedule_main
    simple_schedule_main(prepare_problems,
                         prepare_methods,
                         prepare_conf)
