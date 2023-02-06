import torch
import pol
import numpy as np
from pathlib import Path
from pol.formulas.general import GeneralFormula
from pol.problems.spring_equilibrium import SpringEquilibriumProblem
from pol.solvers.configs.pol_config import POLConfig
from pol.solvers.configs.pd_config import PDConfig
from pol.solvers.nn.mlp import ConditionalMLP


def prepare_methods():
    methods = []

    # Add a proximal operator learning (POL) method.
    methods.append({
        'conf_cls': POLConfig,
        'name': 'pol_mlp',
        'params': {
            'num_horizon': 5,
            'instance_batch_size': 16,
            'num_train_step': 2000,
            'prior_batch_size': 256,
            'transport_weight': 1.0,
            'network_conf': {
                'cls': ConditionalMLP,
                'params': {
                    'hidden_layer_sizes': [128] * 5,
                    'activation': 'relu',
                    'rff_first': False,
                    'proj_first': False
                }
            },
        }
    })

    # For comparison, add a particle descent (PD) method.
    methods.append({
        'conf_cls': PDConfig,
        'name': 'pd',
        'params': {
            'total_num_step': 2000,
            'num_particle': 1024,
            'snapshot_freq': 1,
            'optimizer_formula': GeneralFormula(
                cls=torch.optim.Adam,
                conf={
                    'lr': 1e-2,
                    'weight_decay': 0.0
                }),
        }
    })

    return methods

def generate_fixed_prior_samples(num_free, width):
    gen = torch.Generator()
    gen.manual_seed(1234)

    B = 1024
    F = num_free
    x = width * torch.rand([B, F], generator=gen)
    y = 2 * torch.rand([B, F], generator=gen) - 1
    samples = torch.stack([x, y], -1) # BxFx2
    return samples.reshape(B, -1) # Bx2F

if __name__ == '__main__':
    from pol.runners.universal_runner import UniversalRunner
    from pol.runners.baseline_runner import BaselineRunner
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--restart', action='store_true', default=False)

    args = parser.parse_args()

    methods = prepare_methods()
    device = torch.device('cuda')

    num_free = 2
    width = 1.0

    var_dim = 2 * num_free  # dimension(=D) of the search space
    # For now we train/test on just one problem instance.
    train_thetas = torch.tensor([[0.5, 0.5, 0.5]]) # rest lengths
    test_thetas = train_thetas.clone()
    # We use fixed prior samples for testing, to compare the effect of
    # the learned proximal operator and that of particle descent.
    fixed_prior_samples = generate_fixed_prior_samples(num_free, width)

    common_conf = {
        'seed': 42,
        'device': device,
        'val_aux_data': {
            'test_thetas': test_thetas,
            'fixed_prior_samples': fixed_prior_samples,
            'save_itrs': [0, 5, 10, 15, 20],
        }
    }
    for method in methods:
        is_universal = not (method['conf_cls'] == PDConfig)
        problem_formula = GeneralFormula(
            cls=SpringEquilibriumProblem,
            conf={
                'device': device,
                'num_free': num_free,
                'train_thetas': (
                    train_thetas if is_universal
                    else test_thetas),
                'width': width
            }
        )

        exp_path = 'small_test/{}'.format(method['name'])
        conf_params = {
            'exp_path': exp_path,
            'problem_formula': problem_formula,
            **common_conf,
            **method['params']
        }
        if is_universal:
            conf_params['latent_dim'] = test_thetas.shape[-1]
            conf_params['var_dim'] = var_dim
        conf = method['conf_cls'](**conf_params)
        if is_universal:
            runner = UniversalRunner(conf)
        else:
            runner = BaselineRunner(conf)
        print('Running experiment {}...'.format(conf.exp_path))
        runner.run(validate_only=args.val, restart=args.restart)
