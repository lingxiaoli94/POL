import torch
import os
import pol
import numpy as np
from pathlib import Path
import copy
from pol.problems.symmetry_detection import SymmetryDetection
from pol.datasets.mesh_repository import MeshRepository
from pol.formulas.general import GeneralFormula

def prepare_problems(args):
    latent_dim = 1024
    fixed_pc_count = 2048
    common = {
        'latent_dim': latent_dim,
        'repo_params': {
            'fixed_pc_count': fixed_pc_count,
            'resample_fixed_pc': True
        },
        'params': {
            'squared_distance': False,
            'zero_latent_code': False,
        }
    }
    base_probs = {
        'hexnut': {
            **common,
            'train_catalog_file': '../../assets/hexnut_single_catalog.csv',
            'test_catalog_file': '../../assets/hexnut_single_catalog.csv',
        },
        'cube': {
            **common,
            'train_catalog_file': '../../assets/cube_single_catalog.csv',
            'test_catalog_file': '../../assets/cube_single_catalog.csv',
        },
        'toy': {
            **common,
            'train_catalog_file': '../../assets/toy_mesh_catalog.csv',
            'test_catalog_file': '../../assets/toy_mesh_catalog.csv',
        },
    }

    if 'MCB_DATASET_DIR' in os.environ:
        mcb_dir = Path(os.environ['MCB_DATASET_DIR'])
        base_probs['mcb'] = {
            **common,
            'train_catalog_file': mcb_dir / 'train_catalog.txt',
            'test_catalog_file': mcb_dir / 'test_catalog.txt',
        }

    encoder_dict = {
        'pointnet': {
            'kind': 'pointnet',
            'params': {
            }
        },
        'dgcnn': {
            'kind': 'dgcnn',
            'params': {
            }
        }
    }

    problems = {}
    for prob_name, prob_desc in base_probs.items():
        for enc_name, enc_desc in encoder_dict.items():
            new_desc = copy.deepcopy(prob_desc)
            new_desc['encoder_kind'] = enc_desc['kind']
            new_desc['encoder_params'] = enc_desc['params']
            problems['{}_{}'.format(prob_name, enc_name)] = new_desc

            new_desc = copy.deepcopy(new_desc)
            new_desc['params']['normal_weight'] = 1.0
            problems['{}_nm_{}'.format(prob_name, enc_name)] = new_desc
    return problems

def prepare_methods(args):
    from pol.solvers.configs.common import simple_prepare_methods
    methods = simple_prepare_methods(args)
    return methods

def from_catalog_file_to_repo(catalog_file, repo_params):
    mesh_list = [f.strip() for f in open(catalog_file).readlines()]
    parent_dir = Path(catalog_file).parent
    mesh_list = [(parent_dir / f) for f in mesh_list]
    repo = MeshRepository(mesh_list, **repo_params)
    return repo

def prepare_conf(args, device, prob_desc, method_desc):
    conf_cls = method_desc['conf_cls']

    var_dim = 4
    latent_dim = prob_desc['latent_dim']

    train_repo = from_catalog_file_to_repo(
        prob_desc['train_catalog_file'], prob_desc['repo_params'])
    test_repo = from_catalog_file_to_repo(
        prob_desc['test_catalog_file'], prob_desc['repo_params'])

    problem_formula = GeneralFormula(
        cls=SymmetryDetection,
        conf={
            **prob_desc['params'],
            'device': device,
            'train_repo': (train_repo if conf_cls.is_universal() else test_repo),
            'latent_dim': latent_dim,
            'encoder_kind': prob_desc['encoder_kind'],
            'encoder_params': prob_desc['encoder_params'],
        }
    )

    from pol.solvers.configs.gol_config import GOLConfig
    if conf_cls == GOLConfig:
        itr_whitelist = list(range(100)) + [100 * (i + 1) for i in range(10)]
    else:
        itr_whitelist = list(range(20))
    params = {
        'problem_formula': problem_formula,
        'val_aux_data': {
            'test_repo': test_repo,
            'prior_batch_size': 128,
            'num_itr': torch.as_tensor(itr_whitelist).max().item(),
            'itr_whitelist': itr_whitelist,
        },
        **method_desc['params']
    }
    if conf_cls.is_universal():
        params['latent_dim'] = latent_dim
        params['var_dim'] = var_dim
    else:
        params['instance_batch_size'] = args.instance_batch_size

    return {'params': params, 'conf_cls': conf_cls}


if __name__ == '__main__':
    from pol.solvers.configs.common import simple_schedule_main
    simple_schedule_main(prepare_problems,
                         prepare_methods,
                         prepare_conf)
