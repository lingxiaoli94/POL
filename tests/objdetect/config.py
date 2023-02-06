import torch
import pol
import numpy as np
from pathlib import Path
from pol.formulas.general import GeneralFormula
from pol.problems.objdetect import ObjectDetection
from pol.datasets.objdetect import COCODataset
from pol.solvers.configs.pol_config import POLConfig
from pol.solvers.configs.pd_config import PDConfig
from pol.solvers.nn.mlp import ConditionalMLP
from pol.solvers.nn.simple_nvp import SimpleNVP
import h5py
import copy

def prepare_problems(args):
    crop_size = 400
    problem_dict = {
        'full': {
            'train_dataset': {
                'split': 'train',
                'dataset_name': 'coco-2017-train',
                'max_num_detection': 10,
                'crop_size': crop_size,
                'evaluation': False
            },
            'test_dataset': {
                'split': 'validation',
                'dataset_name': 'coco-2017-validation',
                'max_num_detection': 10,
                'crop_size': crop_size,
                'evaluation': True,
                'keep_original': True
            },
            'latent_dim': 256,
            'box_loss_kind': 'l1',
            'use_regression': False,
            'encoder_kind': 'resnet',
            'encoder_params': {
                'pretrained': True,
                'freeze_bn': False,
            }
        }
    }

    toy_problem = copy.deepcopy(problem_dict['full'])
    toy_problem['train_dataset']['split'] = 'validation'
    toy_problem['train_dataset']['dataset_name'] = 'coco-2017-validation-toy'
    toy_problem['train_dataset']['num_take'] = 32
    toy_problem['test_dataset'] = copy.deepcopy(toy_problem['train_dataset'])
    toy_problem['test_dataset']['evaluation'] = True
    toy_problem['test_dataset']['keep_original'] = True
    problem_dict['toy'] = toy_problem

    new_dict = {}
    for prob_name, prob_desc in problem_dict.items():
        new_desc = copy.deepcopy(prob_desc)
        new_desc['encoder_params']['pe_kind'] = 'none'
        new_dict[f'{prob_name}_npe'] = new_desc

        new_desc = copy.deepcopy(prob_desc)
        new_desc['encoder_params']['pe_kind'] = 'sine'
        new_dict[f'{prob_name}_spe'] = new_desc

        new_desc = copy.deepcopy(prob_desc)
        new_desc['encoder_params']['pe_kind'] = 'rand'
        new_dict[f'{prob_name}_rpe'] = new_desc

    problem_dict.update(new_dict)

    return problem_dict

def prepare_methods(args):
    from pol.solvers.configs.common import simple_prepare_methods
    methods = simple_prepare_methods(args)

    from pol.solvers.configs.fns_config import FNSConfig
    methods['fns'] = {
        'conf_cls': FNSConfig,
        'params': {
            'max_num_box': 20,
            'instance_batch_size': args.instance_batch_size,
            'num_train_step': args.train_step,
            'val_freq': args.val_freq,
        }
    }

    methods['fns_sigmoid'] = copy.deepcopy(methods['fns'])
    methods['fns_sigmoid']['params']['apply_sigmoid'] = True

    from pol.solvers.configs.frcnn_config import FRCNNConfig
    methods['frcnn'] = {
        'conf_cls': FRCNNConfig,
        'params': {
            'max_num_box': 20,
            'instance_batch_size': args.instance_batch_size,
            'num_train_step': args.train_step,
            'val_freq': args.val_freq,
        }
    }

    methods['frcnn50'] = copy.deepcopy(methods['frcnn'])
    methods['frcnn50']['params']['threshold'] = 0.5

    methods['frcnn65'] = copy.deepcopy(methods['frcnn'])
    methods['frcnn65']['params']['threshold'] = 0.65

    methods['frcnn80'] = copy.deepcopy(methods['frcnn'])
    methods['frcnn80']['params']['threshold'] = 0.80

    methods['frcnn95'] = copy.deepcopy(methods['frcnn'])
    methods['frcnn95']['params']['threshold'] = 0.95

    return methods

def prepare_conf(args, device, prob_desc, method_desc):
    conf_cls = method_desc['conf_cls']

    var_dim = 4
    latent_dim = prob_desc['latent_dim']

    if args.val:
        train_dataset = None
    else:
        train_dataset = COCODataset(**prob_desc['train_dataset'])
    test_dataset = COCODataset(**prob_desc['test_dataset'])

    problem_formula = GeneralFormula(
        cls=ObjectDetection,
        conf={
            'device': device,
            'train_dataset': train_dataset,
            'latent_dim': latent_dim,
            'box_loss_kind': prob_desc['box_loss_kind'],
            'use_regression': prob_desc['use_regression'],
            'encoder_kind': prob_desc['encoder_kind'],
            'encoder_params': {**prob_desc['encoder_params']},
        }
    )

    from pol.solvers.configs.gol_config import GOLConfig
    if conf_cls == GOLConfig:
        itr_whitelist = list(range(100)) + [100 * (i + 1) for i in range(50)]
    else:
        itr_whitelist = list(range(10)) + [10 * (i + 1) for i in range(10)]
    params = {
        'seed': 42,
        'problem_formula': problem_formula,
        'ckpt_save_freq': 2500,
        'val_aux_data': {
            'test_dataset': test_dataset,
            'prior_batch_size': 1024,
            'instance_batch_size': 4,
            'num_itr': torch.as_tensor(itr_whitelist).max().item(),
            'itr_whitelist': itr_whitelist,
        },
        **method_desc['params']
    }
    if conf_cls.is_universal():
        params['latent_dim'] = latent_dim
        params['var_dim'] = var_dim

    return {'params': params, 'conf_cls': conf_cls}

if __name__ == '__main__':
    from pol.solvers.configs.common import simple_schedule_main
    simple_schedule_main(prepare_problems,
                         prepare_methods,
                         prepare_conf)
