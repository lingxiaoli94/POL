import copy
import torch
from pol.solvers.nn.simple_nvp import SimpleNVP
from pol.solvers.nn.residual_net import ResidualNet
from pol.solvers.nn.mlp import ConditionalMLP
from pol.formulas.general import GeneralFormula

def complete_conditional_mlp_params(var_dim, latent_dim,
                                    params):
    return {
        'in_dim': var_dim,
        'feature_dim': latent_dim,
        'out_dim': var_dim,
        **params,
    }


def complete_nvp_params(var_dim, latent_dim,
                        params):
    params = {
        'num_layers': var_dim * params['nvp_layers'],
        'feature_dim': latent_dim,
        'input_dim': var_dim,
        **params,
    }
    return {k: v for k, v in params.items() if k not in ['nvp_layers']}

def complete_res_params(var_dim, latent_dim,
                        params):
    params = {
        'feature_dim': latent_dim,
        'input_dim': var_dim,
        **params,
    }
    return {k: v for k, v in params.items() if k not in ['nvp_layers']}


'''
Fill in network configs with var_dim and latent_dim.
'''


def build_network_formula(var_dim, latent_dim, network_conf):

    if network_conf['cls'] == ConditionalMLP:
        params = complete_conditional_mlp_params(var_dim,
                                                 latent_dim,
                                                 network_conf['params'])
    elif network_conf['cls'] == SimpleNVP:
        params = complete_nvp_params(var_dim,
                                     latent_dim,
                                     network_conf['params'])
    elif network_conf['cls'] == ResidualNet:
        params = complete_res_params(var_dim,
                                     latent_dim,
                                     network_conf['params'])
    else:
        params = copy.copy(network_conf['params'])

    return GeneralFormula(network_conf['cls'],
                          params)


def build_validator_list(*,
                         loss_watch_dict,
                         num_train_step,
                         val_freq,
                         val_save_freq,
                         val_aux_data,
                         exp_path):
    from pol.utils.validation.validation import ValidatorFormula
    from pol.utils.validation.scene_extractor import SceneExtractor
    from pol.utils.validation.variable_watcher import VariableWatcher
    from pol.utils.validation.scene_saver import SceneSaver

    val_freq = (num_train_step if val_freq == -1
                else val_freq)
    validator_list = [
        ValidatorFormula(
            10, VariableWatcher,
            {
                'watch_dict': loss_watch_dict
            }
        ),
    ]
    if val_aux_data is not None:
        val_save_freq = (num_train_step if val_save_freq == -1
                         else val_save_freq)
        validator_list.extend([
            ValidatorFormula(
                val_freq, SceneExtractor,
                {
                    'aux_data': val_aux_data
                }
            ),
            ValidatorFormula(
                val_save_freq, SceneSaver,
                {
                    'folder': 'scenes/{}'.format(exp_path),
                    'rm_exists': False
                }
            )])
    return validator_list

def simple_prepare_methods(args):
    val_freq = args.val_freq
    instance_batch_size = args.instance_batch_size
    prior_batch_size = args.prior_batch_size
    train_step = args.train_step
    num_particle = args.num_particle

    network_confs = {
        'mlp': {
            'cls': ConditionalMLP,
            'params': {
                'hidden_layer_sizes': [128] * 5,
                'activation': 'relu',
            }
        },
        'res': {
            'cls': ResidualNet,
            'params': {
                'num_block': 3,
                'hidden_size': 128,
                'input_proj_dim': 128,
                'feature_proj_dim': 128,
            }
        },
        'res_dp': {
            'cls': ResidualNet,
            'params': {
                'num_block': 5,
                'hidden_size': 128,
                'input_proj_dim': 128,
                'feature_proj_dim': 128,
            }
        },
        'res_vdp': {
            'cls': ResidualNet,
            'params': {
                'num_block': 10,
                'hidden_size': 512,
                'input_proj_dim': 512,
                'feature_proj_dim': 512,
            }
        },
    }

    network_confs['res_ns'] = copy.deepcopy(network_confs['res'])
    network_confs['res_ns']['params']['include_scaling'] = False

    weight_dict = {
        'not': 0.0,
        'llot': 0.01,
        'lmot': 0.05,
        'lot': 0.1,
        'mlot': 0.5,
        'mot': 1.0,
        'hmot': 4.0,
        'hot': 10.0,
        'hhot': 100.0,
        'hgot': 400.0,
    }
    methods = {}

    from pol.solvers.configs.pol_config import POLConfig

    for weight_name, transport_weight in weight_dict.items():
        for net_name, net_conf in network_confs.items():
            method_name = 'pol_{}_{}'.format(net_name, weight_name)
            methods[method_name] = {
                'conf_cls': POLConfig,
                'params': {
                    'num_horizon': 5,
                    'instance_batch_size': instance_batch_size,
                    'num_train_step': train_step,
                    'val_freq': val_freq,
                    'prior_batch_size': prior_batch_size,
                    'transport_weight': transport_weight,
                    'network_conf': network_confs[net_name],
                }
            }

            hor10 = copy.deepcopy(methods[method_name])
            hor10['params']['num_horizon'] = 10
            methods[method_name + '_hor10'] = hor10

            hor1 = copy.deepcopy(methods[method_name])
            hor1['params']['num_horizon'] = 1
            methods[method_name + '_hor1'] = hor1

            hor1_hlr = copy.deepcopy(methods[method_name])
            hor1_hlr['params']['optimizer_formula'] = GeneralFormula(
                cls=torch.optim.Adam,
                conf={
                    'lr': 1e-3,
                    'weight_decay': 0.0,  #1e-8
                })
            methods[method_name + '_hor1_hlr'] = hor1_hlr


    lr_dict = {
        'lllr': 1e-5,
        'llr': 1e-4,
        'mlr': 1e-3,
        'hmlr': 5e-3,
        'hlr': 1e-2,
        'hhmlr': 5e-2,
        'hhlr': 1e-1
    }

    from pol.solvers.configs.gol_config import GOLConfig
    from pol.solvers.configs.pd_config import PDConfig
    for lr_name, lr in lr_dict.items():
        for net_name, net_conf in network_confs.items():
            method_name = 'gol_{}_{}'.format(net_name, lr_name)
            methods[method_name] ={
                'conf_cls': GOLConfig,
                'params': {
                    'num_horizon': 5,
                    'num_grad_step': 10,
                    'lr': lr,
                    'instance_batch_size': instance_batch_size,
                    'num_train_step': train_step,
                    'val_freq': val_freq,
                    'prior_batch_size': prior_batch_size,
                    'network_conf': network_confs[net_name],
                }
            }

            gol_single = copy.deepcopy(methods[method_name])
            gol_single['params']['num_grad_step'] = 1
            methods['gol_single_{}_{}'.format(net_name, lr_name)] = gol_single

        methods['pd_{}'.format(lr_name)] = {
            'conf_cls': PDConfig,
            'params': {
                'total_num_step': train_step,
                'num_particle': num_particle,
                'snapshot_freq': train_step // 10,
                'optimizer_formula': GeneralFormula(
                    cls=torch.optim.Adam,
                    conf={
                        'lr': lr,
                        'weight_decay': 0.0,  #1e-8
                    }),
            }
        }

    # Add versions with weight decay.
    decay_methods = {}
    for method_name, method_desc in methods.items():
        if method_name.startswith('pol') or method_name.startswith('gol'):
            decay_method = copy.deepcopy(method_desc)
            decay_method['params']['optimizer_formula'] = GeneralFormula(
                     cls=torch.optim.Adam,
                     conf={'lr': 1e-4, 'weight_decay': 0.0}
                 )
            decay_methods['{}_nodecay'.format(method_name)] = decay_method

    methods.update(decay_methods)

    return methods


def simple_schedule_main(problem_prepare_fn,
                         method_prepare_fn,
                         conf_prepare_fn):
    '''
    Args:
        problem_prepare_fn: takes args and returns a dict
        method_prepare_fn: takes args and returns a dict
        conf_prepare_fn: (device, prob_desc, method_desc) -> {
            'params': {...}
            'conf_cls': PDConfig, POLConfig, or GOLConfig
        }
    '''
    from pol.runners.universal_runner import UniversalRunner
    from pol.runners.baseline_runner import BaselineRunner
    from pol.utils.argparse  import parse_config_args

    args = parse_config_args()
    problems = problem_prepare_fn(args)
    methods = method_prepare_fn(args)

    if args.list_problems:
        print([p for p in problems])
        return
    if args.list_methods:
        print([m for m in methods])
        return

    if args.problem_list:
        problems = {p: v for p, v in problems.items() if p in args.problem_list}
    if args.method_list:
        methods = {m: v for m, v in methods.items() if m in args.method_list}

    device = torch.device('cuda')

    for prob_name, prob_desc in problems.items():
        for method_name, method_desc in methods.items():
            exp_path = '{}/{}'.format(prob_name, method_name)
            conf_params = {
                'device': device,
                'seed': args.seed,
                'exp_path': exp_path,
            }
            tmp = conf_prepare_fn(
                args,
                device,
                prob_desc, method_desc
            )
            conf_params.update(tmp['params'])
            conf_cls = tmp['conf_cls']
            conf = conf_cls(**conf_params)
            if conf_cls.is_universal():
                runner = UniversalRunner(conf)
            else:
                runner = BaselineRunner(conf)
            print('Running experiment {}...'.format(exp_path))
            runner.run(validate_only=args.val, restart=args.restart)
