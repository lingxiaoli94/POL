from .base_config import BaseConfig
from .common import build_network_formula, \
    build_validator_list
import torch
import copy
from pol.runners.universal_runner import UniversalRunner
from pol.formulas.general import GeneralFormula
from pol.solvers.gol import GOL


class GOLConfig(BaseConfig):
    @staticmethod
    def is_universal():
        return True

    def __init__(self,
                 *,
                 num_grad_step,
                 num_horizon,
                 var_dim,
                 latent_dim,
                 instance_batch_size,
                 prior_batch_size,
                 num_train_step,
                 ckpt_save_freq=-1, val_freq=-1,
                 network_conf,
                 lr=1e-3,
                 optimizer_formula=GeneralFormula(
                     cls=torch.optim.Adam,
                     conf={'lr': 1e-4, 'weight_decay': 1e-6}
                 ),
                 val_aux_data=None,
                 val_save_freq=-1,
                 **kwargs):
        super().__init__(
            runner_cls=UniversalRunner,
            **kwargs
        )
        self.log_dir = 'runs/{}'.format(self.exp_path)
        self.ckpt_path = 'saves/{}/ckpt.tar'.format(self.exp_path)
        self.weights = {
            'regression_loss': 1.0, # only 1 loss
        }
        self.ckpt_save_freq = (num_train_step // 5 if ckpt_save_freq == -1
                               else ckpt_save_freq)
        self.instance_batch_size = instance_batch_size
        self.prior_batch_size = prior_batch_size
        self.num_train_step = num_train_step

        self.solver_formula = GeneralFormula(
            cls=GOL,
            conf={
                'device': self.device,
                'num_grad_step': num_grad_step,
                'num_horizon': num_horizon,
                'network_formula': build_network_formula(
                    var_dim, latent_dim, network_conf),
                'lr': lr,
            },
        )
        self.optimizer_formula = optimizer_formula
        self.validator_list = build_validator_list(
            loss_watch_dict={
                'total_loss': 'Loss/total loss',
                'regression_loss': 'Loss/regression loss',
            },
            num_train_step=num_train_step,
            val_freq=val_freq,
            val_save_freq=val_save_freq,
            val_aux_data=val_aux_data,
            exp_path=self.exp_path
        )
