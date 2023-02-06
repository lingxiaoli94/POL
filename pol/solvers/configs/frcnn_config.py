from .base_config import BaseConfig
from .common import build_network_formula, \
    build_validator_list
import torch
import copy
from pol.runners.universal_runner import UniversalRunner
from pol.formulas.general import GeneralFormula
from pol.solvers.frcnn import FRCNN


class FRCNNConfig(BaseConfig):
    @staticmethod
    def is_universal():
        return True

    def __init__(self,
                 *,
                 var_dim,
                 latent_dim,
                 max_num_box,
                 threshold,
                 instance_batch_size,
                 num_train_step,
                 ckpt_save_freq=-1, val_freq=-1,
                 optimizer_formula=GeneralFormula(
                     cls=torch.optim.Adam,
                     conf={'lr': 1e-4, 'weight_decay': 0.0}
                 ),
                 val_aux_data=None,
                 val_save_freq=-1,
                 **kwargs):
        assert(var_dim == 4)
        super().__init__(
            runner_cls=UniversalRunner,
            **kwargs
        )
        self.log_dir = 'runs/{}'.format(self.exp_path)
        self.ckpt_path = 'saves/{}/ckpt.tar'.format(self.exp_path)
        self.weights = {
        }
        self.ckpt_save_freq = (num_train_step // 5 if ckpt_save_freq == -1
                               else ckpt_save_freq)
        self.instance_batch_size = instance_batch_size
        self.prior_batch_size = 1 # dummy
        self.num_train_step = num_train_step

        self.solver_formula = GeneralFormula(
            cls=FRCNN,
            conf={
                'device': self.device,
                'max_num_box': max_num_box,
                'threshold': threshold,
            },
        )
        self.optimizer_formula = optimizer_formula
        loss_watch_dict = {
        }
        self.validator_list = build_validator_list(
            loss_watch_dict=loss_watch_dict,
            num_train_step=num_train_step,
            val_freq=val_freq,
            val_save_freq=val_save_freq,
            val_aux_data=val_aux_data,
            exp_path=self.exp_path
        )
