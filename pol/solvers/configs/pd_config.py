from .base_config import BaseConfig
import torch
from pol.solvers.particle_descent import ParticleDescent
from pol.runners.baseline_runner import BaselineRunner
from pol.formulas.general import GeneralFormula

class PDConfig(BaseConfig):
    @staticmethod
    def is_universal():
        return False

    def __init__(self,
                 *,
                 total_num_step,
                 num_particle,
                 snapshot_freq,
                 instance_batch_size=1024,
                 optimizer_formula=GeneralFormula(
                     cls=torch.optim.Adam,
                     conf={'lr': 1e-2, 'weight_decay': 0.0}
                 ),
                 val_aux_data=None,
                 **kwargs):
        super().__init__(
            runner_cls=BaselineRunner,
            **kwargs,
        )
        self.ckpt_path = 'saves/{}/ckpt.tar'.format(self.exp_path)
        self.solver_formula = GeneralFormula(
            cls=ParticleDescent,
            conf={
                'device': self.device,
                'total_num_step': total_num_step,
                'num_particle': num_particle,
                'optimizer_formula': optimizer_formula,
                'snapshot_freq': snapshot_freq,
                'instance_batch_size': instance_batch_size,
            },
        )
        self.val_aux_data = val_aux_data
        self.fixed_prior_samples = val_aux_data.get('fixed_prior_samples', None)
        self.val_path = 'scenes/{}/result.h5'.format(self.exp_path)
