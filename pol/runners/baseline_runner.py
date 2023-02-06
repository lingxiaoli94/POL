import torch
from tqdm import trange
import numpy as np
import itertools
from pathlib import Path
import shutil
import os


class BaselineRunner:
    def __init__(self, conf):
        self.conf = conf

    def load_ckpt(self, solver, ckpt_path):
        p = Path(ckpt_path)
        if not p.exists():
            print('No checkpoint file found. Use default initialization.')
            return 1
        print('Loading solver from {} ...'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        solver.load_state_dict(ckpt['solver_state_dict'])

        np.random.set_state(ckpt['np_rng_state'])
        torch.set_rng_state(ckpt['torch_rng_state'])

    def save_ckpt(self, solver, ckpt_path):
        print('Saving solver...')
        p = Path(ckpt_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            shutil.copyfile(p, p.with_suffix('.tar.bkup'))

        all_dict = {
            'solver_state_dict': solver.state_dict(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state()
        }

        torch.save(all_dict, ckpt_path)

    def run(self, validate_only=False, restart=False):
        seed = getattr(self.conf, 'seed', None)
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)

        problem_fml = self.conf.problem_formula
        solver_fml = self.conf.solver_formula

        problem = problem_fml.create_instance()
        solver = solver_fml.create_instance()

        solver.init(problem, getattr(self.conf, 'fixed_prior_samples', None))
        ckpt_path = getattr(self.conf, 'ckpt_path', None)
        if ckpt_path:
            if (not restart) or validate_only:
                self.load_ckpt(solver, ckpt_path)

        if not validate_only:
            # For baseline, training dataset is the same as testing, which is
            # contained in the problem instance.
            solver.run(problem)

            if ckpt_path:
                self.save_ckpt(solver, ckpt_path=ckpt_path)

        print('Running validation...')
        val_result = problem.validate(solver, self.conf.val_aux_data)
        val_path = Path(self.conf.val_path)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        from pol.utils.validation.scene_saver import save_scenes
        save_scenes(val_result, val_path)
