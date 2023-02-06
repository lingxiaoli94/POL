import torch
from tqdm import trange
import numpy as np
import random
import itertools
from pathlib import Path
import shutil
import os
from pol.utils.device import transfer_data_batch


class UniversalRunner:
    def __init__(self, conf):
        self.conf = conf

    def load_ckpt(self, problem, solver, optimizer, ckpt_path):
        p = Path(ckpt_path)
        if not p.exists():
            print('No checkpoint file found. Use default initialization.')
            return 1
        ckpt = torch.load(ckpt_path)
        if 'problem_state_dict' in ckpt:
            problem.load_state_dict(ckpt['problem_state_dict'])
        solver.load_state_dict(ckpt['solver_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        global_step = ckpt['global_step']
        print('Loading solver from {} at step {}...'.format(ckpt_path, global_step))

        np.random.set_state(ckpt['np_rng_state'])
        torch.set_rng_state(ckpt['torch_rng_state'])
        return global_step + 1  # return the next step

    def save_ckpt(self, problem, solver, optimizer, global_step, ckpt_path):
        print('Saving solver at global step {}...'.format(global_step))
        p = Path(ckpt_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            shutil.copyfile(p, p.with_suffix('.tar.bkup'))

        all_dict = {
            'solver_state_dict': solver.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state()
        }
        if problem.get_nn_parameters() is not None:
            all_dict['problem_state_dict'] = problem.get_state_dict()

        torch.save(all_dict, ckpt_path)

    def seed_all(self, seed):
        print('Using seed {} for everything.'.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self, validate_only=False, restart=False, magma=True):
        seed = getattr(self.conf, 'seed', None)
        if seed:
            self.seed_all(seed)

        problem_fml = self.conf.problem_formula
        solver_fml = self.conf.solver_formula
        optimizer_fml = self.conf.optimizer_formula

        problem = problem_fml.create_instance()
        solver = solver_fml.create_instance()

        from pol.utils.model_size import compute_model_size
        print('Solver model size: {:3f}MB'.format(compute_model_size(solver)))

        val_dict = {
            'problem': problem,
            'solver': solver,
        }

        solver.set_problem(problem)

        nn_params = [solver.parameters()]
        if problem.get_nn_parameters() is not None:
            # Optionally add encoder parameters.
            nn_params.append(problem.get_nn_parameters())

        optimizer = optimizer_fml.create_instance(itertools.chain(*nn_params))

        init_global_step = 1
        if getattr(self.conf, 'ckpt_path', None):
            val_dict['ckpt_path'] = self.conf.ckpt_path
            if (not restart) or validate_only:
                init_global_step = self.load_ckpt(problem, solver, optimizer, val_dict['ckpt_path'])

        from pol.utils.validation.validation import ValidatorPlan
        val_plan = ValidatorPlan(self.conf.validator_list)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=self.conf.log_dir)
        val_dict['tb_writer'] = writer
        if validate_only:
            problem.set_train_mode(False)
            print('Running validation only at step {}...'.format(init_global_step - 1))
            val_dict['global_step'] = init_global_step - 1
            val_plan.tick(val_dict, tick_all=True)
        else:
            from torch.utils.data import TensorDataset, DataLoader
            train_dataloader = problem.create_train_dataloader(self.conf.instance_batch_size)
            train_data_itr = iter(train_dataloader)

            loop_range = trange(self.conf.num_train_step - init_global_step + 1)
            for i in loop_range:
                problem.set_train_mode(True)
                global_step = init_global_step + i
                try:
                    data_batch = next(train_data_itr)
                except StopIteration:
                    train_data_itr = iter(train_dataloader)
                    data_batch = next(train_data_itr)

                data_batch = transfer_data_batch(data_batch, self.conf.device)
                prior_samples = problem.sample_prior(data_batch,
                                                     self.conf.prior_batch_size)

                losses = solver.compute_losses(data_batch, prior_samples)
                losses = {k: v.mean() for k, v in losses.items()}
                total_loss = 0
                for k, w in self.conf.weights.items():
                    if w > 0:
                        if k not in losses:
                            raise Exception("Cannot find {} among losses!".format(k))
                        total_loss = total_loss + w * losses[k]

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                loop_range.set_description('Total loss: {:6f}'.format(total_loss.item()))

                val_dict.update({
                    'global_step': global_step,
                    'total_loss': total_loss.item(),
                })
                for k, w in self.conf.weights.items():
                    if w > 0:
                        val_dict[k] = losses[k].item()

                # Save before validation.
                if 'ckpt_path' in val_dict and getattr(self.conf, 'ckpt_save_freq', None):
                    if val_dict['global_step'] % self.conf.ckpt_save_freq == 0:
                        self.save_ckpt(problem, solver, optimizer, val_dict['global_step'],
                                       ckpt_path=val_dict['ckpt_path'])

                problem.set_train_mode(False)
                val_plan.tick(val_dict)
