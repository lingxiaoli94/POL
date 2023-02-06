import torch
from tqdm import trange
from pol.formulas.general import GeneralFormula
from pol.utils.device import transfer_data_batch


class ParticleDescent:
    def __init__(self,
                 device, *,
                 total_num_step,
                 num_particle,
                 optimizer_formula,
                 snapshot_freq,
                 instance_batch_size):
        self.device = device
        self.total_num_step = total_num_step
        self.num_particle = num_particle
        self.optimizer_formula = optimizer_formula
        self.history = []
        self.snapshot_freq = max(1, snapshot_freq)
        self.instance_batch_size = instance_batch_size

    def is_universal(self):
        return False

    def init(self, problem, fixed_prior_samples):
        dataset = problem.get_train_dataset()
        num_data = len(dataset)

        self.data_batch_list = []
        self.X_list = []
        self.history_list = []
        self.global_step_list = []
        self.opt_state_dict = {}

        count = 0
        while count < num_data:
            rb = min(count+self.instance_batch_size, num_data)
            data_batch = dataset[count:rb]
            self.data_batch_list.append(data_batch)

            if fixed_prior_samples is not None:
                prior_samples = fixed_prior_samples.unsqueeze(0).expand(
                    data_batch[0].shape[0], -1, -1)
                assert(prior_samples.shape[1] == self.num_particle)
            else:
                prior_samples = problem.sample_prior(data_batch, self.num_particle)
            X = prior_samples.clone()
            self.X_list.append(X)
            self.history_list.append([])
            self.global_step_list.append(0)

            count = rb

    def run_single(self, problem, i):
        assert(i < len(self.X_list))
        num_remain_step = self.total_num_step - self.global_step_list[i]
        if num_remain_step <= 0:
            return
        loop_range = trange(num_remain_step)

        data_batch = transfer_data_batch(self.data_batch_list[i], self.device)
        X = self.X_list[i].to(self.device).detach()
        X.requires_grad_(True)
        opt = self.optimizer_formula.create_instance([X])
        if i in self.opt_state_dict:
            opt.load_state_dict(self.opt_state_dict[i])

        for j in loop_range:
            X_proj = problem.apply_projection(data_batch, X)
            loss = problem.eval_loss(data_batch, X_proj)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()

            opt.step()
            self.global_step_list[i] += 1

            if self.global_step_list[i] % self.snapshot_freq == 0:
                self.history_list[i].append(X_proj.detach().cpu())
            loop_range.set_description('Batch #{}, Loss: {:6f}'.format(i, loss.item()))

        self.X_list[i] = X.detach().cpu()
        self.opt_state_dict[i] = opt.state_dict()

    def run(self, problem):
        for i in range(len(self.X_list)):
            self.run_single(problem, i)

    def state_dict(self):
        return {
            'X': self.X_list,
            'opt': self.opt_state_dict,
            'history': self.history_list,
            'global_step': self.global_step_list
        }

    def load_state_dict(self, state_dict):
        self.X_list = state_dict['X']
        self.opt_state_dict = state_dict['opt']
        self.history_list = state_dict['history']
        self.global_step_list = state_dict['global_step']

    def extract_solutions(self):
        # Combine histories.
        merged_history = []
        for k in range(len(self.history_list[0])):
            merged_history.append(torch.cat(
                [history[k] for history in self.history_list], 0))
        return merged_history
