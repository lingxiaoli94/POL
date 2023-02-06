'''
Common config for all solvers.
'''
class BaseConfig:
    def __init__(self,
                 *,
                 exp_path,
                 runner_cls,
                 problem_formula,
                 seed,
                 device):
        self.exp_path = exp_path
        self.runner_cls = runner_cls
        self.seed = seed
        self.device = device
        self.problem_formula = problem_formula
