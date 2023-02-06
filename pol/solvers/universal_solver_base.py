from abc import ABC, abstractmethod
from pol.problems.problem_base import ProblemBase
import torch

class UniversalSolverBase(ABC):
    def set_problem(self, problem):
        self.problem = problem

    def is_universal(self):
        return True

    def extract_pushed_samples(self, data_batch, prior_samples):
        pass

    @abstractmethod
    def compute_losses(self, data_batch, prior_samples):
        pass
