import torch
import numpy as np
from pol.utils.batch_eval import batch_eval


class SceneExtractor:
    def __init__(self,
                 aux_data):
        self.aux_data = aux_data

    def act(self, var_dict):
        problem = var_dict['problem']
        solver = var_dict['solver']
        scenes = problem.validate(solver, {
            **self.aux_data,
            'var_dict': var_dict,
        })
        var_dict['scenes'] = scenes
