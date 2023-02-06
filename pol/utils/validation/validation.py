import torch


class ValidatorFormula:
    def __init__(self, frequency, actor_cls, actor_cls_params):
        self.frequency = frequency
        self.actor_cls = actor_cls
        self.actor_cls_params = actor_cls_params


class Validator:
    def __init__(self, fml):
        self.frequency = fml.frequency
        self.actor = fml.actor_cls(**fml.actor_cls_params)


class ValidatorPlan:
    def __init__(self, validator_fml_list):
        self.validator_list = []
        for fml in validator_fml_list:
            self.validator_list.append(Validator(fml))

    def tick(self, var_dict, tick_all=False):
        global_step = var_dict['global_step']
        for validator in self.validator_list:
            if tick_all or global_step % validator.frequency == 0:
                validator.actor.act(var_dict)
