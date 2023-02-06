import matplotlib.pyplot
import numpy as np


class VariableWatcher:
    def __init__(self, watch_dict):
        self.watch_dict = watch_dict

    def act(self, var_dict):
        writer = var_dict['tb_writer']
        global_step = var_dict['global_step']
        for k, v in self.watch_dict.items():
            if k in var_dict:
                w = var_dict[k]
                if isinstance(w, matplotlib.pyplot.Figure):
                    writer.add_figure(v, w, global_step=global_step)
                elif isinstance(w, np.ndarray):
                    writer.add_image(v, w, dataformats='HWC', global_step=global_step)
                else:
                    writer.add_scalar(v, w, global_step=global_step)
