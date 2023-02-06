import imageio
from pathlib import Path
import shutil


class ImageWriter:
    def __init__(self, folder, scene_fig_key, fmt='step-{:05}'):
        self.dir = Path(folder)
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.scene_fig_key = scene_fig_key
        self.fmt = fmt

    def act(self, var_dict):
        step = var_dict['global_step']
        imageio.imwrite(
            self.dir / (self.fmt.format(step) + '.png'),
            var_dict[self.scene_fig_key],
            format='.png'
        )
