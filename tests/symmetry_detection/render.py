import torch
import pyvista as pv
import pol
import numpy as np
from pathlib import Path
import imageio
from pol.utils.validation.scene_renderer import \
    render_tri_mesh, render_arrows


obj_list_txt = '../../assets/selected_obj_list.txt'
stage_range = [1, 5, 10, 50]


def custom_render_fn(plotter, scene, stage_id):
    mesh = scene['mesh']
    render_tri_mesh(plotter, mesh['vertices'], mesh['faces'],
                    opacity=0.6)
    tmp = scene['itr_{}'.format(stage_id)]
    R = tmp['reflections']
    N = R[:, :3]
    I = R[:, 3]
    C = N * np.expand_dims(I, -1)

    sdf_loss = tmp['losses']
    normal_loss = tmp['normal_losses']

    render_arrows(plotter, C, N+C, weight=sdf_loss, threshold=1e100)


class Config:
    def __init__(self, shape_id_list, scenes_h5_file):
        self.img_ext = '.png'

        self.camera_spec = {
            'focal_point': (0.0, 0.0, 0.0),
            'position': (4.0, 4.0, 4.0),
        }
        parent_dir = Path(scenes_h5_file).parent
        self.img_root_dir = 'images/pictures/{}/{}'.format(parent_dir.parent.stem,
                                                           parent_dir.stem)
        self.img_fmt = '.png'
        stage_order = list(stage_range)
        stage_order.reverse()
        self.render_desc_list = [
            ('shape_{:05}/stage_r_{:02}'.format(i, j), i,
             lambda plotter, scene, stage_id=j: custom_render_fn(plotter, scene, stage_id))
            for i in shape_id_list
            for j in stage_order
        ]


if __name__ == '__main__':
    from pol.utils.validation.scene_renderer import SceneRenderer
    from pol.utils.validation.scene_saver import load_scenes
    from pol.utils.path import PathHelper

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--xvfb', action='store_true', default=False)
    parser.add_argument('--problem_list', type=str, nargs='+', required=True)
    parser.add_argument('--method_list', type=str, nargs='+', required=True)
    args = parser.parse_args()

    path_helper = PathHelper('.')
    for problem in args.problem_list:
        for method in args.method_list:
            scenes_h5 = path_helper.locate_scene_h5(problem, method)
            scenes = load_scenes(scenes_h5)
            rng = np.random.default_rng(42)
            shape_id_list = range(len(scenes))
            conf = Config(
                shape_id_list=shape_id_list,
                scenes_h5_file=scenes_h5
            )
            renderer = SceneRenderer(
                img_root_dir=conf.img_root_dir,
                render_desc_list=conf.render_desc_list,
                camera_spec=conf.camera_spec,
                use_vfb=args.xvfb,
                img_ext=conf.img_ext)

            renderer.render(scenes)
