import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt
from pol.utils.pv_utils import decorate_tri_faces
from pol.utils.validation.scene_saver import load_scenes
import multiprocessing as mp
import imageio
import tqdm
from pathlib import Path


def render_ndarray_dict(plotter, obj, custom_fn):
    # Modify plotter in place.
    custom_fn(plotter, obj)


def render_plane(plotter, center, normal,
                 arrow_scale=0.2,
                 center_radius=0.02,
                 opacity=0.5):
    arrow = pv.Arrow(center, normal, scale=arrow_scale)
    plotter.add_mesh(arrow, color='blue')
    sphere = pv.Sphere(radius=center_radius, center=center)
    plotter.add_mesh(sphere, color='purple')
    if opacity > 0:
        plane = pv.Plane(center, normal,
                         i_size=5, j_size=5)
        plotter.add_mesh(plane, color='green', opacity=opacity)


def render_arrows(plotter, src, dst, weight=None, vrange=slice(None),
                  threshold=1e100, weight_clip=1e100, line_width=2.0, show_endpoints=True,
                  show_scalar_bar=True, cmap='rainbow'):
    assert(src.shape[0] == dst.shape[0])
    src = src[vrange, :]
    dst = dst[vrange, :]
    count = src.shape[0]
    if show_endpoints:
        pc = pv.PolyData(
            np.concatenate([src, dst], 0), deep=True)
        pc.point_data['labels'] = np.array(
            [i >= count for i in range(2 * count)])
        plotter.add_mesh(pc, show_scalar_bar=False)

    lines = []
    for i in range(count):
        if weight is None or weight[i] <= threshold:
            lines.append(2)
            lines.append(i)
            lines.append(i + count)
    if len(lines) == 0:
        return
    mesh = pv.PolyData(
        np.concatenate([src, dst], 0), lines=lines, deep=True)
    if weight is not None:
        clipped_weight = np.clip(weight, 0, weight_clip)
        mesh.point_data['weights'] = np.concatenate([clipped_weight, clipped_weight], 0)
    plotter.add_mesh(mesh, line_width=line_width, cmap=cmap, show_scalar_bar=show_scalar_bar)


def render_point_clouds(plotter, pc, color='white', point_size=5.0,
                        weight=None, threshold=1e10):
    if weight is not None:
        # Thresholding.
        inds = weight < threshold
        if not inds.any():
            return
        pc = pc[inds, :]
        weight = weight[inds]
    pd = pv.PolyData(pc, deep=True)
    if weight is not None:
        pd.point_data['weight'] = weight
        plotter.add_mesh(pd, cmap='magma', point_size=point_size)
    else:
        plotter.add_mesh(pd, color=color, point_size=point_size)


def render_multi_arrows(plotter, src, dst, weight,
                        vrange=slice(None),
                        arrow_per_point=1,
                        cmap='magma',
                        threshold=1e100):
    # src: Bax3, dst: BaxBpx3, weight: BaxBp
    assert(src.shape[0] == dst.shape[0])
    arrow_per_point = min(arrow_per_point, dst.shape[1])
    src = src[vrange, :]
    dst = dst[vrange, :arrow_per_point, :]
    weight = weight[vrange, :arrow_per_point]  # BaxN
    nv = src.shape[0]

    src_dup = np.repeat(np.expand_dims(src, 0), arrow_per_point, axis=0).reshape(-1, 3)  # NBax3
    dst_dup = np.transpose(dst, axes=[1, 0, 2]).reshape(-1, 3)  # NBax3
    weight_dup = weight.transpose([1, 0]).reshape(-1)  # NBa
    pc = pv.PolyData(np.concatenate([src_dup, dst_dup], 0), deep=True)
    pc.point_data['labels'] = np.array(
        [i >= nv * arrow_per_point for i in range(pc.points.shape[0])])
    plotter.add_mesh(pc)

    lines = []
    for k in range(arrow_per_point):  # Bp
        for i in range(nv):  # Ba
            w = weight_dup[k * nv + i]
            if w <= threshold:
                lines.append(2)
                lines.append(k * nv + i)
                lines.append((k + arrow_per_point) * nv + i)
    mesh = pv.PolyData(
        np.concatenate([src_dup, dst_dup], 0), lines=lines, deep=True)
    if weight is not None:
        mesh.point_data['weights'] = np.concatenate([weight_dup, weight_dup], 0)
    plotter.add_mesh(mesh, cmap=cmap)


def render_impl_fn(plotter,
                   grid,
                   vals,
                   threshold=0.0,
                   color='tan',
                   add_slice=False,
                   isosurf_opacity=1.0):
    cur_pv_grid = pv.UniformGrid(
        grid['dims'], grid['spacing'], grid['origin'])
    cur_pv_grid.point_data['impl'] = (
        vals.transpose([2, 1, 0])  # pyvista use different convention
        .reshape([-1]))

    # Add isosurface.
    if threshold is not None:
        ctr = cur_pv_grid.contour([threshold], scalars='impl')
        if ctr.n_cells > 0 and ctr.n_points > 0:
            plotter.add_mesh(ctr, color=color, opacity=isosurf_opacity)
    if add_slice:
        # Add slicing.
        sliced = cur_pv_grid.slice_orthogonal()
        cmap = plt.cm.get_cmap("viridis", 10)
        plotter.add_mesh(sliced, cmap=cmap)


def render_tri_mesh(plotter,
                    vertices,
                    faces,
                    vertex_weights=None,
                    vertex_weights_style='original',
                    vertex_weights_exp_factor=1.0,
                    color='white',
                    opacity=1.0,
                    wireframe=False):
    mesh_pd = pv.PolyData(vertices, decorate_tri_faces(faces))
    if vertex_weights is not None:
        if vertex_weights_style == 'original':
            weights = vertex_weights
        elif vertex_weights_style == 'exp_neg':
            weights = np.exp(-vertex_weights_exp_factor * vertex_weights)
        else:
            raise Exception('Unknown vertex weight style: {}'.format(vertex_weights_style))
        mesh_pd.point_data['weight'] = weights

    plotter.add_mesh(mesh_pd, color=color if vertex_weights is None else None, opacity=opacity)

    if wireframe:
        edges = mesh_pd.extract_all_edges()
        plotter.add_mesh(edges)


def render_by_type(plotter, obj, options):
    tp = obj['type']
    if tp == 'impl_fn':
        render_impl_fn(plotter, obj['grid'], obj['vals'],
                       **options)
    elif tp == 'tri_mesh':
        render_tri_mesh(plotter, obj['vertices'], obj['faces'],
                        vertex_weights=obj.get('vertex_weights', None),
                        **options)
    elif tp == 'plane':
        render_plane(plotter, obj['center'], obj['normal'])
    elif tp == 'planes':
        idx = options['idx']
        render_plane(plotter, obj['center'][idx, :], obj['normal'][idx, :])
    elif tp == 'arrows':
        render_arrows(plotter, obj['src'], obj['dst'], obj.get('weight', None), **options)
    elif tp == 'ndarray_dict':
        render_ndarray_dict(plotter, obj, **options)
    else:
        raise Exception('Unknown annotation type {}'.format(tp))


class SceneRenderer:
    def __init__(self,
                 img_root_dir,
                 render_desc_list,
                 camera_spec={},
                 use_vfb=False,
                 img_ext='.png',
                 step_fmt='step-{:05}'):
        # 'render_desc_list' must be a list of tuple
        # (img_name, scene_id, render_fn).
        # self.camera = camera
        self.img_root_dir = img_root_dir
        self.render_desc_list = render_desc_list
        self.camera_spec = camera_spec
        self.use_vfb = use_vfb
        self.img_ext = img_ext
        self.step_fmt = step_fmt

    def create_camera_spec(self):
        # Since pv.Camera is not pickable, we need to recreate it in subprocesses.
        camera = pv.Camera()
        if 'focal_point' in self.camera_spec:
            camera.focal_point = self.camera_spec['focal_point']
        if 'position' in self.camera_spec:
            camera.position = self.camera_spec['position']
        if 'azimuth' in self.camera_spec:
            camera.azimuth = self.camera_spec['azimuth']
        return camera

    def create_pv_plotter(self,
                          scene,
                          render_fn):
        '''
        render_fn takes (plotter, scene) and modifies plotter in place.
        This is used both for rendering screenshots (during training) or
        interactive jupyter notebook.
        '''
        plotter = pv.Plotter(off_screen=True)
        plotter.camera = self.create_camera_spec()

        render_fn(plotter, scene)
        return plotter

    def generate_image(self, scene, render_fn):
        plotter = self.create_pv_plotter(scene, render_fn)
        image = plotter.screenshot(None, return_img=True)
        return image

    def process_image(self, D):
        if self.use_vfb:
            pv.start_xvfb()
        img = self.generate_image(D['scene'], D['render_fn'])
        p = Path(D['parent_dir']) / Path(D['img_name'])
        D['lock'].acquire()
        p.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(p) + self.img_ext, img)
        D['lock'].release()

    def render(self, scenes, parent_dir=None, show_progress=True):
        '''
        We create child processes here due to a bug in VTK where the
        plotter crashes after rendering 1000+ images.
        '''

        if parent_dir is None:
            parent_dir = self.img_root_dir

        lock = mp.Lock()

        list_count = len(self.render_desc_list)

        args_list = []
        for i in range(list_count):
            img_name, scene_id, render_fn = self.render_desc_list[i]
            args_list.append({'scene': scenes[scene_id],
                              'render_fn': render_fn,
                              'parent_dir': parent_dir,
                              'img_name': img_name,
                              'lock': lock})

        concurrent_size = 1  # somehow larger sizes lead to the same VTK bug
        loop_range = tqdm.trange(0, list_count, concurrent_size)
        for count in loop_range:
            cur_size = min(concurrent_size, list_count - count)
            procs = [mp.Process(target=self.process_image, args=(args_list[count+i], ))
                     for i in range(cur_size)]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
            loop_range.set_description('Rendering ...')

    def act(self, var_dict):
        if not ('scenes' in var_dict):
            return
        step = var_dict['global_step']
        scenes = var_dict['scenes']

        self.render(scenes,
                    parent_dir=Path(self.img_root_dir) / Path(self.step_fmt.format(step)),
                    show_progress=True)
