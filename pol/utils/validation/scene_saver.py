import h5py
import numpy as np
from pathlib import Path
import shutil
import re
def count_h5_keys(gp, prefix):
    count = 0
    while True:
        key = '{}_{}'.format(prefix, count)
        if key not in gp:
            break
        count += 1
    return count


def save_ndarray_dict(obj, parent_gp):
    for k in obj:
        parent_gp.create_dataset(k, data=obj[k])


def load_ndarray_dict(parent_gp):
    obj = {}
    for k in parent_gp:
        if k != 'type':
            obj[k] = parent_gp[k][:]
    return obj


def save_impl_fn(obj, parent_gp):
    grid = obj['grid']
    grid_gp = parent_gp.create_group('grid')
    grid_gp.create_dataset('dims', data=grid['dims'])
    grid_gp.create_dataset('spacing', data=grid['spacing'])
    grid_gp.create_dataset('origin', data=grid['origin'])
    parent_gp.create_dataset('vals', data=obj['vals'])


def load_impl_fn(parent_gp):
    obj = {}
    grid_gp = parent_gp['grid']
    obj['grid'] = {
        'dims': grid_gp['dims'][:],
        'spacing': grid_gp['spacing'][:],
        'origin': grid_gp['origin'][:]
    }
    obj['vals'] = parent_gp['vals'][:]
    return obj


def save_tri_mesh(obj, parent_gp):
    parent_gp.create_dataset('vertices', data=obj['vertices'])
    parent_gp.create_dataset('faces', data=obj['faces'])
    if 'vertex_weights' in obj:
        parent_gp.create_dataset('vertex_weights', data=obj['vertex_weights'])


def load_tri_mesh(parent_gp):
    obj = {}
    obj['vertices'] = parent_gp['vertices'][:]
    obj['faces'] = parent_gp['faces'][:]
    if 'vertex_weights' in parent_gp:
        obj['vertex_weights'] = parent_gp['vertex_weights'][:]
    return obj


def save_planes(obj, parent_gp):
    parent_gp.create_dataset('center', data=obj['center'])
    parent_gp.create_dataset('normal', data=obj['normal'])


def load_planes(parent_gp):
    obj = {}
    obj['center'] = parent_gp['center'][:]
    obj['normal'] = parent_gp['normal'][:]
    return obj


def save_arrows(obj, parent_gp):
    parent_gp.create_dataset('src', data=obj['src'])
    parent_gp.create_dataset('dst', data=obj['dst'])
    if 'weight' in obj:
        parent_gp.create_dataset('weight', data=obj['weight'])


def load_arrows(parent_gp):
    obj = {}
    obj['src'] = parent_gp['src'][:]
    obj['dst'] = parent_gp['dst'][:]
    if 'weight' in parent_gp:
        obj['weight'] = parent_gp['weight'][:]
    return obj


def save_multi_arrows(obj, parent_gp):
    parent_gp.create_dataset('src', data=obj['src'])
    parent_gp.create_dataset('dst', data=obj['dst'])
    if 'weight' in obj:
        parent_gp.create_dataset('weight', data=obj['weight'])


def load_multi_arrows(parent_gp):
    obj = {}
    obj['src'] = parent_gp['src'][:]
    obj['dst'] = parent_gp['dst'][:]
    if 'weight' in parent_gp:
        obj['weight'] = parent_gp['weight'][:]
    return obj


def save_scene_obj(obj, parent_gp):
    obj_type = obj['type']
    parent_gp.attrs['type'] = obj_type
    if obj_type == 'ndarray_dict':
        save_ndarray_dict(obj, parent_gp)
    elif obj_type == 'impl_fn':
        save_impl_fn(obj, parent_gp)
    elif obj_type == 'plane':
        save_planes(obj, parent_gp)
    elif obj_type == 'planes':
        save_planes(obj, parent_gp)
    elif obj_type == 'arrows':
        save_arrows(obj, parent_gp)
    elif obj_type == 'multi_arrows':
        save_multi_arrows(obj, parent_gp)
    elif obj_type == 'tri_mesh':
        save_tri_mesh(obj, parent_gp)
    else:
        raise Exception('Unknown scene object type {}'.format(obj_type))


def load_scene_obj(parent_gp):
    obj_type = parent_gp.attrs['type']
    if obj_type == 'ndarray_dict':
        obj = load_ndarray_dict(parent_gp)
    elif obj_type == 'impl_fn':
        obj = load_impl_fn(parent_gp)
    elif obj_type == 'plane':
        obj = load_planes(parent_gp)
    elif obj_type == 'planes':
        obj = load_planes(parent_gp)
    elif obj_type == 'arrows':
        obj = load_arrows(parent_gp)
    elif obj_type == 'multi_arrows':
        obj = load_multi_arrows(parent_gp)
    elif obj_type == 'tri_mesh':
        obj = load_tri_mesh(parent_gp)
    else:
        raise Exception('Unknown scene object type {}'.format(obj_type))
    obj['type'] = obj_type
    return obj


def save_scene(scene, parent_gp):
    for key, obj in scene.items():
        gp = parent_gp.create_group(key)
        save_scene_obj(obj, gp)


def load_scene(parent_gp):
    scene = {}
    for key in parent_gp.keys():
        scene[key] = load_scene_obj(parent_gp[key])
    return scene


def save_scenes(scenes, h5_path):
    handle = h5py.File(h5_path, 'w')
    num_scenes = len(scenes)
    for i in range(num_scenes):
        gp = handle.create_group('scene_{}'.format(i))
        save_scene(scenes[i], gp)


def count_h5_keys(gp, prefix):
    count = 0
    while True:
        key = '{}_{}'.format(prefix, count)
        if key not in gp:
            break
        count += 1
    return count

def find_max_h5_key(gp, prefix, return_itr=False):
    max_itr = -1
    max_key = None
    for key in gp:
        m = re.match('{}_([0-9]+)'.format(prefix), key)
        if m is not None:
            cur_itr = int(m.group(1))
            if cur_itr > max_itr:
                max_itr = cur_itr
                max_key = key
    if return_itr:
        return max_itr
    return max_key


def load_scenes(h5_path, scene_ids=None, unroll=True):
    handle = h5py.File(h5_path, 'r')
    scenes = []
    if scene_ids is not None:
        for i in scene_ids:
            key = 'scene_{}'.format(i)
            scene = load_scene(handle[key]) if unroll else handle[key]
            scenes.append(scene)
    else:
        count = 0
        while True:
            key = 'scene_{}'.format(count)
            if key not in handle:
                break
            scene = load_scene(handle[key]) if unroll else handle[key]
            scenes.append(scene)
            count += 1
    return scenes


class SceneSaver:
    def __init__(self, folder, fmt='step-{:05}', rm_exists=True):
        self.dir = Path(folder)
        if self.dir.exists() and rm_exists:
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt

    def act(self, var_dict):
        step = var_dict['global_step']
        scenes = var_dict['scenes']
        save_scenes(scenes, self.dir / (self.fmt.format(step) + '.h5'))
