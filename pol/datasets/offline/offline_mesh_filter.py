import argparse
import torch
from pathlib import Path
import shutil
import csv
import h5py

from pol.utils.tri_mesh import TriMesh

def should_keep(args, mesh_file):
    mesh = TriMesh.from_file(mesh_file)
    result = True
    result = result and mesh.vertices.shape[0] <= args.max_num_vertex
    result = result and mesh.faces.shape[0] <= args.max_num_faces
    return result

def mirror_descent(args, catalog_csv_writer,
                   in_cur_dir,
                   suffices=['.obj'],
                   whitelist=None):

    for in_child in in_cur_dir.iterdir():
        if in_child.is_dir():
            mirror_descent(args, catalog_csv_writer, in_child,
                           suffices=suffices,
                           whitelist=whitelist)
        elif in_child.suffix in suffices:
            if (whitelist is not None) and (not str(in_child) in whitelist):
                continue
            if should_keep(args, in_child):
                catalog_csv_writer.writerow([in_child])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_root_dir', type=str)
    parser.add_argument('catalog_csv', type=str)
    parser.add_argument('--whitelist', type=str)
    parser.add_argument('--max_num_vertex', type=int, default=5000)
    parser.add_argument('--max_num_faces', type=int, default=10000)
    parser.add_argument('--suffix', type=str, nargs='+', default=['.obj'])
    args = parser.parse_args()

    in_root_dir = Path(args.in_root_dir)

    csv_file = open(args.catalog_csv, 'w', newline='')
    catalog_csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|')
    whitelist = None
    if args.whitelist:
        whitelist = [f.strip() for f in open(args.whitelist, 'r').readlines()]
    mirror_descent(args, catalog_csv_writer, in_root_dir, whitelist=whitelist)
