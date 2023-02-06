import argparse
import torch
from pathlib import Path
from pol.datasets.partial_scan import generate_partial_scan
from pol.problems.mesh_sampler import MeshSampler
import shutil
import csv
import h5py


def generate_scan_h5(mesh_file, out_h5_file):
    sampler = MeshSampler(mesh_file=mesh_file, scaling=True, device=torch.device('cuda'))
    scan = generate_partial_scan(sampler, 32, 2048)  # 32x2048
    handle = h5py.File(out_h5_file, 'w')
    handle.create_dataset('vertices', data=sampler.vertices)
    handle.create_dataset('faces', data=sampler.faces)
    handle.create_dataset('scans', data=scan.detach().cpu().numpy())
    handle.close()


def mirror_descent(catalog_csv_writer,
                   in_cur_dir, out_cur_dir,
                   suffices=['.obj'],
                   whitelist=None):
    if not out_cur_dir.exists():
        out_cur_dir.mkdir(parents=True)

    for in_child in in_cur_dir.iterdir():
        if in_child.is_dir():
            mirror_descent(catalog_csv_writer, in_child, out_cur_dir / in_child.stem,
                           suffices=suffices,
                           whitelist=whitelist)
        elif in_child.suffix in suffices:
            if (whitelist is not None) and (not str(in_child) in whitelist):
                continue
            h5_path = out_cur_dir / (in_child.stem + '.h5')
            generate_scan_h5(in_child, h5_path)
            catalog_csv_writer.writerow([h5_path, in_child])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_root_dir', type=str)
    parser.add_argument('out_root_dir', type=str)
    parser.add_argument('catalog_csv', type=str)
    parser.add_argument('--whitelist', type=str)
    args = parser.parse_args()

    in_root_dir = Path(args.in_root_dir)
    out_root_dir = Path(args.out_root_dir)
    if out_root_dir.exists():
        shutil.rmtree(out_root_dir)

    csv_file = open(args.catalog_csv, 'w', newline='')
    catalog_csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='|')
    whitelist = None
    if args.whitelist:
        whitelist = [f.strip() for f in open(args.whitelist, 'r').readlines()]
    mirror_descent(catalog_csv_writer, in_root_dir, out_root_dir,
                   whitelist=whitelist)
