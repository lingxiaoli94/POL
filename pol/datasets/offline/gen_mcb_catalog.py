from pathlib import Path
import argparse
import meshio
import random
import numpy as np


def search(folder, num_triangle_lim, num_per_category, print_stats=False):
    result = []
    is_leaf = True
    for child in folder.iterdir():
        if child.is_dir():
            is_leaf = False

    if is_leaf:
        for f in folder.iterdir():
            if f.suffix == '.obj':
                try:
                    mesh = meshio.read(f)
                    num_triangle = mesh.cells_dict['triangle'].shape[0]
                    if (num_triangle_lim == -1 or
                            num_triangle_lim >= num_triangle):
                        result.append(f)
                except Exception as e:
                    print("Exception thrown when processing {}!".format(f))
        if num_per_category != -1 and len(result) > num_per_category:
            random.shuffle(result)
            result = result[:num_per_category]

        print('Extracted {} objs from category {}'.format(
            len(result), folder))
        if print_stats:
            triangle_counts = []
            for f in result:
                mesh = meshio.read(f)
                triangle_counts.append(mesh.cells_dict['triangle'].shape[0])
            triangle_counts = np.array(triangle_counts)
            print('max #T: {}, avg #T: {}'.format(
                triangle_counts.max(), triangle_counts.mean()))

    else:
        for child in folder.iterdir():
            if child.is_dir():
                child_result = search(child, num_triangle_lim, num_per_category)
                result.extend(child_result)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str)
    parser.add_argument('output_txt', type=str)
    parser.add_argument('--triangle_lim', type=int, default=-1)
    parser.add_argument('--category_lim', type=int, default=-1)
    args = parser.parse_args()

    result = search(Path(args.root_dir), args.triangle_lim, args.category_lim)
    ofs = open(args.output_txt, 'w')
    for f in result:
        ofs.write('{}\n'.format(f))
