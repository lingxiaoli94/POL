import meshio
import argparse
from pol.utils.scaling import scale_to_unit_cube
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_file', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('--shrec', default=False, action='store_true')
    args = parser.parse_args()

    mesh = meshio.read(args.mesh_file)
    vertices = mesh.points
    vertices = scale_to_unit_cube(vertices)
    faces = mesh.cells_dict['triangle']

    if args.shrec:
        faces = faces - 1
        print('faces min: {}, max: {}'.format(faces.min(), faces.max()))

    scaled_mesh = meshio.Mesh(
        vertices,
        [
            ('triangle', faces)
        ]
    )
    scaled_mesh.write(args.out_file)
