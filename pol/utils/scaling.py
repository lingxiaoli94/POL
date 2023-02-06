import numpy as np


# This scales things to a unit sphere.
def scale_to_unit_cube(vertices):
    cmin = np.amin(vertices, 0)
    cmax = np.amax(vertices, 0)
    center = (cmin + cmax) / 2
    scale = np.linalg.norm(vertices - center, axis=-1).max()

    vertices = (vertices - center) / scale

    return vertices
