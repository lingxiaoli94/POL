import pyvista
import numpy as np


def decorate_tri_faces(faces):
    return np.hstack([np.expand_dims(np.full([faces.shape[0]], 3), 1), faces])


def unpack_tri_faces(faces):
    assert(faces.ndim == 1 and faces.shape[0] % 4 == 0)
    new_faces = []
    for i in range(faces.shape[0] // 4):
        assert(faces[4*i] == 3)
        for k in range(1, 4):
            new_faces.append(faces[4*i+k])
    return np.array(new_faces).reshape(-1, 3)
