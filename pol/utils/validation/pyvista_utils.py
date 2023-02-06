import pyvista
import numpy as np


def expand_faces(faces):
    return np.hstack([np.expand_dims(np.full([faces.shape[0]], 3), 1), faces])
