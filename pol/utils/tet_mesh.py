import numpy as np
import meshio
from .tri_mesh import TriMesh
from pol.utils.scaling import scale_to_unit_cube

kFacesOfATet = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]])


class TetMesh:
    def __init__(self, V, T, fix_orientation=True, scaling=True):
        # V - Nx3, T - Tx3
        if scaling:
            V = scale_to_unit_cube(V)
        self.vertices = V
        self.tets = T

        if fix_orientation:
            self.fix_orientation()

    def fix_orientation(self):
        for i in range(self.tets.shape[0]):
            t = self.tets[i, :]  # 4
            ps = np.stack([self.vertices[t[k + 1], :] - self.vertices[t[0], :] for k in range(3)],
                          0)  # 3x3
            if np.linalg.det(ps) < 0:
                self.tets[i, 1] = t[2]
                self.tets[i, 2] = t[1]

    def extract_surface(self):
        # returns: (surf_vert_idxs, surf_faces)
        # surf_vert_idxs is a list of idxs on surface
        # surf_faces is Fx3 with new idxs
        # surf_faces_old is Fx3 with old idxs
        face_loop_set = set()
        for i in range(self.tets.shape[0]):
            t = self.tets[i, :]
            for j in range(kFacesOfATet.shape[0]):
                f = kFacesOfATet[j, :]
                face_loop_set.add((t[f[0]], t[f[1]], t[f[2]]))

        surf_faces_old = []
        for i in range(self.tets.shape[0]):
            t = self.tets[i, :]
            for j in range(kFacesOfATet.shape[0]):
                f = kFacesOfATet[j, :]
                fc = (t[f[0]], t[f[1]], t[f[2]])

                on_surface = True
                for p in range(3):
                    fc_rev = (fc[p], fc[(p+2) % 3], fc[(p+1) % 3])
                    if fc_rev in face_loop_set:
                        on_surface = False
                if on_surface:
                    surf_faces_old.append(fc)
        surf_faces_old = np.array(surf_faces_old)

        v_cnt = 0
        v_dict = {}
        surf_vert_idxs = []
        for i in surf_faces_old.reshape(-1):
            if i not in v_dict:
                v_dict[i] = v_cnt
                surf_vert_idxs.append(i)
                v_cnt += 1
        surf_faces = []
        for i in range(surf_faces_old.shape[0]):
            surf_faces.append(np.array(
                [v_dict[surf_faces_old[i, j]] for j in range(3)]
            ))
        return (np.array(surf_vert_idxs),
                np.array(surf_faces),
                np.array(surf_faces_old))

    def extract_surface_mesh(self):
        surf_vert_idxs, surf_faces, _ = self.extract_surface()
        return TriMesh(self.vertices[surf_vert_idxs, :],
                       surf_faces)

    @staticmethod
    def from_file(file):
        mesh = meshio.read(file)
        return TetMesh(mesh.points, mesh.cells_dict['tetra'])
