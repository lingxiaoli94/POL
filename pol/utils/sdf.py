import torch
import geomlib


class SignedDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, proj_info,
                unsigned=True,
                sgn_origin=None,
                vol_vertices=None, vol_tets=None):
        dist, face_idx, w1, w2 = geomlib.triangle_projection_3d(points, proj_info)
        if unsigned:
            sgn = torch.ones_like(dist)
        else:
            if vol_vertices is not None and vol_tets is not None:
                sgn = geomlib.point_tet_mesh_test(points, vol_vertices, vol_tets)
            else:
                assert(sgn_origin is not None)
                sgn = geomlib.point_tri_mesh_test(points,
                                                  proj_info.vertices,
                                                  proj_info.faces,
                                                  sgn_origin)
        ctx.proj_info = proj_info
        ctx.save_for_backward(points, face_idx, w1, w2, sgn)
        return dist * sgn.to(dist), face_idx

    @staticmethod
    def backward(ctx, dist_grad_output, idx_grad_output):
        points, face_idx, w1, w2, sgn = ctx.saved_tensors
        proj_points = ctx.proj_info.bary_points(face_idx, w1, w2)
        d = sgn.unsqueeze(1) * (points - proj_points)
        d /= torch.linalg.norm(d, dim=-1, keepdim=True) + 1e-8  # Bx3
        return d * dist_grad_output.unsqueeze(1), None, None, None, None, None
