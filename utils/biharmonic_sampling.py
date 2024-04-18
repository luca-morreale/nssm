
import torch
from torch.nn import functional as F


def sample_surface_biharmonic(num_samples, points, faces, params_to_sample):

    tris = points[faces]
    if points.size(1) < 3:
        tris = F.pad(tris, (0, 1))
    vec_cross = torch.cross(tris[:, 0] - tris[:, 2],
                            tris[:, 1] - tris[:, 2], dim=-1)

    weights = vec_cross.squeeze(-1).pow(2).sum(-1) # face area
    faces_idx = torch.multinomial(weights, num_samples, replacement=True)

    bar_coo = rand_barycentric_coords(num_samples) # w0, w1, w2

    bar_biharmonic = add_noise_to_barycentric(bar_coo)

    # build up points from samples
    # P = w0 * A + w1 * B + w2 * C
    P = sample_triangles(tris, faces_idx, bar_coo)
    P_biharmonic = sample_triangles(tris, faces_idx.unsqueeze(-1).repeat(1, 3).reshape(-1), bar_biharmonic)
    if points.size(1) < 3:
        P = P[:, :2]

    normals = F.normalize(vec_cross, dim=-1)[faces_idx]

    # get vertices
    if len(params_to_sample) > 0:

        params_samples = []
        params_biharmonic = []
        for param in params_to_sample:
            p = sample_triangles(param[faces], faces_idx, bar_coo)
            p_biharmonic = sample_triangles(param[faces], faces_idx.unsqueeze(-1).repeat(1, 3).reshape(-1), bar_biharmonic)
            params_samples.append(p)
            params_biharmonic.append(p_biharmonic)

        return P, P_biharmonic, normals, params_samples, params_biharmonic

    return P, P_biharmonic, normals


def add_noise_to_barycentric(barycentric):
    barycentric = torch.stack(barycentric, dim=1)
    # noise = torch.randn([barycentric.shape[0], 3, 3]) * 0.01
    noise = torch.eye(3).reshape(1, 3, 3) * - 0.1

    new_bary = noise + barycentric.reshape(-1, 1, 3)
    # make sure the coords are within the bounds [0-1]
    new_bary[new_bary > 1.0] = 1.0
    new_bary[new_bary < 0.0] = 0.0
    # normalize them (sum up to 1)
    new_bary = new_bary.reshape(-1, 3)
    new_bary = new_bary / new_bary.sum(dim=-1).reshape(-1, 1)

    new_bary = [new_bary[:, 0].reshape(-1, 1),
                new_bary[:, 1].reshape(-1, 1),
                new_bary[:, 2].reshape(-1, 1)]

    return new_bary





def sample_triangles(tris, selected_faces, bar_coo):
    A = tris[selected_faces, 0]
    B = tris[selected_faces, 1]
    C = tris[selected_faces, 2]

    # build up points from samples
    samples = bar_coo[0].to(tris.device) * A + \
                bar_coo[1].to(tris.device) * B + \
                bar_coo[2].to(tris.device) * C

    return samples


def rand_barycentric_coords(num_points):

    uv = torch.rand(num_points, 2)
    u, v = uv[:, 0:1], uv[:, 1:]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    # if torch.isnan(w0).any() or torch.isnan(w1).any() or torch.isnan(w2).any():
    #     return rand_barycentric_coords(num_points)

    return w0, w1, w2


