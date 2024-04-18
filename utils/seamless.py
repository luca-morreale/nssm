
import numpy as np
import torch

from .algebra import rotation_2D


def sphere_tiling(points_2D, return_index=None):

    theta = 90 * np.pi / 180
    R     = rotation_2D(theta).to(points_2D.device)
    a     = torch.tensor([-1,  1]).reshape(1, 2).float().to(points_2D.device)
    c     = torch.tensor([ 1, -1]).reshape(1, 2).float().to(points_2D.device)

    def tile(points, center, Rs):
        tiles = []
        for R in Rs:
            tiles.append((points - center).matmul(R) + center)
        return tiles

    # rotate around a
    tiles  = tile(points_2D, a, [ R.matmul(R), R, R.t() ])
    tiles += [points_2D]
    # rotate around c
    tiles += tile(points_2D, c, [ R.t(), R, R.matmul(R) ])

    if return_index:
        return tiles[return_index]

    return torch.cat(tiles, dim=0)


def compute_sphere_transformation(points_2D):

    theta = 90 * np.pi / 180
    R_90  = rotation_2D(theta).to(points_2D.device)
    a     = torch.tensor([-1,  1]).reshape(1, 2).float().to(points_2D.device)
    c     = torch.tensor([ 1, -1]).reshape(1, 2).float().to(points_2D.device)

    ## build rotation and translation for seamless
    top_mask    = points_2D[:, 1] >  1.0
    left_mask   = points_2D[:, 0] < -1.0
    right_mask  = points_2D[:, 0] >  1.0
    bottom_mask = points_2D[:, 1] < -1.0

    top_tile         = top_mask    & ~left_mask & ~right_mask
    topleft_tile     = top_mask    & left_mask  & ~right_mask
    left_tile        = left_mask   & ~top_mask  & ~bottom_mask
    bottom_tile      = bottom_mask & ~left_mask & ~right_mask
    bottomright_tile = bottom_mask & right_mask & ~left_mask
    right_tile       = right_mask  & ~top_mask  & ~bottom_mask

    translations = torch.zeros_like(points_2D)
    rotations    = torch.eye(2, device=points_2D.device).unsqueeze(0).repeat(points_2D.size(0), 1, 1)

    translations[top_tile | left_tile | topleft_tile] = a
    translations[bottom_tile | bottomright_tile | right_tile] = c

    rotations[top_tile]         = R_90.t()
    rotations[left_tile]        = R_90
    rotations[topleft_tile]     = R_90.matmul(R_90)
    rotations[right_tile]       = R_90
    rotations[bottom_tile]      = R_90.t()
    rotations[bottomright_tile] = R_90.matmul(R_90)

    return translations, rotations
