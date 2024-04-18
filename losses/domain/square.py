
import torch
from torch.nn import Module

from utils import square_domain_mask


class SquareDomain(Module):

    ## Is there a way to avoid this? i.e. not calling this
    def __init__(self, **kwargs):
        super().__init__()

    def domain_mask(self, points2D, tris_points2D):
        ## unit square domain [-;-1] [1;1]
        return square_domain_mask(points2D)


    def boundary_distances(self, points2D_target, points2D_source):

        left, right, top, bottom = self.boundary_mask(points2D_source)

        boundary_distances = (points2D_target[left,   0] + 1.0).abs() + \
                             (points2D_target[right,  0] - 1.0).abs() + \
                             (points2D_target[top,    1] - 1.0).abs() + \
                             (points2D_target[bottom, 1] + 1.0).abs()

        return boundary_distances


    def boundary_mask(self, points2D_source):

        with torch.no_grad():
            left   = (points2D_source[..., 0] == -1.0).squeeze()
            right  = (points2D_source[..., 0] ==  1.0).squeeze()
            top    = (points2D_source[..., 1] ==  1.0).squeeze()
            bottom = (points2D_source[..., 1] == -1.0).squeeze()

        return left, right, top, bottom

    def distance_to_boundary(self, param):
        sd = torch.min((param.abs() - 1.0).abs(), dim=1)[0] # distance to closest boundary
        return sd
