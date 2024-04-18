
import torch
import numpy as np

from utils import rotation_2D
from utils import compute_sphere_transformation

from .neural_map import NeuralMap

class SphereSeamlessMap(NeuralMap):

    def __init__(self, config):
        super().__init__(config)

        self.register_buffer('a',    torch.tensor([-1.0,  1.0]).reshape(1, 2))
        self.register_buffer('c',    torch.tensor([ 1.0, -1.0]).reshape(1, 2))
        self.register_buffer('R_90', rotation_2D(90 * np.pi / 180).float())


    def forward_map(self, points2D, R):
        mapped_points = super().forward_map(points2D, R)
        mapped_points_no_rot = mapped_points

        ## build rotation and translation for seamless
        with torch.no_grad():
            translations, rotations = compute_sphere_transformation(mapped_points)

        mapped_points = (mapped_points - translations).unsqueeze(1).bmm(rotations).squeeze() + translations

        return mapped_points, mapped_points_no_rot

    def forward(self, points2D, R, C_source, C_target):
        points2D.requires_grad_(True)
        points3D_source = self.source_surface(points2D) # forward source surface map
        mapped_points, mapped_points_no_rot = self.forward_map(points2D, R) # forward neural map
        points3D_target = self.target_surface(mapped_points) # forward target surface map
        return points3D_target * C_target, mapped_points, mapped_points_no_rot, points2D, points3D_source * C_source
