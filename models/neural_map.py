
import torch
from torch.nn import Module

from .mlp import *

class ParametrizationMap(Module):

    def __init__(self, config):
        super().__init__()

        source_surface_struct = config['source_surface'] # structure of the surface map
        map_struct            = config['map'] # structure of the neural map

        self.source_surface = globals()[source_surface_struct['name']](source_surface_struct) # create network for surface map
        self.neural_map     = globals()[map_struct['name']](map_struct) # create network for neural map

        ## load surface models
        if 'path' in map_struct:
            self.neural_map.load_state_dict(torch.load(map_struct['path'], map_location='cpu'))

        self.source_surface.load_state_dict(torch.load(source_surface_struct['path'], map_location='cpu'))

        ## disable grads for surface map
        self.disable_network_gradient(self.source_surface)
        self.init_map_weights()


    def disable_network_gradient(self, network):
        for param in network.parameters():
            param.requires_grad_(False)


    def init_map_weights(self):
        # init with identity by adding point at the end
        with torch.no_grad():
            if hasattr(self.neural_map.mlp[-1], 'weight'):
                self.neural_map.mlp[-1].weight.fill_(0.0)
                self.neural_map.mlp[-1].bias.fill_(0.0)
            else:
                for i in range(len(self.neural_map.mlp)):
                    self.neural_map.mlp[i].residual[2].weight *= 0.01
                    self.neural_map.mlp[i].residual[2].bias *= 0.01


    def forward(self, points2D):
        points3D_source = self.source_surface(points2D) # forward surface map
        mapped_points   = self.neural_map(points2D) # forward neural map

        return mapped_points, points3D_source



class NeuralMap(ParametrizationMap):

    def __init__(self, config):
        super().__init__(config)

        target_surface_struct = config['target_surface'] # get structure of target surface map

        self.target_surface = globals()[target_surface_struct['name']](target_surface_struct) # create network for target surface map

        ## load surface models
        self.target_surface.load_state_dict(torch.load(target_surface_struct['path'], map_location='cpu'))

        ## disable grads
        self.disable_network_gradient(self.target_surface)


    def forward(self, points2D, R, C_source, C_target):
        points3D_source = self.source_surface(points2D) # forward source surface map
        mapped_points   = self.forward_map(points2D, R) # forward neural map
        points3D_target = self.target_surface(mapped_points) # forward target surface map

        return points3D_target * C_target, mapped_points, points3D_source * C_source

    def forward_map(self, points2D, R):

        rot_points    = points2D.matmul(R)
        mapped_points = self.neural_map(rot_points) + rot_points[..., :2]

        return mapped_points
