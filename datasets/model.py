
import torch
from math import ceil

from utils import extract_boundary_mask
from utils import sample_surface

from .mixin import DatasetMixin


class ModelDataset(DatasetMixin):

    def __init__(self, config):

        self.sample_path = config['sample_path']
        self.num_points  = config['num_points']

        self.sample = self.read_sample(self.sample_path)

        # read surface sample
        self.points        = self.sample['points'].float()
        self.param         = self.sample['param'].float()
        self.faces         = self.sample['faces'].long()
        self.normals       = self.sample['normals'].float()
        self.visual_param  = self.sample['oversampled_param'].float()
        self.visual_points = self.sample['oversampled_points'].float()
        self.visual_faces  = self.sample['oversampled_faces'].long()
        self.name          = self.sample['name']

        ## split into batches
        self.num_batches = ceil(self.points.size(0) / self.num_points)
        self.batchs_idx = self.split_to_blocks(self.points.size(0), self.num_batches)

        ## check how mask normals
        self.mask_normals = True if 'mask_normals' not in config else config['mask_normals']
        if self.mask_normals:
            self.mask_type = config['mask_normals_type']

        self.num_boundary_samples = config.get('num_boundary_samples', 128)
        self.boundary_buddies, boundary_mask = self.find_boundary_buddies(self.visual_points, self.visual_faces)
        self.boundary2D = self.visual_param[boundary_mask]
        self.boundary3D = self.visual_points[boundary_mask]

        # boundary does not match in 2D!! It can only match in 3D
        assert (self.boundary3D - self.boundary3D[self.boundary_buddies]).abs().sum() < 1.0e-4, 'Boundaryd does not match'


    def find_boundary_buddies(self, V, F):
        # 1. find points which are the same
        boundary_mask  = extract_boundary_mask(V.numpy(), F.long().numpy())
        src_bnd_points = V[boundary_mask]

        # 2. compute all pairwise distances
        distances = torch.cdist(src_bnd_points.double(), src_bnd_points.double()) # super important to use double!
        tmp_distances = distances.clone()
        tmp_distances[torch.arange(distances.shape[0]), torch.arange(distances.shape[0])] = 1e8
        idx_match = tmp_distances.argmin(dim=0)

        mask_distances = tmp_distances[torch.arange(distances.shape[0]), idx_match] > 1.0e-4

        assert mask_distances.sum() == 2, f'Too many cones - {mask_distances.sum()}'

        idx_match[mask_distances] = torch.arange(distances.shape[0])[mask_distances]
        idx_match = idx_match.reshape(-1)

        return idx_match, boundary_mask


    def sample_boundary_points(self, num_samples):
        # 1. sample indices of boundary points
        boundary_idx = torch.randperm(self.boundary2D.size(0))[:num_samples]

        # 2. get corresponding point
        boundary_buddy_idx = self.boundary_buddies[boundary_idx].reshape(-1)

        # 3. get boundary points
        boundary2D = self.boundary2D[torch.cat([boundary_idx, boundary_buddy_idx], dim=0)].clone()
        boundary3D = self.boundary3D[torch.cat([boundary_idx, boundary_buddy_idx], dim=0)].clone()

        return boundary2D, boundary3D

    def __len__(self):
        return 1

    def __getitem__(self, index):

        # 1. get index of a random batch
        idx = torch.randperm(self.num_batches)[0]
        idx = self.batchs_idx[idx]

        # 2. get informatio for the batch
        points  = self.points[idx]
        params  = self.param[idx]
        normals = self.normals[idx]

        # 3. sample extra points from the surface
        params_to_sample = [self.param.clone()]
        P, n, p = sample_surface(self.num_points - 2*self.num_boundary_samples, self.points,
                                 self.faces, params_to_sample, method='pytorch3d')

        # 4. sample boundary points (to enforce boundary matching)
        boundary_samples2D, boundary_samples3D = self.sample_boundary_points(self.num_boundary_samples)
        # add 3D point on boundary

        # 4. concat sampled points
        params  = torch.cat([params, p[0], boundary_samples2D], dim=0)
        points  = torch.cat([points, P, boundary_samples3D], dim=0)
        normals = torch.cat([normals, n], dim=0)

        # 5. mask normals
        mask_normals = torch.ones(params.size(0)).bool()
        if self.mask_normals:
            if self.mask_type == 'circle':
                mask_normals = (params.pow(2).sum(-1) < 0.99).bool() # mask for normals
            elif self.mask_type == 'square':
                mask_normals = (params.abs() < 0.99).prod(-1).bool() # mask for normals
            # remove boundary points
            mask_normals[-boundary_samples2D.shape[0]:] = False

        # 6. mask boundary
        mask_boundary = torch.zeros(params.size(0)).bool()
        mask_boundary[-boundary_samples2D.shape[0]:] = True

        # 7. build dictionary
        data_dict = {
                'param':   params,
                'gt':      points,
                'normals': normals,
                'mask_normals':  mask_normals,
                'mask_boundary': mask_boundary,
        }

        return data_dict


    def num_checkpointing_samples(self):
        return 1


    def get_checkpointing_sample(self, index):

        data_dict = {}
        data_dict['param'] = self.param
        data_dict['gts']   = self.points
        data_dict['faces'] = self.faces
        data_dict['name']  = self.name

        if self.visual_param is not None:
            data_dict['oversampled_param'] = self.visual_param
            data_dict['oversampled_faces'] = self.visual_faces

        return data_dict
