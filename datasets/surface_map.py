
import numpy as np
import torch
import igl

from utils import print_info
from utils import sample_surface

from .mixin import DatasetMixin


class SurfaceMapDataset(DatasetMixin):

    def __init__(self, config):

        self.config         = config
        self.num_points     = config.get('num_points', 1024)
        self.num_bnd_points = config.get('num_boundary_points', 128)
        self.num_matches    = config.get('num_matches', 15)

        self.read_surfaces(config)
        self.read_matches(config)


    def read_surfaces(self, config):
        self.sample_source = self.read_sample(config['sample_source'])
        self.sample_target = self.read_sample(config['sample_target'])

        ## extract all data
        print_info('Reading source surface')
        self.source_uvs       = self.sample_source['param']
        self.source_faces     = self.sample_source['faces']
        self.source_points    = self.sample_source['points']
        self.source_C         = self.sample_source['C'] # normalization constant
        self.source_boundary  = None
        self.source_name      = self.sample_source['name']

        boundary_idx              = igl.boundary_loop(self.source_faces.numpy())
        self.source_boundary      = self.source_uvs[boundary_idx]

        self.boundary_mask_source = torch.zeros(self.source_uvs.size(0)).bool()
        self.boundary_mask_source[boundary_idx] = True

        print_info('Reading target surface')
        self.target_uvs       = self.sample_target['param']
        self.target_faces     = self.sample_target['faces']
        self.target_points    = self.sample_target['points']
        self.target_C         = self.sample_target['C'] # normalization constant
        self.target_name      = self.sample_target['name']

        boundary_idx              = igl.boundary_loop(self.target_faces.numpy())
        self.target_boundary      = self.target_uvs[boundary_idx]
        self.boundary_mask_target = torch.from_numpy(boundary_idx).bool()


    def read_matches(self, config):
        ### Extract landmarks data
        if 'matches_path' in config:
            print_info('Reading matches from file')
            matches_path          = config['matches_path']
            matches_dict          = torch.load(matches_path)
            self.source_landmarks = matches_dict['source'].long()
            self.target_landmarks = matches_dict['target']
        else:
            print_info('Reading matches from list')
            self.source_landmarks = torch.tensor(config['source_landmarks']).long()
            self.target_landmarks = torch.tensor(config['target_landmarks']).long()

        print_info('Extracting matches')
        self.lands_source = self.source_uvs[self.source_landmarks].float()
        if len(self.target_landmarks.size()) > 1:
            self.lands_target = self.target_landmarks.clone().float()
        else:
            self.target_landmarks = self.target_landmarks.long()
            self.lands_target = self.target_uvs[self.target_landmarks].float()

        self.R, _ = self.compute_lands_rotation(self.lands_source, self.lands_target)


    def remove_duplicated_matches(self, source_landmarks, target_landmarks):
        _, idx = np.unique(source_landmarks, return_index=True)
        source_landmarks = source_landmarks[idx]
        target_landmarks = target_landmarks[idx]
        _, idx = np.unique(target_landmarks, return_index=True)
        source_landmarks = source_landmarks[idx]
        target_landmarks = target_landmarks[idx]
        return source_landmarks, target_landmarks


    def sample_landmarks(self, return_idx=False):

        #### 1. define how many landmarks to sample
        num_landmarks_sample = min(self.lands_source.size(0), self.num_matches)

        #### 2. sample indices
        idx = torch.randperm(self.lands_source.size(0))[:num_landmarks_sample]

        #### 3. get landmarks
        lands_source = self.lands_source[idx].clone()
        lands_target = self.lands_target[idx].clone()

        #### 4. find indices among all set
        source_lands_idx = self.source_landmarks[idx]

        if return_idx:
            return lands_source, lands_target, source_lands_idx
        return lands_source, lands_target


    def __len__(self):
        return 1


    def __getitem__(self, index):

        ## sample 2D parametrization
        params_to_sample = [self.source_uvs]
        _, _, params_all = sample_surface(self.num_points, self.source_points,
                                    self.source_faces, params_to_sample, method='pytorch3d')

        params = params_all[0]

        data_dict = {
            'source_points':    params,
            'R':                self.R,
            't':                -self.t,
            'C_source':         self.source_C,
            'C_target':         self.target_C,
            'target_domain':    None,
            'boundary':         self.source_boundary,
            'landmarks':        self.lands_source,
            'target_landmarks': self.lands_target,
        }

        ## add domain triangulation in case domain is not a disk or a square
        if 'domain_faces' in self.sample_target:
            data_dict['target_domain'] = self.sample_target['domain_vertices'][self.sample_target['domain_faces']]

        return data_dict


    def num_checkpointing_samples(self):
        return 1

    def get_checkpointing_sample(self, index):

        #### 1. sample landmarks
        lands_source, lands_target, lands_idx = self.sample_landmarks(return_idx=True)

        #### 2. create landmark mask
        landmarks_mask = torch.zeros(self.source_uvs.size(0)).bool()
        landmarks_mask[lands_idx] = True

        #### 3. build batch
        data_dict = {}
        data_dict['source_points']    = self.source_uvs
        data_dict['target_points']    = self.target_uvs
        data_dict['source_points_3D'] = self.source_points
        data_dict['target_points_3D'] = self.target_points
        data_dict['source_faces']     = self.source_faces
        data_dict['target_faces']     = self.target_faces
        data_dict['C_source']         = self.source_C
        data_dict['C_target']         = self.target_C
        data_dict['landmarks']        = lands_source
        data_dict['target_landmarks'] = lands_target
        data_dict['target_name']      = self.target_name + '_' + str(index)
        data_dict['source_name']      = self.source_name + '_' + str(index)
        data_dict['boundary']         = self.source_boundary
        data_dict['name']             = self.source_name + '_' + self.target_name + '_' + str(index)
        data_dict['index']            = index
        data_dict['R']                = self.R.t()
        data_dict['landmarks_mask']   = landmarks_mask
        data_dict['boundary_mask']    = self.boundary_mask_source

        ## optional parameters inside the sample for visualization
        if 'visual_v' in self.sample_source:
            data_dict['visual_uv'] = {'xz':self.sample_source['visual_v'][:,[0,2]], # this is okay for cp2
                                      'xy':self.sample_source['visual_v'][:,[0,1]], # this is okay for cp2
                                      'yz':self.sample_source['visual_v'][:,[1,2]] }# this is okay for cp2
        if 'visual_v' in self.sample_target:
            data_dict['visual_uv_target'] = {'xz':self.sample_target['visual_v'][:,[0,2]], # this is okay for cp2
                                      'xy':self.sample_target['visual_v'][:,[0,1]], # this is okay for cp2
                                      'yz':self.sample_target['visual_v'][:,[1,2]] }# this is okay for cp2

        if hasattr(self, 'map'):
            data_dict['map_gt']   = self.map_gt

        if 'domain_faces' in self.sample_target:
            data_dict['target_domain'] = self.sample_target['domain_vertices'][self.sample_target['domain_faces']]

        if 'oversampled_param' in self.sample_source:
            data_dict['oversampled_param'] = self.sample_source['oversampled_param'].float()
            data_dict['oversampled_faces'] = self.sample_source['oversampled_faces'].long()


        return data_dict
