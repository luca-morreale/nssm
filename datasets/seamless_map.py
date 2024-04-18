
import torch
import numpy as np
import igl

from utils import compute_vertices_normal
from utils import print_info
from utils import print_error
from utils import rotation_2D
from utils import sample_surface_biharmonic

from .surface_map import SurfaceMapDataset


class SeamlessMapDataset(SurfaceMapDataset):

    def read_surfaces(self, config):
        super().read_surfaces(config)

        if not  ('oversampled_boundary_buddies_indices' in self.sample_source and \
                 'oversampled_boundary_buddies_indices' in self.sample_target):
            print_error('Missing boundary buddies information')
            exit(1)

        self.boundary_buddies      = self.sample_source['oversampled_boundary_buddies_indices']
        self.boundary_buddies_mask = self.sample_source['oversampled_boundary_buddies_mask'].squeeze()

        # set self loop as valid to cones otherwise it is problematic later
        cones_mask = self.boundary_buddies_mask.sum(-1) == 0
        self.boundary_buddies_mask[cones_mask, 0] = True

        boundary_idx = igl.boundary_loop(self.sample_source['oversampled_faces'].numpy())
        self.source_boundary = self.sample_source['oversampled_param'][boundary_idx]


        normals = compute_vertices_normal(self.source_points, self.source_faces)
        self.source_normals       = torch.from_numpy(normals)
        self.source_normals_color = torch.from_numpy(normals + 1) / 2

        normals = compute_vertices_normal(self.target_points, self.target_faces)
        self.target_normals       = torch.from_numpy(normals)
        self.target_normals_color = torch.from_numpy(normals + 1) / 2

        self.a    = torch.tensor([-1.0,  1.0]).reshape(1, 2)
        self.c    = torch.tensor([ 1.0, -1.0]).reshape(1, 2)
        self.b1   = torch.tensor([ 1.0,  1.0]).reshape(1, 2)
        self.b2   = torch.tensor([-1.0, -1.0]).reshape(1, 2)
        self.R_90 = rotation_2D(90 * np.pi / 180)

        # # remove cones from boundary points
        self.mask_cone_a  = (self.source_uvs - self.a).pow(2).sum(-1) < 1.0e-4
        self.mask_cone_c  = (self.source_uvs - self.c).pow(2).sum(-1) < 1.0e-4
        self.mask_cone_b1 = (self.source_uvs - self.b1).pow(2).sum(-1) < 1.0e-4
        self.mask_cone_b2 = (self.source_uvs - self.b2).pow(2).sum(-1) < 1.0e-4


    def read_matches(self, config):
        # super().read_matches(config)

        # save conde indices - assumed correspondences
        self.source_cones = self.sample_source['cones_cut']
        self.target_cones = self.sample_target['cones_cut']

        # There is an ambiguity in the order of b
        # thus order first two cones and fix order of the last cone (b) later

        # find rotation based on first two cones
        cones_src = self.source_uvs[self.source_cones[:2]]
        cones_tgt = self.target_uvs[self.target_cones[:2]]

        # basic rotation is 90deg
        theta = 90 * np.pi / 180
        R     = rotation_2D(theta)
        # try all rotations
        errors = []
        for _ in range(4):
            err = (cones_src - cones_tgt).pow(2).sum()
            errors.append(err)
            cones_src = cones_src.matmul(R.t())
        # find the one with lowest error
        opt_rotation = np.argmin(errors)
        self.R = rotation_2D(90 * opt_rotation * np.pi / 180)

        # now fix matches for b such that it has lowest error
        cones_src = self.source_uvs[self.source_cones[2:]]
        cones_tgt = self.target_uvs[self.target_cones[2:]]
        err_0 = (cones_src.matmul(self.R.t()) - cones_tgt).pow(2).sum() # error same order cones
        cones_src = self.source_uvs[self.source_cones[[3,2]]]
        err_1 = (cones_src.matmul(self.R.t()) - cones_tgt).pow(2).sum() # error reversed order cones
        if err_1 < err_0: # flip cones order
            self.source_cones[[2,3]] = self.source_cones[[3,2]]


    def read_matches(self, config):
        ### Extract landmarks data
        print_info('Reading matches from file')
        matches_path = config['matches_path']
        matches_dict = torch.load(matches_path)


        self.lands_source3D = torch.cat(matches_dict['source'], dim=0)
        self.lands_target3D = torch.cat(matches_dict['target'], dim=0)

        self.source_faces_original_idx = torch.arange(self.source_faces.shape[0]) # self.sample_source['F_idx_uncut']
        self.target_faces_original_idx = torch.arange(self.target_faces.shape[0]) # self.sample_target['F_idx_uncut']

        if 'source2D' in matches_dict:
            self.lands_source = torch.cat(matches_dict['source2D'], dim=0).float()
            self.lands_target = torch.cat(matches_dict['target2D'], dim=0).float()
        else:

            lands_source_face_idx = torch.cat(matches_dict['source_faces'], dim=0).cpu()
            lands_target_face_idx = torch.cat(matches_dict['target_faces'], dim=0).cpu()
            lands_source_bary     = torch.cat(matches_dict['source_bary'], dim=0).unsqueeze(2).cpu()
            lands_target_bary     = torch.cat(matches_dict['target_bary'], dim=0).unsqueeze(2).cpu()


            # translate face index
            lands_source_face_idx = self.source_faces_original_idx[lands_source_face_idx]
            lands_target_face_idx = self.target_faces_original_idx[lands_target_face_idx]

            # mask out faces with negative index
            mask_faces_source = ~(lands_source_face_idx < 0) # negative index are removed faces
            mask_faces_target = ~(lands_target_face_idx < 0) # negative index are removed faces
            mask_faces        = mask_faces_source * mask_faces_target # make 1 mask

            lands_source_face_idx = lands_source_face_idx[mask_faces]
            lands_target_face_idx = lands_target_face_idx[mask_faces]

            # mask out barycentric coordinates
            lands_source_bary = lands_source_bary[mask_faces]
            lands_target_bary = lands_target_bary[mask_faces]

            if not torch.all((lands_source_bary.squeeze(-1).sum(dim=1) - 1.0).pow(2) < 1.0e-4):
                mask_error = (lands_source_bary.squeeze(-1).sum(dim=1) - 1.0).pow(2) > 1.0e-4
                print_error(lands_source_bary[mask_error])
            assert torch.all((lands_source_bary.squeeze(-1).sum(dim=1) - 1.0).pow(2) < 1.0e-4)
            assert torch.all((lands_target_bary.squeeze(-1).sum(dim=1) - 1.0).pow(2) < 1.0e-4)

            # mask out 3D points
            self.lands_source3D = self.lands_source3D[mask_faces]
            self.lands_target3D = self.lands_target3D[mask_faces]

            lands_source_face = self.source_faces[lands_source_face_idx]
            lands_target_face = self.target_faces[lands_target_face_idx]
            lands_source_tris = self.source_uvs[lands_source_face]
            lands_target_tris = self.target_uvs[lands_target_face]

            self.lands_source = (lands_source_tris * lands_source_bary.cpu()).sum(1).reshape(-1, 2).float()
            self.lands_target = (lands_target_tris * lands_target_bary.cpu()).sum(1).reshape(-1, 2).float()


            lands_source_tris = self.source_points[lands_source_face]
            lands_target_tris = self.target_points[lands_target_face]

            # this "undo" rotation (3D landmarks from DinoViT has rotation inside, this way it is removed)
            self.lands_source3D = (lands_source_tris * lands_source_bary.cpu()).sum(1).reshape(-1, 3).float()
            self.lands_target3D = (lands_target_tris * lands_target_bary.cpu()).sum(1).reshape(-1, 3).float()


        self.R, _ = self.compute_lands_rotation(self.lands_source, self.lands_target)
        self.t = torch.zeros(2)

        # Fix cones and save them
        # (fix as in fix order)
        # save conde indices - assumed correspondences
        self.source_cones = self.sample_source['cones_cut']
        self.target_cones = self.sample_target['cones_cut']

        # There is an ambiguity in the order of b
        # thus order first two cones and fix order of the last cone (b) later

        # find rotation based on first two cones
        cones_src = self.source_uvs[self.source_cones[:2]]
        cones_tgt = self.target_uvs[self.target_cones[:2]]

        # basic rotation is 90deg
        theta = 90 * np.pi / 180
        R     = rotation_2D(theta)
        # try all rotations
        errors = []
        for _ in range(4):
            err = (cones_src - cones_tgt).pow(2).sum()
            errors.append(err)
            cones_src = cones_src.matmul(R.t())
        # find the one with lowest error
        opt_rotation = np.argmin(errors)
        self.R = rotation_2D(90 * opt_rotation * np.pi / 180)

        # now fix matches for b such that it has lowest error
        cones_src = self.source_uvs[self.source_cones[2:]]
        cones_tgt = self.target_uvs[self.target_cones[2:]]
        err_0 = (cones_src.matmul(self.R.t()) - cones_tgt).pow(2).sum() # error same order cones
        cones_src = self.source_uvs[self.source_cones[[3,2]]]
        err_1 = (cones_src.matmul(self.R.t()) - cones_tgt).pow(2).sum() # error reversed order cones
        if err_1 < err_0: # flip cones order
            self.source_cones[[2,3]] = self.source_cones[[3,2]]



    def sample_surface_domain(self):
        params_to_sample = [self.source_uvs]
        _, _, _, param_all, param_biharmonic = sample_surface_biharmonic(self.num_points,
                                                        self.source_points, self.source_faces, params_to_sample)

        return torch.cat([param_all[0], param_biharmonic[0]], dim=0)


    def build_masks(self, tot_num_points, num_sampled_points, num_boundary_points, num_landmarks):
        #### 1. make masks boundary and landmarks
        boundary_mask = torch.zeros(tot_num_points).bool()
        lands_mask    = torch.zeros(tot_num_points).bool()

        start = num_sampled_points
        end   = start + num_boundary_points
        boundary_mask[start:end]    = True
        lands_mask[end:] = True

        return boundary_mask, lands_mask


    def sample_boundary(self, num_samples):
        #### 1. sample indices of boundary points
        boundary_idx = torch.randperm(self.source_boundary.size(0))[:num_samples]
        boundary     = self.source_boundary[boundary_idx].clone()

        #### 2. generate mask for copies copies around a (top and right)
        top_bnd    = (boundary[:, 1] ==  1.0)
        left_bnd   = (boundary[:, 0] == -1.0) & ~top_bnd
        right_bnd  = (boundary[:, 0] ==  1.0) & ~top_bnd
        bottom_bnd = (boundary[:, 1] == -1.0) & ~(right_bnd | left_bnd) # force disjoint

        assert (top_bnd | right_bnd | left_bnd | bottom_bnd).sum() == boundary.size(0), "some points are not mapped"

        #### 3. generate copies
        a = self.a.clone().reshape(1, 2).repeat((top_bnd | left_bnd).sum(), 1)
        c = self.c.clone().reshape(1, 2).repeat((right_bnd | bottom_bnd).sum(), 1)
        R_top    = self.R_90.t().unsqueeze(0).repeat(top_bnd.sum(), 1, 1)
        R_left   = self.R_90.unsqueeze(0).repeat(left_bnd.sum(), 1, 1)
        R_right  = self.R_90.unsqueeze(0).repeat(right_bnd.sum(), 1, 1)
        R_bottom = self.R_90.t().unsqueeze(0).repeat(bottom_bnd.sum(), 1, 1)

        translations = torch.cat([a, c], dim=0)
        rotations    = torch.cat([R_top, R_left, R_right, R_bottom], dim=0)

        #### 4. stitch back together
        boundary_reorder = torch.cat([boundary[top_bnd], boundary[left_bnd], boundary[right_bnd], boundary[bottom_bnd]], dim=0).clone()
        buddies          = (boundary_reorder - translations).unsqueeze(1).bmm(rotations).squeeze() + translations

        assert (buddies.abs() > 1.0).sum(-1).bool().sum() == 0, f"there are some points outside the domain {(buddies.abs() > 1.0).sum(-1).bool().sum()}"

        boundary = torch.cat([boundary_reorder, buddies.clone()], dim=0)

        return boundary, translations, rotations


    def sample_landmarks(self, return_idx=False):

        #### 1. define how many landmarks to sample
        num_landmarks_sample = min(self.lands_source.size(0), self.num_matches)

        #### 2. sample indices
        idx = torch.randperm(self.lands_source.size(0))[:num_landmarks_sample]

        #### 3. get landmarks
        lands_source = self.lands_source[idx].clone()
        lands_target = self.lands_target[idx].clone()

        if return_idx:
            return lands_source, lands_target, 0

        return lands_source, lands_target


    def __getitem__(self, index):

        #### 1. sample 2D parametrization
        param = self.sample_surface_domain()
        num_sampled_points = param.size(0)

        #### 2. sample boundary points
        boundary, translations, rotations = self.sample_boundary(self.num_bnd_points)

        #### 3. sample landmarks
        lands_source, lands_target = self.sample_landmarks()

        num_lands = lands_source.size(0)

        #### 4. put all together
        param = torch.cat([param, boundary, lands_source], dim=0)
        tot_num_points = param.size(0)

        assert tot_num_points == num_sampled_points + 2*self.num_bnd_points + num_lands

        #### 5. make masks boundary and landmarks
        boundary_mask, lands_mask = self.build_masks(tot_num_points, num_sampled_points, boundary.shape[0], num_lands)

        assert boundary_mask.sum() == 2*self.num_bnd_points, "error boundary mask"
        assert lands_mask.sum() == num_lands, "error landmarks mask"

        #### 6. clone boundary mask for corresponding points
        boundary_buddy_mask  = boundary_mask.clone()
        half_boundary_points = self.num_bnd_points

        #### 7. find indices in the mask which are "active"
        boundary_indices = boundary_mask.nonzero()

        #### 8. set half points as boundary and half as corresponding points
        boundary_buddy_mask[boundary_indices[:half_boundary_points]] = False
        boundary_mask[boundary_indices[half_boundary_points:]]       = False

        #### 9. assemble into a dict
        data_dict = {
            'source_points':        param,
            'C_source':             self.source_C,
            'C_target':             self.target_C,
            'target_landmarks':     lands_target,
            'index':                index,
            'boundary_mask':        boundary_mask,
            'boundary_buddy_mask':  boundary_buddy_mask,
            'landmarks_mask':       lands_mask,
            'R':                    self.R.t(),
            'boundary_translation': translations,
            'boundary_rotation':    rotations,
            'source_cones':         self.source_uvs[self.source_cones],
            'target_cones':         self.target_uvs[self.target_cones],
        }

        return data_dict


    def get_checkpointing_sample(self, index):
        #### 1. get checkpointing data
        data_dict    = super().get_checkpointing_sample(index)
        param        = data_dict['source_points']

        #### 2. put data in the dictionary
        data_dict['target_name']   += '_' + str(index)
        data_dict['source_name']   += '_' + str(index)
        data_dict['name']          += '_' + str(index)
        data_dict['R']              = self.R.t()
        data_dict['source_colors']  = self.source_normals_color
        data_dict['target_colors']  = self.target_normals_color

        landmarks_mask = torch.zeros(param.size(0)).bool()
        landmarks_mask[self.mask_cone_a] = True
        landmarks_mask[self.mask_cone_c] = True
        landmarks_mask[self.mask_cone_b1] = True
        landmarks_mask[self.mask_cone_b2] = True
        data_dict['landmarks_mask'] = landmarks_mask

        #### 5. remove boundary (there is no concept of boundary)
        del data_dict['boundary_mask']

        return data_dict


    def get_matches(self):
        return self.lands_source, self.lands_target

    def remove_matches(self, mask):
        self.lands_source   = self.lands_source[~mask]
        self.lands_target   = self.lands_target[~mask]
        self.lands_source3D = self.lands_source3D[~mask]
        self.lands_target3D = self.lands_target3D[~mask]

