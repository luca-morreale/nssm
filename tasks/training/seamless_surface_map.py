
from .surface_map import SurfaceMapTrainRunner


class SeamlessMapTrainRunner(SurfaceMapTrainRunner):
    ## surface to surface map trainer

    def compute_losses(self, model, experiment, model_out, batch):
        loss, logs = super().compute_losses(model, experiment, model_out, batch)

        loss_obj = experiment['loss']

        points2D_source = model_out[3]

        #### 1. tiling loss - all points should respect the tiling property
        loss_tiling = self.compute_tiling_term(model, points2D_source, batch)
        logs['loss_tiling'] = loss_tiling.detach()

        #### 2. constraints loss force map to respect cones
        loss_cones = self.compute_cone_term(model, batch)
        logs['loss_cones'] = loss_cones.detach()

        loss = loss + \
            loss_obj.reg_tiling * loss_tiling + \
            loss_obj.reg_cones * loss_cones

        logs['loss'] = loss.detach()

        return loss, logs

    def compute_distortion_term(self, loss_obj, model_out, batch):

        points3D_target = model_out[0]
        points2D_target = model_out[1]
        points2D_target_no_rot = model_out[2]
        points2D_source = model_out[3]
        points3D_source = model_out[4]

        mask = ~(batch['boundary_mask'] | batch['landmarks_mask'])

        #### 1. compute all gradients
        J_h = loss_obj.surf_map_loss.gradient(out=points2D_target_no_rot, wrt=points2D_source) # this is different from surface_map
        J_g = loss_obj.surf_map_loss.gradient(out=points3D_source, wrt=points2D_source)
        J_f = loss_obj.surf_map_loss.gradient(out=points3D_target, wrt=points2D_target)

        #### 2. compute loss for inversion
        loss_inversion = loss_obj.surf_map_loss.fold_regularization(J_h).mean()

        #### 3. compose jacobians
        J_fh    = J_f.matmul(J_h)
        J_g_inv = loss_obj.surf_map_loss.invert_J(J_g)
        #J  = J_fh.matmul(J_g_inv)[mask] # boundary and landmarks are excluded here
        J  = J_fh.matmul(J_g_inv)[:1024*4] # boundary and landmarks are excluded here
        #### 4. compute distortion from first fundamental form

        num_points = J.shape[0]
        size_block = num_points // 4
        J_samples  = J[:size_block] # for each point p we have 3 points nearby (tot 4)
        J_neigh    = J[size_block:]
        J_neigh    = J_neigh.unsqueeze(1).reshape(size_block, 3, J_neigh.shape[-2], J_neigh.shape[-1])

        loss_smoothing = (J_samples.unsqueeze(1) - J_neigh).pow(2).sum(-1).sum(-1).mean()

        return loss_inversion, loss_smoothing

    def compute_tiling_term(self, model, points2D_source, batch):

        boundary_mask         = batch['boundary_mask']
        buddies_boundary_mask = batch['boundary_buddy_mask']
        boundary_t            = batch['boundary_translation']
        boundary_R            = batch['boundary_rotation']

        _, mapped_bnd         = model.forward_map(points2D_source[boundary_mask], batch['R']) # get points w/o rotation
        _, mapped_bnd_buddies = model.forward_map(points2D_source[buddies_boundary_mask], batch['R']) # get points w/o rotation

        matching_boundary = (mapped_bnd - boundary_t).unsqueeze(1).bmm(boundary_R).squeeze(1) + boundary_t
        loss_tiling_2D    = (matching_boundary - mapped_bnd_buddies).pow(2).sum(-1)
        loss_tiling_2D    = loss_tiling_2D.mean()

        return loss_tiling_2D

    def compute_cone_term(self, model, batch):
        source_cones = batch['source_cones']
        target_cones = batch['target_cones']
        R            = batch['R']

        _, mapped_cones = model.forward_map(source_cones, R)
        loss_cones = (target_cones - mapped_cones).pow(2).sum(-1).mean()

        return loss_cones

