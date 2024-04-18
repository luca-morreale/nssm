
import torch
from runners import TrainRunner


class SurfaceMapTrainRunner(TrainRunner):
    ## surface to surface map trainer

    def forward_model(self, batch, model, experiment):

        points2D_source = batch['source_points']
        R               = batch['R']
        C_target        = batch['C_target']
        C_source        = batch['C_source']

        points2D_source.requires_grad_(True)

        model_out = model(points2D_source, R, C_source, C_target)
        return model_out


    def compute_losses(self, model, experiment, model_out, batch):
        loss = 0.0

        points3D_target = model_out[0]
        points2D_target = model_out[1]


        ## start energy estimation
        loss_obj = experiment['loss']
        logs = {}

        #### 1. compute loss boundary
        loss_boundary = self.compute_boundary_term(loss_obj, points2D_target, batch)
        logs['loss_boundary'] = loss_boundary.detach()

        #### 2. compute loss landmarks 2D
        loss_lands2D = self.compute_landmarks2D_term(loss_obj, points2D_target, batch)
        logs['loss_landmarks_2D'] = loss_lands2D.detach()

        #### 3. compute loss landmarks 3D
        loss_lands3D = self.compute_landmarks3D_term(loss_obj, points3D_target, batch, model)
        logs['loss_landmarks_3D'] = loss_lands3D.detach()

        #### 4. compute loss inversion and distortion
        loss_inversion, loss_distortion = self.compute_distortion_term(loss_obj, model_out, batch)
        logs['loss_inversion']  = loss_inversion.detach()
        logs['loss_distortion'] = loss_distortion.detach()

        #### 7. aggregate all
        loss =  loss_obj.reg_boundary      * loss_boundary + \
                loss_obj.reg_landmarks2D   * loss_lands2D + \
                loss_obj.reg_landmarks3D   * loss_lands3D + \
                loss_obj.reg_distortion    * loss_distortion + \
                loss_obj.reg_folding       * loss_inversion

        logs['loss'] = loss.detach()

        return loss, logs

    def compute_boundary_term(self, loss_obj, points2D_target, batch):

        loss_boundary = torch.zeros(1, device=points2D_target.device)
        if loss_obj.reg_boundary > 0:
            boundary_mask      = batch['boundary_mask']
            boundary_distances = loss_obj.domain.boundary_distances(points2D_target[boundary_mask], None).abs()
            loss_boundary      = boundary_distances.mean()

        return loss_boundary

    def compute_domain_term(self, loss_obj, points2D_target):
        domain_mask = loss_obj.domain.domain_mask(points2D_target, None)
        loss_domain = torch.zeros(1, device=points2D_target.device)
        if loss_obj.reg_domain > 0:
            points_outside = points2D_target[~domain_mask]
            loss_domain    = points_outside.pow(2).sum(-1).mean()

        return loss_domain

    def compute_landmarks2D_term(self, loss_obj, points2D_target, batch):
        landmarks_gt = batch['target_landmarks']
        lands_mask   = batch['landmarks_mask']
        return self.compute_landmarks_term(loss_obj, points2D_target, landmarks_gt, lands_mask, loss_obj.reg_landmarks2D)

    def compute_landmarks3D_term(self, loss_obj, points3D_target, batch, model):
        lands_mask     = batch['landmarks_mask']
        landmarks_gt2D = batch['target_landmarks']
        landmarks_gt   = model.target_surface(landmarks_gt2D) * batch['C_target']
        return self.compute_landmarks_term(loss_obj, points3D_target, landmarks_gt, lands_mask, loss_obj.reg_landmarks3D)

    def compute_landmarks_term(self, loss_obj, points_target_pred, points_target_gt, mask, weight):
        loss_lands = torch.zeros(1, device=points_target_pred.device)
        if weight > 0:
            loss_lands = loss_obj.forward_landmarks(points_target_pred, points_target_gt, mask).mean()
        return loss_lands

    def compute_distortion_term(self, loss_obj, model_out, batch):

        points3D_target = model_out[0]
        points2D_target = model_out[1]
        points2D_source = model_out[2]
        points3D_source = model_out[3]

        domain_mask = loss_obj.domain.domain_mask(points2D_target, None)
        #### 1. compute all gradients
        J_h = loss_obj.surf_map_loss.gradient(out=points2D_target, wrt=points2D_source)
        J_g  = loss_obj.surf_map_loss.gradient(out=points3D_source, wrt=points2D_source)
        J_f  = loss_obj.surf_map_loss.gradient(out=points3D_target, wrt=points2D_target)

        #### 2. compute loss for inversion
        loss_inversion = loss_obj.surf_map_loss.fold_regularization(J_h).mean()

        #### 3. compose jacobians
        J_fh    = J_f.matmul(J_h)
        J_g_inv = loss_obj.surf_map_loss.invert_J(J_g)
        J       = J_fh.matmul(J_g_inv)[domain_mask]
        # First Fundamental Form
        FFF = J.transpose(1,2).matmul(J)
        #### 4. compute distortion from first fundamental form
        loss_distortion = loss_obj.surf_map_loss.map_distortion(FFF).mean()

        return loss_inversion, loss_distortion

