
import torch

from differential import DifferentialModule
from runners import TrainRunner


class SurfaceTrainRunner(TrainRunner, DifferentialModule):
    ## trainer for NSM

    def forward_model(self, batch, model, experiment):
        param  = batch['param']

        if experiment['loss'].reg_normals > 0.0:
            param.requires_grad_(True)

        points = model(param)

        return points

    def compute_losses(self, model, experiment, model_out, batch):
        loss_obj = experiment['loss']

        gt     = batch['gt']
        points = model_out
        loss, logs = loss_obj(points, gt)
        logs['loss_distance'] = loss.detach()

        # normal regularization
        loss_normals = self.compute_normal_term(loss_obj, points, batch['param'], batch)
        # boundary matching points
        loss_boundary = self.compute_boundary_term(loss_obj, points, batch)

        loss = loss + \
                loss_obj.reg_normals * loss_normals + \
                loss_obj.reg_boundary * loss_boundary
        logs['loss_norm'] = loss_normals.detach()
        logs['loss_boundary'] = loss_boundary.detach()
        logs['loss'] = loss.detach()

        return loss, logs

    def compute_normal_term(self, loss_obj, points, param, batch):
        loss_normals = torch.zeros(1, device=points.device)
        if loss_obj.reg_normals > 0.0:
            gt_normals = batch['normals'].view(-1, 3)
            mask       = batch['mask_normals'].view(-1)
            mask_gt    = mask[:gt_normals.shape[0]]

            pred_normals    = self.compute_normals(out=points, wrt=param).view(-1, 3)
            loss_normals, _ = loss_obj(pred_normals[mask], gt_normals[mask_gt])

        return loss_normals

    def compute_boundary_term(self, loss_obj, points, batch):
        mask_boundary   = batch['mask_boundary']
        boundary_points = points[mask_boundary] # Nx3
        N = boundary_points.size(0) // 2

        loss_boundary, _ = loss_obj(boundary_points[:N], boundary_points[N:])

        return loss_boundary
