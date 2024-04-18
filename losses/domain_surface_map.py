
import torch

from .domain import UnitCircleDomain, SquareDomain
from .landmarks import L2Loss, L1Loss, LhalfLoss, WL1Loss
from .mixin import Loss
from .surface_map import IsometricSurfaceMapLoss, ARAPSurfaceMapLoss
from .surface_map import ConformalSurfaceMapLoss, EquiarealSurfaceMapLoss


class DomainSurfaceMapLoss(Loss):

    reg_boundary    = 0.0
    reg_domain      = 0.0
    reg_landmarks2D = 0.0
    reg_landmarks3D = 0.0
    reg_rotation    = 0.0
    reg_harmonic    = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        surf_map_name = kwargs['surf_map']
        domain_name   = kwargs['domain']
        lands_name    = kwargs['landmarks']

        self.surf_map_loss  = globals()[surf_map_name](**kwargs)
        self.domain         = globals()[domain_name](**kwargs)
        self.landmarks_loss = globals()[lands_name](**kwargs)

        self.surf_map_loss.distance_to_boundary = self.domain.distance_to_boundary


    def forward(self, target_points3D, target_points2D, source_points2D, source_points3D, boundary_mask, lands_mask, target_landmarks):

        domain_mask = self.domain.domain_mask(target_points2D, None)

        losses = self.forward_losses(target_points3D, target_points2D, source_points2D, source_points3D, boundary_mask, lands_mask, target_landmarks, domain_mask)

        distortion_errors, point_boundary_dist, landmarks_dist, domain_error = losses

        # per_point_Jh = distortion_errors[-1]

        ## aggregate
        loss, logs = self.aggregate_losses(domain_mask, distortion_errors, point_boundary_dist, landmarks_dist, domain_error)

        # geo_grad = self.surf_map_loss.geometric_gradient(loss, per_point_Jh, domain_mask)

        # logs['geometric_grad_avg'] = geo_grad.mean().detach()
        # logs['geometric_grad_med'] = geo_grad.median().detach()

        return loss, logs

    def aggregate_losses(self, domain_mask, distortion_errors, point_boundary_dist, landmarks_dist, domain_error):
        ## aggregate
        loss_boundary  = point_boundary_dist.mean()
        loss_landmarks = landmarks_dist.mean()
        loss_domain    = domain_error.mean()

        loss_distortion, logs = self.surf_map_loss.aggregate_distortion_terms(*distortion_errors[:3], domain_mask=domain_mask)

        loss = loss_distortion + \
            self.reg_boundary * loss_boundary + \
            self.reg_domain * loss_domain + \
            self.reg_landmarks2D * (loss_landmarks if loss_landmarks > 0 else 0.0)


        logs['loss_boundary']   = loss_boundary.detach()
        logs['loss_landmarks']  = loss_landmarks.detach()
        logs['loss_domain']     = loss_domain.detach()

        return loss, logs


    def forward_losses(self, target_points3D, target_points2D, source_points2D, source_points3D, boundary_mask, lands_mask, target_landmarks, domain_mask):
        distortion_errors   = self.forward_distortion(target_points3D, target_points2D, source_points2D, source_points3D)
        point_boundary_dist = self.forward_boundary(target_points2D, None, boundary_mask)
        domain_error        = self.forward_domain(target_points2D, domain_mask)
        landmarks_dist      = self.forward_landmarks(target_points2D, target_landmarks, lands_mask) # in 2D
        return distortion_errors, point_boundary_dist, landmarks_dist, domain_error


    def forward_distortion(self, target_points3D, target_points2D, source_points2D, source_points3D):
        distortion_errors =  self.surf_map_loss.compute_distortion_losses(target_points3D, target_points2D, source_points2D, source_points3D)

        return distortion_errors[0], distortion_errors[2], distortion_errors[3], distortion_errors[1]


    def forward_boundary(self, source_boundary_target, source_boundary, mask):
        if self.reg_boundary > 0:
            boundary_distances = self.domain.boundary_distances(source_boundary_target[mask], source_boundary).abs()
        else:
            boundary_distances =  torch.zeros(2, device=mask.device)
        return boundary_distances

    def forward_landmarks(self, mapped_lands, target_lands, mask):
        if len(mapped_lands) > 0:
            ## Compute landmark distances
            F = target_lands.size(1)
            lands_distance = self.landmarks_loss(mapped_lands[mask].reshape(-1, F), target_lands.reshape(-1, F))
        else:
            lands_distance = torch.zeros(target_lands.shape[0], device=target_lands.device)
        return lands_distance

    def forward_domain(self, target_points2D, domain_mask):
        if self.reg_domain > 0:
            points_outside = target_points2D[~domain_mask]
            domain_error   = points_outside.pow(2).sum(-1) if points_outside.nelement() != 0 else torch.zeros_like(target_points2D)
        else:
            domain_error   = torch.zeros(2, device=target_points2D.device)
        return domain_error

    def string(self):
        out = self.__class__.__name__
        if self.reg_boundary > 0:
            out += f'_regBound-{self.reg_boundary:.2e}'
        if self.reg_domain > 0:
            out += f'_regDom-{self.reg_domain:.2e}'
        if self.reg_landmarks2D > 0:
            out += f'_regLand-{self.reg_landmarks2D:.2e}'
        if self.reg_rotation > 0:
            out += f'_regR-{self.reg_rotation:.2e}'
        if self.reg_landmarks3D > 0:
            out += f'_regL3D-{self.reg_landmarks3D:.2e}'
        if self.reg_harmonic > 0:
            out += f'_regH-{self.reg_harmonic:.2e}'
        # get regularizer string from surf_map_loss, i.e., remove class name
        out_map = self.surf_map_loss.string().split('_')
        out    += '_' + '_'.join(out_map[1:])
        return out
