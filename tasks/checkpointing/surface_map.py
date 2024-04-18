
import torch

from runners import CheckpointRunner


class SurfaceMapCheckpointRunner(CheckpointRunner):

    def checkpoint_sample(self, sample, model, experiment, ckpt_info):

        torch.set_grad_enabled(self.run_losses)

        sample['source_points'] = self.move_to_device(sample['source_points'])
        sample['R']             = self.move_to_device(sample['R'])

        model_outputs   = self.forward(sample, model, experiment)
        losses, scalars = self.forward_losses(model_outputs, sample, experiment['loss'])

        model.R = sample['R']

        torch.set_grad_enabled(False)
        self.checkpoint(model_outputs, losses, model, sample, experiment, ckpt_info, scalars)
        model.zero_grad()

    def forward(self, batch, model, experiment):

        points2D_source = batch['source_points']
        R               = batch['R']
        C_target        = batch['C_target']
        C_source        = batch['C_source']

        points2D_source.requires_grad_(True)

        model_out = model(points2D_source, R, C_source, C_target)
        return model_out

    def forward_losses(self, model_outputs, sample, loss_obj):

        if not self.run_losses:
            return [], {}

        points3D_target, points2D_target, source_uvs, points3D_source = model_outputs

        lands_target  = self.move_to_device(sample['target_landmarks'])
        lands_mask    = self.move_to_device(sample['landmarks_mask'])
        boundary_mask = self.move_to_device(sample['boundary_mask'])

        ###### LOSSES
        domain_mask = loss_obj.domain.domain_mask(points2D_target, None)
        losses      = loss_obj.forward_losses(points3D_target, points2D_target, source_uvs, points3D_source, boundary_mask, lands_mask, lands_target, domain_mask)
        per_point_distortion = losses[0][0]
        per_point_folding    = losses[0][-2]
        per_point_Jh         = losses[0][-1]

        ### aggregate lossess
        loss, _ = loss_obj.aggregate_losses(domain_mask, *losses)

        ### compute scalar quantities
        geo_grad = loss_obj.surf_map_loss.geometric_gradient(loss, per_point_Jh, domain_mask=None)

        ### compute distance from GT map if it exists
        scalars = {
                   'distortion': per_point_distortion.detach(), \
                   'folding':    per_point_folding.detach(), \
                   'geo_grad':   geo_grad.detach()
                  }

        return losses, scalars


    def checkpoint(self, model_outputs, losses, model, sample, experiment, ckpt_info, scalars):
        lands_mask          = self.move_to_device(sample['landmarks_mask'])
        points2D_target     = model_outputs[1]
        lands_source_mapped = points2D_target.reshape(-1,2)[lands_mask.reshape(-1)]

        faces_filtered = self.filter_faces(model_outputs, sample, experiment['loss'])

        ###### SAVE surfaces
        self.save_all_surfaces(sample, model_outputs, faces_filtered, scalars, model, experiment, ckpt_info)


        if ckpt_info.generate_report:
            self.report_error_metrics(losses, ckpt_info)



    def save_all_surfaces(self, sample, model_outputs, faces_filtered, scalars, model, experiment, ckpt_info):

        self.save_landmarks3D(model_outputs, faces_filtered, model, sample, ckpt_info)

        self.save_map_as_surface(model_outputs, scalars, faces_filtered, sample, ckpt_info)

        self.save_constant_surfaces(model_outputs, model, sample, ckpt_info)

        if 'oversampled_param' in sample:
            self.save_oversampled_map(model, sample, experiment['loss'], ckpt_info)


    def filter_faces(self, model_outputs, sample, loss_obj):
        source_faces    = sample['source_faces']
        points2D_target = model_outputs[1]

        domain_mask    = loss_obj.domain.domain_mask(points2D_target, None).cpu()
        faces_mask     = domain_mask[source_faces].prod(dim=-1).bool()
        faces_filtered = source_faces[faces_mask]

        return faces_filtered


    def save_landmarks3D(self, model_outputs, faces_filtered, model, sample, ckpt_info):
        lands_mask   = self.move_to_device(sample['landmarks_mask'])
        lands_target = self.move_to_device(sample['target_landmarks'])
        target_C     = sample['C_target']
        source_C     = sample['C_source']

        source_GT_points = sample['source_points_3D'] * source_C
        target_GT_points = sample['target_points_3D'] * target_C
        source_faces     = sample['source_faces']
        target_faces     = sample['target_faces']
        target_name      = sample['target_name']
        source_name      = sample['source_name']

        points3D_target, _, _, points3D_source = model_outputs

        lands3D_source = points3D_source[lands_mask]
        lands3D_mapped = points3D_target[lands_mask]
        lands3D_target = model.target_surface(lands_target) * target_C

        ### landmarks
        self.append_surface_data([source_GT_points, target_GT_points], [source_faces, target_faces],
                                None, 'correspondences', ['GT', source_name, target_name],
                                ckpt_info, scalars={'landmarks':[lands3D_source, lands3D_target]}, constant=True)
        self.append_surface_data([source_GT_points, points3D_target], [source_faces, faces_filtered],
                                None, 'correspondences', [source_name, target_name],
                                ckpt_info, scalars={'landmarks':[lands3D_source, lands3D_mapped]})


    def save_map_as_surface(self, model_outputs, scalars, faces_filtered, sample, ckpt_info):
        target_name  = sample['target_name']

        points3D_target, _, source_uvs, _ = model_outputs

        color_mask             = scalars['folding'] > 0.0
        colors                 = torch.ones_like(points3D_target).cpu() * 0.83
        colors[color_mask, 0]  = 1.0
        colors[color_mask, 1:] = 0.0

        self.append_surface_data(points3D_target, faces_filtered, source_uvs, 'surface', [target_name, 'filtered'], ckpt_info, scalars=scalars, colors=colors)
        prefix_name = self.build_prefix_name([target_name, 'filtered'], None)
        self.save_landmarks_list(points3D_target, faces_filtered, sample['landmarks_mask'], ckpt_info.checkpoint_dir, prefix_name)


    def save_oversampled_map(self, model, sample, loss_obj, ckpt_info):
        source_uvs   = self.move_to_device(sample['oversampled_param'])
        source_faces = sample['oversampled_faces']
        target_C     = sample['C_target']
        target_name  = sample['target_name']

        ##### oversampled domain surface
        points3D_target, points2D_target, _, _ = model(source_uvs)

        ### filter overlapping mesh
        domain_mask    = loss_obj.domain.domain_mask(points2D_target, None).cpu()
        faces_mask     = domain_mask[source_faces].prod(dim=-1).bool()
        faces_filtered = source_faces[faces_mask]

        self.append_surface_data(points3D_target * target_C, faces_filtered, source_uvs, 'surface', [target_name, 'filtered', 'oversampled'], ckpt_info)

