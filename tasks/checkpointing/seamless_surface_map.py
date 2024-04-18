
import torch

from .surface_map import SurfaceMapCheckpointRunner

from utils import save_uv_layout


class SeamlessMapCheckpointRunner(SurfaceMapCheckpointRunner):

    def forward(self, sample, model, experiment):
        torch.set_grad_enabled(False)

        model.R = self.move_to_device(sample['R'])
        return super().forward(sample, model, experiment)


    def forward_losses(self, model_outputs, sample, loss_obj):
        return [], {}


    def checkpoint(self, model_outputs, losses, model, sample, experiment, ckpt_info, scalars):
        points2D_target = model_outputs[1]

        face_masks = self.filter_faces(model_outputs, sample, experiment['loss'])

        ###### SAVE surfaces
        self.save_all_surfaces(sample, model_outputs, face_masks, ckpt_info)
        self.save_landmarks3D(model_outputs, face_masks, model, sample, ckpt_info)


        save_uv_layout('mapping.png', model_outputs[1], sample['source_faces'])
        save_uv_layout('mapping_no_rot.png', model_outputs[2], sample['source_faces'])
        save_uv_layout('mapping_source.png', model_outputs[3], sample['source_faces'])

        # remove 30% of matches based on distance every 5k iterations
        if ((ckpt_info.epoch + 1) % 10000 == 0) and ckpt_info.epoch > 0:
            percent = 0.2
            src_matches, tgt_matches = experiment['dataset'].dataset.get_matches()
            src_matches = src_matches.cuda()
            tgt_matches = tgt_matches.cuda()
            pts3D, _, _, _, _ = model.forward(src_matches, sample['R'], sample['C_target'], sample['C_source'])
            pts3D_target = model.target_surface(tgt_matches) * sample['C_target']

            distances = (pts3D - pts3D_target).pow(2).sum(-1)
            smallest_outlier_distance  = torch.topk(distances, k=int(distances.shape[0] * percent))[0][-1]
            mask_matches = distances > smallest_outlier_distance

            # lambda_ = -torch.log(0.01) / smallest_outlier_distance

            experiment['dataset'].dataset.remove_matches(mask_matches.cpu())

            # TODO soft weigthing on the matches
            if (ckpt_info.epoch + 1) >= 10000:
                # TODO increase the weighting of the landmark terms or get subjugated
                experiment['loss'].reg_landmarks3D *= 1.42



    def save_all_surfaces(self, sample, model_outputs, faces_filtered, ckpt_info):
        self.save_map_as_surface(model_outputs, faces_filtered, sample, ckpt_info)


    def filter_faces(self, model_outputs, sample, loss_obj):
        source_faces    = sample['source_faces']
        points2D_target = model_outputs[1]

        domain_masks = loss_obj.domain.domain_mask(points2D_target, None).cpu()
        faces_masks  = domain_masks[source_faces].prod(dim=-1).bool()

        return faces_masks.cpu()


    def save_map_as_surface(self, model_outputs, faces_masks, sample, ckpt_info):
        source_name  = sample['source_name']
        source_faces = sample['source_faces']

        points3D_target, _, _, source_uvs, _ = model_outputs
        visible_faces       = source_faces[faces_masks]
        self.append_surface_data(points3D_target, visible_faces, source_uvs, 'surface', [source_name, 'map'], ckpt_info)

    def save_landmarks3D(self, model_outputs, faces_masks, model, sample, ckpt_info):
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

        points3D_target, _, _, _, points3D_source = model_outputs

        lands3D_source = points3D_source[lands_mask]
        lands3D_mapped = points3D_target[lands_mask]
        lands3D_target = model.target_surface(lands_target) * target_C

        faces_filtered = source_faces[faces_masks]

        ### landmarks
        self.append_surface_data([source_GT_points, target_GT_points], [source_faces, target_faces],
                                None, 'correspondences', ['GT', source_name, target_name],
                                ckpt_info, scalars={'landmarks':[lands3D_source, lands3D_target]}, constant=True)
        self.append_surface_data([source_GT_points, points3D_target], [source_faces, faces_filtered],
                                None, 'correspondences', [source_name, target_name],
                                ckpt_info, scalars={'landmarks':[lands3D_source, lands3D_mapped]})

