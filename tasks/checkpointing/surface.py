
import logging

from runners import CheckpointRunner


class SurfaceCheckpointRunner(CheckpointRunner):

    def checkpoint_sample(self, sample, model, experiment, ckpt_info):
        param = sample['param']
        gts   = sample['gts']
        faces = sample['faces']
        name  = sample['name']
        param = self.move_to_device(param)
        gts   = self.move_to_device(gts)
        faces = self.move_to_device(faces)

        pred_points = model(param)
        pt_distance = (pred_points - gts).pow(2).sum(-1)

        scalars = {'l2error': pt_distance}

        self.append_surface_data(pred_points, faces, param, 'surface', [name], ckpt_info, scalars=scalars)
        self.append_surface_data(gts, faces, param, 'surface', ['GT', name], ckpt_info, constant=True)

        ## check using sampled param
        if 'oversampled_param' in sample:
            param_large = sample['oversampled_param']
            faces_large = sample['oversampled_faces']
            param_large = self.move_to_device(param_large)
            pred_points_large = model(param_large)

            self.append_surface_data(pred_points_large, faces_large, param_large, 'surface', [name, 'oversampled'], ckpt_info)

