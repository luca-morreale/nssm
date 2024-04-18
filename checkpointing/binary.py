
import torch

from .mixin import Mixin


class BinaryCheckpointing(Mixin):

    # ================================================================== #
    # =================== save model =================================== #
    def save_model(self, checkpoint_folder, model, name=''):
        folder    = self.compose_out_folder(checkpoint_folder, ['models'])
        file_name = '{}'.format(name)

        # save weights
        model_path = '{}/weights{}.pth'.format(folder, file_name)
        torch.save(model.state_dict(), model_path)

        # save model object
        model_path = '{}/model{}.pth'.format(folder, file_name)
        torch.save(model, model_path)
    # ================================================================== #

