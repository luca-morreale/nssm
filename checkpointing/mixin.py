
import os
import shutil

from utils import mkdir
from utils import tensor_to_numpy


class Mixin(object):

    def empty_function(self, **kwargs):
        pass

    # =================== Move torch data to numpy ===================== #
    def move_to_numpy(self, data):
        return tensor_to_numpy(data, squeeze=True)

    # =================== Create folder ================================ #
    def compose_out_folder(self, checkpoint_dir, sub_folders):
        out_folder = os.path.join(checkpoint_dir, *sub_folders)
        mkdir(out_folder)

        return out_folder

    def compute_filename(self, checkpoint_dir, folder, file_prefix, type, extension):
        out_folder = self.compose_out_folder(checkpoint_dir, [folder])
        name       = '{}{}.{}'.format(file_prefix, type, extension)
        filename   = '{}/{}'.format(out_folder, name)
        return name, filename

    def clean_folder(self, checkpoint_dir, folder):
        out_folder = self.compose_out_folder(checkpoint_dir, [folder])
        shutil.rmtree(out_folder, ignore_errors=True, onerror=None)
        mkdir(out_folder)

    # =================== Concat tags ================================ #
    def build_prefix_name(self, tags, ckpt_info, trailing=True):
        ## join list of tags
        out = '_'.join(tags) + '_'
        if self.save_timelapse and ckpt_info is not None:
            out +=  '{0:05d}_'.format(ckpt_info.epoch)

        if not trailing:
            out = out[:-1]
        return out
