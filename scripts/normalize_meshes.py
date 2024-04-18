
import os
import pymeshlab
from argparse import ArgumentParser

from utils import print_info
from utils import logging


def preprocess(shape_folder, shape_name):
    print_info(f'Processing {shape_name}')

    shape_path = os.path.join(shape_folder, f'meshes/{shape_name}.obj')

    ms   = pymeshlab.MeshSet()
    ms.load_new_mesh(shape_path)

    # preprocessing
    ms.compute_matrix_from_scaling_or_normalization(unitflag=True)
    ms.compute_matrix_from_translation(traslmethod=1)
    ms.compute_normal_per_vertex()
    ms.compute_normal_per_face()

    ms.save_current_mesh(shape_path)


parser = ArgumentParser(description='Convert files to off')
parser.add_argument('--folder', type=str, help='pair path',           required=True, default=None)
parser.add_argument('--verbose',   action='store_true', help='verbose', required=False, default=False)
args = parser.parse_args()

logging.LOGGING_INFO = args.verbose

preprocess(args.folder, 'source')
preprocess(args.folder, 'target')
