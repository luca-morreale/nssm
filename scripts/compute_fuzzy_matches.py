
import mitsuba as mi
import argparse
import torch
import os
import numpy as np

mi.set_variant('llvm_ad_rgb')

from utils import logging
from utils import print_error
from utils import print_info

from dino_wrapper import Dino
from mitsuba_wrapper import Mitsuba


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield torch.stack(lst[i:i + n], dim=0)


parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
parser.add_argument('--load_size',     default=369, type=int, help='load size of the input image.')
parser.add_argument('--layer',         default=9, type=int, help="layer to create descriptors from.")
parser.add_argument('--bin',           action='store_true',  default=False, help="create a binned descriptor if True.")

parser.add_argument('--folder',        type=str, help='first shape', required=True)
parser.add_argument('--rotation',      type=float, help='rotation for ablation', default=None, required=False)
parser.add_argument('--num_rotations', default=10, type=int, help='Final number of correspondences.')
parser.add_argument('--save_imgs',     action='store_true', help='save images w/ correspondences.', required=False, default=False)
parser.add_argument('--save_mesh',     action='store_true', help='save meshes w/ correspondences.', required=False, default=False)
parser.add_argument('--debug',         action='store_true', help='debug visualizations', required=False, default=False)
parser.add_argument('--figure',        type=int, help='save specific visualization for the paper', required=False, default=None)
parser.add_argument('--mode',          type=str, choices=['full', 'bb', 'bidirectional'], help='what type of matches to use', default='full', required=False)
parser.add_argument('--verbose',       action='store_true', default=False, help='verbose', required=False)

args = parser.parse_args()

logging.LOGGING_INFO = args.verbose

torch.set_grad_enabled(False)

outfolder = os.path.join(args.folder, 'matches')
os.makedirs(outfolder, exist_ok=True)

shape1_path = os.path.join(args.folder, 'meshes/source.obj')
shape2_path = os.path.join(args.folder, 'meshes/target.obj')


rotations_path = os.path.join(args.folder, 'alignment/rotations.txt')
if not os.path.exists(rotations_path):
    print_error('Missing automatic alignment. Terminating.')
    exit(1)

# automatic read alignment
print_info('Reading automatic alignment')
lines = np.loadtxt(rotations_path).astype(np.float64)
R1 = lines[0, :]
R2 = lines[1, :]

if args.rotation is not None:
    R1 += args.rotation


mitsuba_helper = Mitsuba(shape1_path, shape2_path, args.num_rotations, radius=15, two_axis=True, resolution=896)
dino_helper    = Dino(args.mode, args.layer, 'key', args.bin, 0.05)

print_info('Start rendering images')
scenes, images1, images2 = mitsuba_helper.render_all_images(R1, R2)
print_info('Done rendering images')


print_info('Start extracting descriptor')
batched_images_1 = list(chunks(images1, 10))
descriptors_1, num_patches_1, saliency_maps_1, fg_masks_1 = dino_helper.run_dino(batched_images_1, permute=True)
batched_images_2 = list(chunks(images2, 10))
descriptors_2, num_patches_2, saliency_maps_2, fg_masks_2 = dino_helper.run_dino(batched_images_2, permute=True)
print_info('Done extracting descriptor')


descriptors_1   = torch.cat(descriptors_1, dim=0)
saliency_maps_1 = torch.stack(saliency_maps_1, dim=0)
fg_masks_1      = torch.stack(fg_masks_1, dim=0)
descriptors_2   = torch.cat(descriptors_2, dim=0)
saliency_maps_2 = torch.stack(saliency_maps_2, dim=0)
fg_masks_2      = torch.stack(fg_masks_2, dim=0)

print_info('Start lifting matches')
output = mitsuba_helper.lift_matches(dino_helper, scenes, images1[0].shape[:2], descriptors_1, descriptors_2, fg_masks_1, fg_masks_2, num_patches_1)
src_3D, tgt_3D, src_fcs, tgt_fcs, src_bary, tgt_bary, similarity = output
print_info('Done lifting matches')


if not (args.debug or args.figure is not None):
    print_info('Saving matches')
    torch.save({'source':src_3D, 'target':tgt_3D, 'source_faces':src_fcs, 'target_faces':tgt_fcs, 'source_bary':src_bary, 'target_bary':tgt_bary}, os.path.join(outfolder, 'matches.pth'))
