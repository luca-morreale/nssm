
import mitsuba as mi
import argparse
import torch
from tqdm import trange
import os
import numpy as np

mi.set_variant('llvm_ad_rgb')


from utils import logging
from utils import print_info
from utils import seed_everything

from dino_wrapper import Dino
from dino_wrapper import chunk_cosine_sim
from mitsuba_wrapper import Mitsuba


parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor point correspondences.')
parser.add_argument('--folder',        type=str, help='first shape', required=True)
parser.add_argument('--num_rotations', default=10, type=int, help='Final number of correspondences.')
parser.add_argument('--save_imgs',     action='store_true', help='save images w/ correspondences.', required=False, default=False)
parser.add_argument('--save_mesh',     action='store_true', help='save meshes w/ correspondences.', required=False, default=False)
parser.add_argument('--test',          action='store_true', help='test visualizations', required=False, default=False)
parser.add_argument('--mode',          type=str, choices=['full', 'bb', 'bidirectional'], help='what type of matches to use', default='bb', required=False)
parser.add_argument('--seed',          type=int, help='seed', default=None, required=False)
parser.add_argument('--layer',         default=11, type=int, help="layer to create descriptors from.")
parser.add_argument('--bin',           action='store_true',  default=False, help="create a binned descriptor if True.")
parser.add_argument('--verbose',       action='store_true', default=False, help='verbose', required=False)

args = parser.parse_args()

logging.LOGGING_INFO = args.verbose

if args.seed is not None:
    seed_everything(args.seed)

torch.set_grad_enabled(False)

outfolder = os.path.join(args.folder, 'alignment')
os.makedirs(outfolder, exist_ok=True)

random_rot1 = np.random.rand(1) * 360
random_rot2 = np.random.rand(1) * 360

shape1_path = os.path.join(args.folder, 'meshes/source.obj')
shape2_path = os.path.join(args.folder, 'meshes/target.obj')

mitsuba_helper = Mitsuba(shape1_path, shape2_path, args.num_rotations, radius=15, two_axis=False, resolution=448)
dino_helper = Dino('bb', args.layer, 'key', args.bin, 0.05)

print_info('Rendering all images')
_, images1, images2 = mitsuba_helper.render_all_images(random_rot1, random_rot2)
print_info('Done rendering images')

images1 = torch.stack(images1)
images2 = torch.stack(images2)


print_info('Extracting descriptors')
descriptors_1, num_patches_1, saliency_maps_1, fg_masks_1 = dino_helper.run_dino([images1], permute=True)
descriptors_2, num_patches_2, saliency_maps_2, fg_masks_2 = dino_helper.run_dino([images2], permute=True)
print_info('Done')

matches_mat = np.zeros([len(descriptors_1), len(descriptors_1)])


print_info('Start string matching')
for j in range(len(descriptors_2)):
    for i in range(len(descriptors_1)):

        ##########
        desc1 = descriptors_1[i].clone().to(dino_helper.device)
        desc2 = descriptors_2[j].clone().to(dino_helper.device)

        similarities = chunk_cosine_sim(desc1, desc2,
                                        fg_masks_1[i].to(dino_helper.device),
                                        fg_masks_2[j].to(dino_helper.device))

        # calculate best buddies
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1_fg = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2_fg = sim_2[0, 0], nn_2[0, 0]

        bbs_mask_fg = nn_2_fg[nn_1_fg] == torch.arange(nn_1_fg.shape[0], device=nn_1_fg.device)
        bbs_mask_fg_2 = nn_1_fg[nn_2_fg] == torch.arange(nn_2_fg.shape[0], device=nn_2_fg.device)
        num_matches = bbs_mask_fg.sum().cpu()
        matches_mat[j,i] = num_matches



repeated_mat = np.concatenate([matches_mat.T, matches_mat.T], axis=1)
alignments_numbers = []

for off in trange(len(descriptors_2)):
    alignments_numbers.append(np.trace(repeated_mat, offset=off))


idx_num  = np.argmax(alignments_numbers)
diagonal = np.diagonal(repeated_mat, offset=idx_num)
diagonal = np.concatenate([diagonal[idx_num:], diagonal[:idx_num]], axis=0)  # reorder diagonal
idx_front_view = np.argmax(diagonal)

print_info('Found match')


phis = mitsuba_helper.phis

rotation_source = (phis[idx_front_view] + random_rot1[0]) % 360
rotation_target = (phis[idx_front_view] + random_rot2[0] + phis[idx_num]) % 360

print_info('Saving to file')
# write for dino vit auto format
filename = os.path.join(outfolder, 'rotations.txt')
with open(filename, 'w') as stream:
    stream.write(f'{(phis[idx_front_view] + random_rot1[0]) % 360} 0.0\n')
    stream.write(f'{(phis[idx_front_view] + random_rot2[0] + phis[idx_num]) % 360} 0.0\n')

# write for humans to read
filename = os.path.join(outfolder, 'scores.txt')
with open(filename, 'w') as stream:
    stream.write(f'Random rotation shape 1: {random_rot1[0]} degrees\n')
    stream.write(f'Random rotation shape 2: {random_rot2[0]} degrees\n')
    stream.write(f'Initial misalignment of {(random_rot1 - random_rot2)[0]} degrees\n')
    stream.write('==================================================\n')

    stream.write('\n')
    stream.write('Number of matches:\n')
    for v in alignments_numbers:
        stream.write(f'{int(v):d} ')
    stream.write('\n')
    stream.write(f'Selected {idx_num} with rotation {phis[idx_num]} degrees\n')
    stream.write(f'Misalignment of {(random_rot1 - ((random_rot2 + phis[idx_num]) % 360))[0]} degrees\n')
    stream.write('==================================================\n')

    stream.write('\n')
    stream.write(f'Front view: {idx_front_view} in aligned views\n')
    stream.write(f'Rotation from random rotation: {phis[idx_front_view]}\n')
    stream.write('Number of matches:\n')
    for v in diagonal:
        stream.write(f'{int(v):d} ')
    stream.write('\n')
    stream.write('==================================================\n')
    stream.write(f'Complete rotation from start shape_1: {(phis[idx_front_view] + random_rot1[0]) % 360}\n')
    stream.write(f'Complete rotation from start shape_2: {(phis[idx_front_view] + random_rot2[0] + phis[idx_num]) % 360}\n')
    stream.write('==================================================\n')



if args.save_imgs:

    H = images1.shape[1]

    all_images2 = [ images2[idx_num:] ] + [ images2[:idx_num] ]
    images2 = torch.cat(all_images2, dim=0)

    aligned_images = np.concatenate([ images1.permute(1, 0, 2, 3).reshape(H, -1, 3),
                                      images2.permute(1, 0, 2, 3).reshape(H, -1, 3)], axis=0)
    im = dino_helper.to_pil(aligned_images)
    im.save(os.path.join(outfolder, f'aligned_images.png'))

    front_view = [ images1[idx_front_view], images2[idx_front_view] ]
    front_view = np.concatenate(front_view, axis=1)
    im         = dino_helper.to_pil(front_view)
    im.save(os.path.join(outfolder, f'front_view.png'))
