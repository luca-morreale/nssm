
import numpy as np
import os
import torch
import igl
from argparse import ArgumentParser
from scipy.spatial import KDTree

from mesh_cut import cut_mesh
from utils import automap_paths
from utils import read_mesh
from utils import write_mesh
from utils import logging
from utils import print_info
from utils import compute_vertices_normal
from utils import upsample_mesh


def output_edges(filename, path):
    with open(filename, 'w') as stream:
        for (a, b) in path:
            stream.write(f'{int(a)} {int(b)}\n')


def process_shape(shape_path, cones_path, name):
    print_info(f'Start processing {shape_path}')

    #### 1. read mesh
    V, F, _, _ = read_mesh(shape_path)
    N = compute_vertices_normal(V, F)

    #### 2. read cones
    cones = np.loadtxt(cones_path)
    cones = cones.astype(int)

    #### 3. cut mesh
    data = cut_mesh(V, F, cones)
    V  = data['V']
    F  = data['F']
    UV = data['uv']
    C  = data['C']
    write_mesh(os.path.join(outfolder, 'debug.obj'), V*C, F, UV, None)

    #### 4. save cut mesh
    sample = {}
    sample['points'] = torch.from_numpy(V).float()
    sample['faces']  = torch.from_numpy(F).long()
    sample['param']  = torch.from_numpy(UV).float()
    sample['C']      = C
    sample['cut_edges'] = data['cut_edges']

    cut_path = []
    for (a, b) in zip(sample['cut_edges'][:-1], sample['cut_edges'][1:]):
        cut_path.append([int(a), int(b)])
    cut_path = torch.tensor(cut_path).int()
    # remove edges connecting two cones - not sure why it happens
    mask = cut_path == cones[0]
    for c in cones[1:]:
        mask = mask | (cut_path == c)
    mask = mask.sum(axis=1) > 1
    cut_path = cut_path[~mask]
    # if mask.sum() > 0:
    #     sample['cut_edges'] = sample['cut_edges'][~mask]
    sample['cut_path'] = cut_path


    #### 5. save mapping to original mesh
    uncut_to_cut = data['uncut_to_cut']
    cut_to_uncut = data['cut_to_uncut']

    sample['cones_uncut'] = torch.tensor(cones).long()
    sample['cones_cut']   = torch.from_numpy(np.hstack([ uncut_to_cut[el] for el in cones ])).long()

    sample['F_idx_uncut'] = torch.arange(sample['faces'].shape[0])
    sample['V_idx_uncut'] = torch.zeros(sample['points'].shape[0])

    for k, v in cut_to_uncut.items():
        sample['V_idx_uncut'][k] = v
    sample['V_idx_uncut'] = sample['V_idx_uncut'].long()

    # rearrange normals from original mesh
    sample['normals'] = torch.from_numpy(N).float()[sample['V_idx_uncut']]

    #### 6. save correspondences of duplicated points
    bnd = igl.boundary_loop(F)
    max_num_buddies = max([ len(el) for el in uncut_to_cut.values() ])

    sample['boundary']       = torch.from_numpy(bnd).long()
    boundary_buddies_indices = torch.ones(bnd.shape[0], max_num_buddies) * torch.arange(bnd.shape[0]).reshape(-1, 1)
    boundary_buddies_mask    = torch.zeros(bnd.shape[0], max_num_buddies).bool()
    boundary_buddies_indices = boundary_buddies_indices.long()

    for k, list_cut_idx in uncut_to_cut.items():
        # if it has only 1 correspondence then it is not a boundary point
        if len(list_cut_idx) == 1:
            continue
        # convert vertex index to boundary index
        bnd_indices = []
        for idx in list_cut_idx:
            # find index in the boundary list
            bnd_idx = np.nonzero(bnd == idx)[0][0]
            bnd_indices.append(bnd_idx)

        num_buddies = len(bnd_indices)-1
        for bnd_idx in bnd_indices:
            buddies = [ el for el in bnd_indices if el!=bnd_idx ]
            # update correspondences indices
            boundary_buddies_indices[bnd_idx, :num_buddies] = torch.tensor(buddies).long()
            # update mask to remember which indices are valid
            boundary_buddies_mask[bnd_idx, :num_buddies]    = True

    sample['boundary_buddies_indices'] = boundary_buddies_indices
    sample['boundary_buddies_mask']    = boundary_buddies_mask


    #### 7. upsample cut mesh
    print_info('Upsampling')
    points, faces, uvs = upsample_mesh(V, F, UV, threshold=0.3)
    sample['oversampled_points'] = torch.from_numpy(points).float()
    sample['oversampled_faces']  = torch.from_numpy(faces).long()
    sample['oversampled_param']  = torch.from_numpy(uvs).float()

    # find buddies for oversampled points
    bnd_ov = igl.boundary_loop(faces)
    tree = KDTree(points[bnd_ov])
    dist, idx = tree.query(points[bnd_ov], k=max_num_buddies)
    boundary_buddies_indices = torch.ones(bnd_ov.shape[0], max_num_buddies) * torch.arange(bnd_ov.shape[0]).reshape(-1, 1)
    boundary_buddies_mask    = torch.zeros(bnd_ov.shape[0], max_num_buddies).bool()
    boundary_buddies_indices = boundary_buddies_indices.long()

    for i, idx_i in enumerate(idx):
        mask_buddies = ~(dist[i] > 1.0e-4)
        boundary_buddies_indices[i, mask_buddies] = torch.from_numpy(idx_i[mask_buddies]).long()
        boundary_buddies_mask[i, mask_buddies]    = True

    sample['oversampled_boundary_buddies_indices'] = boundary_buddies_indices
    sample['oversampled_boundary_buddies_mask']    = boundary_buddies_mask

    print_info(f'Done')

    sample['name'] = name

    return sample


##############################################################################
parser = ArgumentParser(description='Cut mesh open')
parser.add_argument('--folder',  type=str, help='pair path',          required=True, default=None)
parser.add_argument('--verbose', action='store_true', help='verbose', required=False, default=False)
args = parser.parse_args()

logging.LOGGING_INFO = args.verbose

torch.set_grad_enabled(False)

##############################################################################
paths = automap_paths(args.folder)
outfolder = os.path.join(args.folder, 'samples')
os.makedirs(outfolder, exist_ok=True)

cones_path = os.path.join(args.folder, 'cut/source_cones.txt')
data = process_shape(paths['source_mesh'], cones_path, 'source')
torch.save(data, os.path.join(outfolder, 'source.pth'))
write_mesh(os.path.join(outfolder, 'source.obj'), data['points']*data['C'], data['faces'], data['param'], None)
output_edges(os.path.join(paths['cut'], 'source_edges.txt'), data['cut_path'])

cones_path = os.path.join(args.folder, 'cut/target_cones.txt')
data = process_shape(paths['target_mesh'], cones_path, 'target')
torch.save(data, os.path.join(outfolder, 'target.pth'))
write_mesh(os.path.join(outfolder, 'target.obj'), data['points']*data['C'], data['faces'], data['param'], None)
output_edges(os.path.join(paths['cut'], 'target_edges.txt'), data['cut_path'])
