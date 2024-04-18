
import os
import torch
import numpy as np
import trimesh
import scipy.sparse as sps

from argparse import ArgumentParser
from tqdm import trange

from utils import automap_paths
from utils import read_mesh
from utils import save_correspondences
from utils import logging
from utils import print_info
from utils import geodesic_distance
from utils import seed_everything


def accumulate_matches_to_matrix(path, faces_s, faces_t, matches, recompute=False):
    size    = (faces_s.shape[0], faces_t.shape[0])
    sps_mat = sps.coo_array(size) # empty matrix

    print_info('Computing matrix of matches')
    for i in trange(len(matches['source_faces'])):

        idx_tris_source = matches['source_faces'][i]
        idx_tris_target = matches['target_faces'][i]

        sps_mat = sps_mat + sps.coo_array((torch.ones(len(idx_tris_source)),
                                            (idx_tris_source.reshape(-1),
                                            idx_tris_target.reshape(-1))), shape=size)
    sps_mat = sps_mat.tocoo()

    return sps_mat

def compute_geodesic_distances(distances_path, verts, faces, norm_factor, topk_idx, k, recompute=False):

    def compute_distances(verts, faces, norm_factor, topk_idx, k):
        topk_distances = np.zeros([k,k])
        for i in trange(k-1):
            for j in range(i+1,k):
                dist = geodesic_distance(verts, faces,
                                faces[topk_idx[i]][0:1], faces[topk_idx[j]][0:1],
                                                    norm=False) / norm_factor
                topk_distances[i,j] = dist
        return topk_distances

    if os.path.exists(distances_path) and not recompute:
        topk_distances = torch.load(distances_path)
        if topk_distances.shape[0] > k:
            topk_distances = topk_distances[:k, :k]
        if topk_distances.shape[0] != k:
            topk_distances = compute_distances(verts, faces, norm_factor, topk_idx, k)
            torch.save(topk_distances, distances_path)
    else:
        topk_distances = compute_distances(verts, faces, norm_factor, topk_idx, k)
        torch.save(topk_distances, distances_path)
    return topk_distances

parser = ArgumentParser(description='Extract cut points')
parser.add_argument('--folder',      type=str, help='path to source mesh', required=True)
parser.add_argument('--dist',        type=float, help='minimum distance', required=False, default=0.3)
parser.add_argument('--topk',        type=int, required=False, default=3)
parser.add_argument('--seed',        type=int, required=False, default=111)
parser.add_argument('--mode',        type=str, choices=['fuzzy','strict'], required=False, default='strict')
parser.add_argument('--filtering',   action='store_true', required=False, default=False)
parser.add_argument('--verbose',     action='store_true', required=False, default=False)
parser.add_argument('--recompute',   action='store_true', required=False, default=False)
args = parser.parse_args()

logging.LOGGING_INFO = args.verbose

seed_everything(args.seed)
torch.set_grad_enabled(False)

paths = automap_paths(args.folder)

out_folder = paths['cut']
os.makedirs(out_folder, exist_ok=True)

#################################################
############        Load data        ############
#################################################

source_path  = os.path.join(paths['source_mesh'])
target_path  = os.path.join(paths['target_mesh'])
matches_path = os.path.join(paths['matches'])

print_info(f'Loading meshes')
verts_s, faces_s, _, _ = read_mesh(source_path)
verts_t, faces_t, _, _ = read_mesh(target_path)

print_info('Moving to torch')
verts_s = torch.from_numpy(verts_s).float()
verts_t = torch.from_numpy(verts_t).float()
faces_s = torch.from_numpy(faces_s).long()
faces_t = torch.from_numpy(faces_t).long()


print_info('Loading matches')
matches = torch.load(matches_path)

assert len(matches['source_faces']) == 200, "Number of view pairs should be 200"

idx_faces_source = torch.cat(matches['source_faces'], dim=0).numpy()
idx_faces_target = torch.cat(matches['target_faces'], dim=0).numpy()

idx_faces_source = np.unique(idx_faces_source)
idx_faces_target = np.unique(idx_faces_target)

filename = os.path.join(out_folder, 'matches_matrix.pth')
sps_mat = accumulate_matches_to_matrix(filename, faces_s, faces_t, matches, args.recompute)

#################################################
############        View data        ############
#################################################

length_mesh_s = trimesh.Trimesh(verts_s.numpy(), faces_s.numpy(), process=False).area
length_mesh_t = trimesh.Trimesh(verts_t.numpy(), faces_t.numpy(), process=False).area

best_target_by_source = np.array(sps_mat.argmax(axis=1).tolist()).reshape(-1)
best_matches_view     = best_target_by_source[idx_faces_source]

best_target_by_source = sps.csgraph.maximum_bipartite_matching(sps_mat, 'column')
best_source_by_target = sps.csgraph.maximum_bipartite_matching(sps_mat, 'row')

# idx_faces_target
mask_target = torch.ones(faces_t.shape[0]).bool()
mask_target[idx_faces_target] = False
best_source_by_target[mask_target] = -1

mask = best_target_by_source[idx_faces_source] >= 0
idx_faces_source = idx_faces_source[mask]

best_matches_view = best_target_by_source[idx_faces_source] # cols
best_match_for_target = best_source_by_target[best_matches_view]

best_buddies = best_match_for_target == idx_faces_source
# print(best_buddies.sum(), idx_faces_source.shape)

idx_faces_source  = idx_faces_source[best_buddies]
best_matches_view = best_target_by_source[idx_faces_source] # cols


# access matrix to get number of occurences
sps_csc_mat       = sps_mat.tocsc()
matches_scores = []
for r, c in zip(idx_faces_source, best_matches_view):
    matches_scores.append(sps_csc_mat[r, c])
matches_scores = np.array(matches_scores)

matches_score_idx = np.argsort(matches_scores)[::-1]

# print(np.sort(matches_scores)[::-1][:10])


#################################################
############        Top K            ############
#################################################
k = args.topk
print_info(f'Selecting top {k} points')

selected_topk_idx = [matches_score_idx[0]]

# compute geodesic distance and has to be greater then threshold
for i in matches_score_idx[1:]:
    is_too_close = False
    for j in selected_topk_idx:

        dist_s = geodesic_distance(verts_s, faces_s,
                        faces_s[idx_faces_source[i]][0:1], faces_s[idx_faces_source[j]][0:1],
                                            norm=False).item() / length_mesh_s
        dist_t = geodesic_distance(verts_t, faces_t,
                        faces_t[best_matches_view[i]][0:1], faces_t[best_matches_view[j]][0:1],
                                            norm=False).item() / length_mesh_t
        if (dist_s + dist_t) / 2.0 < args.dist:
            is_too_close = True
            break

    if not is_too_close:
        selected_topk_idx.append(i)

    if len(selected_topk_idx) == k:
        break


# topk_score_idx    = matches_score_idx[:k]
topk_score_idx    = selected_topk_idx
topk_faces_source = idx_faces_source[topk_score_idx]
topk_faces_target = best_matches_view[topk_score_idx]


print_info('Saving visualization of best points')
L1 = verts_s[faces_s[topk_faces_source][:, 0]].reshape(-1,3)
L2 = verts_t[faces_t[topk_faces_target][:, 0]].reshape(-1,3)

filename = os.path.join(out_folder, 'topk_points.ply')
# save_correspondences(filename, verts_s, faces_s, verts_t, faces_t, L1, L2, axis=2)
save_correspondences(filename, verts_t, faces_t, verts_s, faces_s, L2, L1, axis=2) 


print_info('Computing geodesic distance between best points')
distances_path = os.path.join(out_folder, 'distances_source.pth')
topk_distances_source = compute_geodesic_distances(distances_path, verts_s, faces_s, length_mesh_s, topk_faces_source, k, recompute=args.recompute)

distances_path = os.path.join(out_folder, 'distances_target.pth')
topk_distances_target = compute_geodesic_distances(distances_path, verts_t, faces_t, length_mesh_t, topk_faces_target, k, recompute=args.recompute)
topk_distances = topk_distances_source / 2 + topk_distances_target / 2 # account for target
topk_distances = topk_distances + topk_distances.T # make it symmetric

topk_distances_source = topk_distances_source + topk_distances_source.T # make it symmetric
topk_distances_target = topk_distances_target + topk_distances_target.T # make it symmetric

# DO NOT remove self-loop or the seam is going to have problems

# save cones such that the centeral points (min distance is last)
idx_center = np.argmin(topk_distances.max(axis=1))

# select index of the cone
source_cones = faces_s[topk_faces_source][:, 0].tolist()
target_cones = faces_t[topk_faces_target][:, 0].tolist()
# TODO find a better way to select the cone
# now is the first face index which might not be the correct match

if idx_center == 0:
    source_cones = source_cones[1:] + [source_cones[0]]
    target_cones = target_cones[1:] + [target_cones[0]]
elif idx_center < len(source_cones) - 1:
    source_cones = source_cones[:idx_center] + source_cones[idx_center+1:] + [source_cones[idx_center]]
    target_cones = target_cones[:idx_center] + target_cones[idx_center+1:] + [target_cones[idx_center]]
# if it is last then it is already done

# save cones
source_cones_path = os.path.join(out_folder, 'source_cones.txt')
target_cones_path = os.path.join(out_folder, 'target_cones.txt')

np.savetxt(source_cones_path, np.array(source_cones).astype(int))
np.savetxt(target_cones_path, np.array(target_cones).astype(int))

