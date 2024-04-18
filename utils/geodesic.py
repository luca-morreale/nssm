
import numpy as np
import torch

from .logging import print_info


def geodesic_distance(pos, face, src=None, dest=None, norm=True, max_distance=None, num_workers=0):
    r"""
        Computes (normalized) geodesic distances of a mesh given by :obj:`pos`

        A copy paste from pytorch geometric
        except uses a different library for multiprocessing
        (the other one was crashing)
    """
    import gdist
    try:
        import multiprocess as mp
    except:
        print_info('multiprocess module not found')
        num_workers = 0

    max_distance = float('inf') if max_distance is None else max_distance

    if norm:
        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        norm = (area.norm(p=2, dim=1) / 2).sum().sqrt().item()
    else:
        norm = 1.0

    dtype = pos.dtype

    pos = pos.detach().cpu().to(torch.double).numpy()
    face = face.detach().cpu().to(torch.int).numpy()

    if src is None and dest is None:
        out = gdist.local_gdist_matrix(pos, face,
                                       max_distance * norm).toarray() / norm
        return torch.from_numpy(out).to(dtype)

    if src is None:
        src = np.arange(pos.shape[0], dtype=np.int32)
    else:
        src = src.detach().cpu().to(torch.int).numpy()

    dest = None if dest is None else dest.detach().cpu().to(torch.int).numpy()

    # def _parallel_loop(pos, face, src, dest, max_distance, norm, i):
    def _parallel_loop(args):
        pos, face, src, dest, max_distance, norm, i = args
        s = src[i:i + 1]
        d = None if dest is None else dest[i:i + 1]
        out = gdist.compute_gdist(pos, face, s, d, max_distance * norm) / norm
        return torch.from_numpy(out).float()
        # return out[0]



    num_workers = mp.cpu_count() if num_workers <= -1 else num_workers
    if num_workers > 0:
        p = mp.Pool(num_workers)
        outs = p.map(_parallel_loop, [(pos, face, src, dest, max_distance, norm, i)
                                                    for i in range(len(src))])
    else:
        outs = [ _parallel_loop([pos, face, src, dest, max_distance, norm, i])
                                                    for i in range(len(src)) ]

    out = torch.cat(outs, dim=0)

    if dest is None:
        out = out.view(-1, pos.shape[0])

    return out

