'''
This code is a python version of the "TreeCutter" from Euclidean Orbifold implementation
Original code: https://github.com/noamaig/euclidean_orbifolds
'''

import numpy as np
import igl

from utils import print_error
from utils import print_info
from utils import compute_mesh_geo_measures
from utils.parametrize import slim

from .tree_cutter import TreeCutter


class Flattener():
    # selfect that generates the embedding.

    def __init__(self, V, F, inds):
        self.V    = V
        self.F    = F
        self.inds = inds

    def get_cut_mesh(self):
        if not hasattr(self, 'V_cut'):
            print_error('You must first cut the mesh')
            exit(1)

        data = {}
        data['V']  = self.V_cut
        data['F']  = self.F_cut
        data['uv'] = self.uvs
        data['cut_to_uncut'] = self.cutIndsToUncutInds
        data['uncut_to_cut'] = self.uncutIndsToCutInds
        data['C'] = self.C
        data['cut_edges'] = np.concatenate(self.pathPairs_uncut, axis=0)


        return data



    def cut(self):
        # cut the mesh before flattening
        print_info('Start cutting')

        N = len(self.inds)
        if N == 3:
            root = N - 1
            fixedPairs = np.vstack([np.ones(N-1)*root, np.arange(N-1)]).T
        else:
            root = 1
            fixedPairs = np.array([[0, 2], [2, 3], [3, 1]])

        fixedPairs = fixedPairs.astype(int)

        tree = np.zeros([N, N])
        tree[fixedPairs[:, 0], fixedPairs[:, 1]] = 1
        tree = tree.astype(int)

        cutter = TreeCutter(self.V, self.F, tree, self.inds, root)

        print_info('Mesh cut start.')
        cutter.cutTree()

        print_info('Mesh cut done!')

        self.V_cut = cutter.V
        self.F_cut = cutter.F
        self.uncutIndsToCutInds = cutter.uncutIndsToCutInds
        self.cutIndsToUncutInds = cutter.cutIndsToUncutInds
        self.pathPairs = cutter.pathPairs
        self.pathPairs_uncut = cutter.uncut_paths

        # write_mesh('debug.ply', self.V_cut, self.F_cut, None, None)

    def make_linear_constraints(self, v1, v2, inds):
        # Add constraints so that all vertices in inds are sequentially
        # placed on a line between v1 and v2, with spacing proportional
        # to edge lengths

        # Measure the distance between consecutive vertices
        d = np.sqrt(np.sum((self.V_cut[inds[:-1], :] - self.V_cut[inds[1:], :])**2, axis=1))
        d = np.hstack([0, np.cumsum(d) / np.sum(d)])

        # Add the constraint
        constraints = []
        for i in range(len(inds) - 1):
            constraints.append(v1 * (1-d[i]) + v2 * d[i])

        return np.vstack(constraints)

    def flatten(self):
        # flatten the mesh to one of the orbifolds. In the end the
        # position of each vertex is stored in the property `flat_V`.

        if not hasattr(self, 'V_cut'):
            self.cut()

        print_info('Flattening')

        # # # # # # # # # # # # # # # # # # # # # # #
        #  Boundary conditions
        # # # # # # # # # # # # # # # # # # # # # # #

        startP = self.uncutIndsToCutInds[self.inds[0]]
        assert len(startP) == 1, 'length of startP wrong'

        # set boundary vertices to fixed positions on square.
        pathEnds = []
        pathEnds = [ pt_i[[0,-1]] for pt_i in self.pathPairs ]
        pathEnds = np.hstack(pathEnds).astype(int)
        pathEnds = np.unique(pathEnds)

        all_binds = igl.boundary_loop(self.F_cut)
        all_binds = all_binds[::-1]

        ind = np.where(all_binds == startP)[0][0]
        all_binds = np.concatenate((all_binds[ind:], all_binds[0:ind]))
        p = np.where(np.isin(all_binds, pathEnds))[0]

        side_0 = all_binds[p[0]:p[1]+1]
        side_1 = all_binds[p[1]:p[2]+1]
        side_2 = all_binds[p[2]:p[3]+1]
        side_3 = np.hstack([all_binds[p[3]:], all_binds[0]])

        uv_init_0 = self.make_linear_constraints(np.array([-1,  1]), np.array([ 1,  1]), side_0)
        uv_init_1 = self.make_linear_constraints(np.array([ 1,  1]), np.array([ 1, -1]), side_1)
        uv_init_2 = self.make_linear_constraints(np.array([ 1, -1]), np.array([-1, -1]), side_2)
        uv_init_3 = self.make_linear_constraints(np.array([-1, -1]), np.array([-1,  1]), side_3)

        bnd_uv = np.concatenate([uv_init_0, uv_init_1, uv_init_2, uv_init_3], axis=0)
        bnd = np.hstack([side_0[:-1], side_1[:-1], side_2[:-1], side_3[:-1]])

        assert bnd_uv.shape[0] == bnd.shape[0]
        assert bnd.shape[0] == igl.boundary_loop(self.F_cut).shape[0]


        print_info('Constraint generation done!')


        # compute the flattening by solving the boundary conditions
        # while satisfying the convex combination property with L
        print_info('Mapping to plane with slim!')

        bnd_uv = np.array(bnd_uv.tolist())
        V_cut  = np.array(self.V_cut.tolist())

        ## Harmonic parametrization for the internal vertices
        uv_init = igl.harmonic_weights(V_cut, self.F_cut, bnd, bnd_uv, 1)

        if len(igl.flipped_triangles(uv_init, self.F_cut).shape) > 0:
            uv_init = igl.harmonic_weights_uniform_laplacian(self.F_cut, bnd, bnd_uv, 1) # use uniform laplacian

        assert np.all(((uv_init[bnd] - bnd_uv)**2).sum(-1) <  1.0e-4), "modified boundary during initialization of uv"

        C, _ = compute_mesh_geo_measures(V_cut, self.F_cut, 4)

        bnd_constrain_weight = 1.0e35
        uvs = slim(V_cut*C, self.F_cut, uv_init, bnd, bnd_uv, 30, bnd_constrain_weight)

        assert np.all(((uvs[bnd] - bnd_uv)**2).sum(-1) <  1.0e-4), "modified boundary during parametrization"

        self.uvs = uvs
        self.C   = C

        print_info('Done flattening!')

        return uvs
