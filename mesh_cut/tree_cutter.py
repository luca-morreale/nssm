'''
This code is a python version of the "TreeCutter" from Euclidean Orbifold implementation
Original code: https://github.com/noamaig/euclidean_orbifolds
'''

import numpy as np
import igl
from collections import defaultdict

from utils import print_error
from utils import convert_mesh_to_graph
from utils import find_shortest_path


class TreeCutter():

    def __init__(self, V, F, tree, treeindices, root):

        self.V = V
        self.F = F
        self.N = V.shape[0]

        self.pathPairs = []
        self.treeRoot  = root
        self.treeStructure = tree

        self.treeIndices   = treeindices
        self.finishedPaths = 0
        self.alreadyCut    = False

        self.uncutIndsToCutInds = defaultdict(list)
        self.cutIndsToUncutInds = defaultdict(None)
        for i in range(self.N):
            self.uncutIndsToCutInds[i].append(i)
            self.cutIndsToUncutInds[i] = i

        self.directTree()



    def directTree(self):
        # make sure the tree is directed
        tree = self.treeStructure
        N    = tree.shape[0]
        directedTree = np.zeros([N, N])
        # Perform BFS on tree.
        # stack that holds nodes to visit
        roots=[self.treeRoot]
        # perform bfs
        while len(roots) > 0: # nodes in stack
            # pop node from stack
            root  = roots[0]
            roots = roots[1:]
            # find all nodes with edges to it
            sons = np.concatenate([np.nonzero(tree[root])[0], np.nonzero(tree[:, root])[0]], axis=0)

            # make sure none of the children are in roots - that would
            # mean a cycle in the original undirected tree also
            assert len(np.intersect1d(sons, roots)) == 0, "cycle in the original tree"

            # insert all the children as children of the current node
            directedTree[root, sons] = 1
            # delete the adjacencies between children and current node
            # (so as to not make double edges when visiting children)
            tree[root, sons] = 0
            tree[sons, root] = 0

            # add children to nodes to visit
            roots.extend(sons.tolist())

        self.treeStructure = directedTree.astype(int)


    def cutTree(self):
        if self.alreadyCut:
            print_error('Can only cut once!')
            exit(1)

        self.alreadyCut = 1
        self.cutTreeRecurse(self.treeRoot)


    def cutTreeRecurse(self, root):
        sons = np.where(self.treeStructure[root])[0]
        if not len(sons):
            return

        star_paths  = []
        uncut_paths = []
        sourceInd = self.treeIndices[root]
        for son in sons:
            targetInd = self.treeIndices[son]

            # splitCenterNode updates automatically the faces thus need to rebuild the graph every time
            graph = convert_mesh_to_graph(self.V, self.F)

            # find boundary
            binds = igl.boundary_loop(self.F)
            if len(binds) > 0:
                binds = np.setdiff1d(binds, np.array([sourceInd, targetInd]))
                # remove boundary nodes
                graph.remove_nodes_from(binds)

            newPath = find_shortest_path(sourceInd, targetInd, graph)
            uncut_paths.append(newPath)

            star_paths.append(self.split_mesh_by_path(newPath))
            self.finishedPaths += 1
        # star_paths = np.concatenate(star_paths, axis=0).astype(int)
        self.splitCenterNode(self.treeIndices[root], star_paths)
        self.uncut_paths = uncut_paths

        for son in sons:
            self.cutTreeRecurse(son)



    def newTrisToInsert(obj, tri, shared_edge, ind_to_insert):
        # find the index that is not part of the edge we are to split
        otherind = np.setdiff1d(tri, shared_edge)
        # find the place of the ind
        indplace = np.where(tri == otherind)[0][0]
        # set the tri s.t. the other ind is first and the edge to split is in [2 3]
        tri = np.concatenate((tri[indplace:], tri[:indplace]))
        # create the two tris: [new e1 split] and [new e3 split]
        two_tris = np.array([[tri[0], tri[1], ind_to_insert],
                             [ind_to_insert, tri[2], tri[0]]])
        for i in range(2):
            assert len(np.unique(two_tris[i,:])) == 3
        return two_tris


    def splitCenterNode(self, center ,starPathPairs):
        # after splitting a "star", that is all sons of a current root
        # node, we need to duplicate the root several times, as it is
        # not duplicated during the actual cutting.
        # center - index of the root of the "star"
        # starPathPairs - the pathPairs of the star


        # find all tris touching the center vertex
        inds = np.where(np.any(np.isin(self.F, center), axis=1))[0]
        # now gonna split the one-rign to groups of adjacent tris
        groups=[]
        # inds is the stack of tris to assign to a group
        while True:
            # get the first tri from the stack
            theGroup = inds[0]
            # now expand the group from the seed
            while True:
                # get all vertices in current tri group
                vs = np.unique(self.F[theGroup])
                # remove the center
                vs = np.setdiff1d(vs, center)

                # find all tris in the one ring that have a vertex in the group (not the center)
                newMembers = np.where(np.any(np.isin(self.F[inds], vs), axis=1))[0]
                # if exhausted all tris, stop
                if len(newMembers) == 0:
                    break
                # if found new members add them to group
                theGroup = np.hstack([theGroup, inds[newMembers]])
                # and remove them from the stack
                inds = np.delete(inds, newMembers)
            # add the new group
            groups.append(np.unique(theGroup))
            # if handled all tris in one-ring, finish.
            if len(inds) == 0:
                break

        # now insert copies of the center tri and update the adjacencies
        group_centers = []
        for i, g in enumerate(groups):
            # tris in current group
            t = self.F[g]
            # if it's the first group no need to assign a new ind, we will just use the existing one
            # (so it is assigned to one group)
            if i > 0:
                # insert copy of center
                centerInd = self.V.shape[0]
                self.V    = np.vstack((self.V, self.V[center]))
                self.cutIndsToUncutInds[centerInd] = center
                self.uncutIndsToCutInds[center].append(centerInd)
            else:
                centerInd = center
            # update all instances of original vertex with the new one
            t[t == center] = centerInd
            group_centers.append(centerInd)
            self.F[g] = t
            # correct the paths


        # for j in range(starPathPairs.shape[0]):
        new_star_path_pairs = [None for _ in starPathPairs]

        for j, pair in enumerate(starPathPairs):
            # get a pair of (coreposidning) paths
            # for each of the pair
            centers = np.full(2, np.nan)
            for k in range(2):
                for i, g in enumerate(groups):
                    # if this half of the pair is in g it should get the new ind.
                    # since the star paths do not contain the centerVertex if they
                    # share a member with the group it must be some other vertex
                    # than the center one.
                    if np.any(np.isin(pair[:, k], self.F[g])):
                        assert np.isnan(centers[k]), 'something is wrong'

                        centers[k] = group_centers[i]

            assert not np.any(np.isnan(centers)), "not fixing all centers"
            pair = np.vstack((centers, pair))
            new_star_path_pairs[j] = pair

        starPathPairs = new_star_path_pairs

        for j in range(len(self.pathPairs)):
            # get a pair of (corresponding) paths
            pair = self.pathPairs[j]

            # for each of the pair
            # TODO: make each pair correspond to group and then assign the
            # new center according to that.
            centers = np.full(2, np.nan)
            for k in range(2):
                for i in range(len(groups)):
                    # current group
                    g = groups[i]
                    # if this half of the pair is in g it should get the new ind
                    # for the old paths we check all inds except for the last one,
                    # as the last one cannot be a member of the groups unless its
                    # the center of the star, in which case it being a member is not
                    # indicative to which group this path belongs
                    if np.any(np.isin(pair[0:-1,k], self.F[g])):
                        assert(np.isnan(centers[k]))
                        centers[k] = group_centers[i]

            assert(np.isnan(centers[0]) == np.isnan(centers[1]))
            # if nan means this path is not part of the star - nothing to do
            if not np.isnan(centers[0]):
                pair[-1,:] = centers
                self.pathPairs[j] = pair

        self.pathPairs.extend(starPathPairs)


    def split_mesh_by_path(self, p):
        # split the mesh by a given list of indices that describe a list
        # of adjacent edges to cut.


        # TODO - need to check if crossing existing edge on other paths,
        # if so need to refuse split

        # will hold which tris are to left\right of cut
        # will hold which tris are to left\right of cut
        left  = []
        right = []

        # go over the entire path
        for e in zip(p[:-1], p[1:]):
            # find the two tris that are adjacent to it
            tris_to_split = np.where(np.sum(np.isin(self.F, e), axis=1) == 2)[0]

            assert len(tris_to_split) == 2

            # take the 1st tri to split of the pair
            tri = self.F[tris_to_split[0]]
            # check its orientation wrt the edge
            ind1 = np.where(tri == e[0])[0]# [0]
            ind2 = np.where(tri == e[1])[0]# [0]
            inds = [ind1, ind2]
            # positive orientation
            if (inds == [0, 1]) or (inds == [1, 2]) or (inds == [2, 0]):
                left.append(tris_to_split[0])
                right.append(tris_to_split[1])
            else: # negative orientation
                left.append(tris_to_split[1])
                right.append(tris_to_split[0])

        # find tris that touch ANY vertex on the path that's not an end point
        inds = np.where(np.any(np.isin(self.F, p[1:-1]), axis=1))[0]

        # remove from these the tris we already found to be adjacent to edges
        inds = np.setdiff1d(inds, left)
        inds = np.setdiff1d(inds, right)


        for _ in range(1000):
            # find all tris adjacent to a tri on the right side
            for j in range(len(right)):
                r = np.where(np.sum(np.isin(self.F[inds], self.F[right[j]]), axis=1) >= 2)[0]
                right = np.concatenate([right, inds[r]])
            # find all tris adjacent to a tri on the left side
            for j in range(len(left)):
                l = np.where(np.sum(np.isin(self.F[inds], self.F[left[j]]), axis=1) >= 2)[0]
                left = np.concatenate([left, inds[l]])
            # make sure left and right are adjoint
            right = np.setdiff1d(right, left)
            # remove the found tris from the inds
            inds = np.setdiff1d(inds, right)
            inds = np.setdiff1d(inds, left)
            # if finished all touching tris we can finish
            if len(inds) == 0:
                break

        # will hold the correspondences between the two sides of the seam
        cur_path_corr = []
        # go over all points not an end point
        for pj in p[1:-1]:
            # duplicate vertex
            new_pathV = self.V[pj]
            newInd    = self.V.shape[0]

            self.V = np.vstack([self.V, new_pathV])

            # we change the indices of all tris on the left side of the cut
            tleft = self.F[left]  # take the tris
            tleft[tleft == pj] = newInd  # replace the ind
            self.F[left] = tleft  # insert tris back
            cur_path_corr.append([pj, newInd])  # insert new pair into correspondence
            self.uncutIndsToCutInds[pj].append(newInd)
            self.cutIndsToUncutInds[newInd] = pj

        cur_path_corr = np.array(cur_path_corr)
        # add the last vertex on path. We do not split it, but we need it to keep
        # track of which vertices are on which edge
        cur_path_corr = np.vstack([cur_path_corr, [p[-1], p[-1]]])

        return cur_path_corr
