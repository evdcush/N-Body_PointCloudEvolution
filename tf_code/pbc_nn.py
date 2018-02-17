import numpy as np
import os, code, sys, time
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf
import chainer.functions as F
import tf_utils as utils

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
#=============================================================================
# load data
#=============================================================================
n_P = 32
#X_data = utils.load_datum(n_P, 0.6, normalize_data=True)
X_data = np.load('X16_06.npy')
X = X_data

'''
FULL CUBE
@ + + + + + + + + + + + + + + + @
+\                              +\
+ \                             + \
+  \                            +  \
+   \                           +   \
+    \                          +    \
+     @ + + + + + + + + + + + + + + + @
+     +                         +     +
+     +                         +     +
+     +                         +     +
+     +                         +     +
+     +                         +     +
+     +                         +     +
+     +                         +     +
+     +                         +     +
+     +                         +     +
@ + + + + + + + + + + + + + + + @     +
 \    +                          \    +
  \   +                           \   +
   \  +                            \  +
    \ +                             \ +
     \+                              \+
      @ + + + + + + + + + + + + + + + @

Notes:
3*4 edge cubes
2*4 corner mini-cubes
6   face cubes

'''


"""
Boundary threshold represents the absolute distance from a boundary
to be considered for possible bounding to another part of the cube.

Cube coordinates are rescaled to the [0, 1] interval. So a threshold of
0.1 means that if a particles is within [0, .1] or [.9, 1] along any coordinate
axis (X,Y,Z), then it is possible that it may be bounded to another part of the
cube.

e.g.: with threshold 0.1, and particle p has coord (0.4437, 0.9247, 0.2231)
than it may be bounded to another part of the cube in the next redshift,
like (0.4437, 0.0103, 0.2231)
"""
#=============================================================================
# boundary utils
#=============================================================================
def face_outer(particle, bound): # ret shape (1,3)
    # face only has one coordinate in boundary, so only one relocation
    ret = bound + particle
    return ret[None,:]

def edge_outer(particle, bound):
    # edge has two coordinates in boundary, so 3 relocations (edge, face, face)
    zero_idx = list(bound).index(0)
    edge = np.roll(np.array([[0,1,1],[0,1,0],[0,0,1]]), zero_idx, 1)
    return (edge * bound) + particle

def corner_outer(particle, bound): # ret shape (7, 3)
    # corner has 3 coordinates in boundary, so 7 relocations:
    # (corner, edge, edge, edge, face, face, face)
    corner = np.array([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1]])
    return (corner * bound) + particle

def get_outer(particle, bound, num_boundary):
    assert num_boundary > 0
    if num_boundary == 1:
        return face_outer(particle, bound)
    elif num_boundary == 2:
        return edge_outer(particle, bound)
    else:
        return corner_outer(particle, bound)

def pad_cube_boundaries(x, boundary_threshold):
    """ check all particles for boundary conditions and
    relocate boundary particles
    For map, just make a 1D vector V where
    V[i-N] = ith boundary particles original idx in X
    V[(i+1)-N] = (i+1)th boundary particles original idx in X

    I wonder if you could just do one corner_outer over the extracted corners
    in x, edge_outer on extracted edges, and so forth, while saving indices?
    Args:
        x (ndarray): data array, shape (n_P, 3)
    Returns: expanded x, index_list
    """
    N, D = x.shape
    idx_list = np.array([], dtype=np.int32) # keep in mind idx need to be offset by N

    # boundary
    lower = boundary_threshold
    upper = 1 - boundary_threshold
    bound_x = np.where(x >= upper, -1, np.where(x <= lower, 1, 0))
    bound_x_count = np.count_nonzero(bound_x, axis=-1)

    # get bound and add to clone
    for idx in range(N):
        num_boundary = bound_x_count[idx]
        if num_boundary > 0:
            # get particles to add to clone
            outer_particles = get_outer(x[idx], bound_x[idx], num_boundary)
            # add indices
            idx_list = np.append(idx_list, [idx] * outer_particles.shape[0])
            # concat to clone
            x = np.concatenate((x, outer_particles), axis=0)
    return x, idx_list

def get_kneighbors_pcube(x, idx_map, N, K):
    """ get kneighbor graph from padded cube
    x is padded cube of shape (M, 3),
    where M == (N + number of added boundary particles)
    Args:
        x (ndarray): padded cube, of shape (M, 3)
        idx_map (ndarray): shape (M-N,) indices
        N: number of particles in original cube
        K: number of nearest neighbors
    """
    kgraph = kneighbors_graph(x, K, include_self=True)[:N].indices
    kgraph_outer = kgraph >= N
    for k_idx, is_outer in enumerate(kgraph_outer):
        if is_outer:
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            outer_idx = kgraph[k_idx]
            kgraph[k_idx] = idx_map[outer_idx - N]
            #kgraph[k_idx] = idx_map[k_idx - N]
    return kgraph.reshape(N,K)


def pbc_kneighbors(X, K, boundary_threshold=0.1):
    """
    """
    # get boundary range
    lower = boundary_threshold
    upper = 1 - boundary_threshold
    mb_size, N, D = X.shape

    # graph init
    adjacency_list = np.zeros((mb_size, N, K), dtype=np.int32)

    for b in range(mb_size):
        # get expanded cube
        clone = np.copy(X[b,:,:3])
        padded_cube, idx_map = pad_cube_boundaries(clone, boundary_threshold)

        # get neighbors from padded_cube
        kgraph_idx = get_kneighbors_pcube(padded_cube, idx_map, N, K)
        adjacency_list[b] = kgraph_idx
    return adjacency_list



#=============================================================================
# nearest neighbors graph and remap
#=============================================================================

def get_adjacency_list(X_in, K):
    """ search for K nneighbors, and return offsetted indices in adjacency list
    Args:
        X_in (numpy ndarray): input data of shape (mb_size, N, 6)
    """
    mb_size, N, D = X_in.shape
    adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
    for i in range(mb_size):
        # this returns indices of the nn
        graph_idx = kneighbors_graph(X_in[i, :, :3], K, include_self=True).indices
        graph_idx = graph_idx.reshape([N, K]) #+ (N * i)  # offset idx for batches
        adj_list[i] = graph_idx
    return adj_list


#=============================================================================
# sample data
mb_size = 8
x = X[:mb_size]
K = 14

threshold = 0.1
alist_pbc = pbc_kneighbors(x, K, boundary_threshold=threshold)
alist_pbc2 = pbc_kneighbors(x, K, boundary_threshold=0.05)
alist_pbc_0 = pbc_kneighbors(x, K, boundary_threshold=0)
alist = get_adjacency_list(x, K)


total = np.prod(alist.shape)
larger = np.sum(alist_pbc == alist)
tigher = np.sum(alist_pbc2 == alist)
test = np.sum(alist == alist_pbc_0)

