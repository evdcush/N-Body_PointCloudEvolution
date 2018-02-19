import numpy as np
from sklearn.neighbors import KDTree, kneighbors_graph, radius_neighbors_graph
import sklearn as skl
import time, os, code, glob, struct
import pickle
import scipy

import tf_nn as nn
import tf_utils as utils

#=============================================================================
# load data
#=============================================================================
num_particles = 32
zX = 0.6
zY = 0.0
#X = utils.load_npy_data(num_particles, (zX, zY), normalize=True)
load_start_time = time.time()
#X = load_datum(num_particles, zX, normalize_data=True)[...,:3]
X = np.load('X06.npy')
print('loaded_data: {}'.format(time.time() - load_start_time))
X06 = X

# batch
batch_size = 8


#=============================================================================
# distance and nn functions
#=============================================================================
def distance_sklearn_metric(x, metric='euclidean'):
    dist = skl.metrics.pairwise.pairwise_distances(x, metric=metric)
    return dist

def get_pcube_csr(x, idx_map, N, K):
    """ get kneighbor graph from padded cube
    x is padded cube of shape (M, 3),
    where M == (N + number of added boundary particles)
    Args:
        x (ndarray): padded cube, of shape (M, 3)
        idx_map (ndarray): shape (M-N,) indices
        N: number of particles in original cube
        K: number of nearest neighbors
    """
    kgraph = kneighbors_graph(x, K, include_self=True)[:N]
    for i in range(len(kgraph.indices)):
        cur = kgraph.indices[i]
        if cur >= N:
            kgraph.indices[i] = idx_map[cur - N]
    return kgraph[:,:N].astype(np.float32)

#=============================================================================
# sandbox
#=============================================================================
x = X06[0] # (n_P, 3)

# pbc
lower = 0.08
upper = 1 - lower
x_clone = np.copy(x)
N, D = x_clone.shape
pad_start = time.time()
x_padded, idx_list = nn.pad_cube_boundaries(x_clone, lower)
print('make padded cube: {}'.format(time.time() - pad_start))


# dists
#x_dist = distance_sklearn_metric(x)
#x_pad_dist = distance_sklearn_metric(x_padded)[:N,:]
#idx = np.argsort(x_pad_dist)

# tree
leaf_size = 40 # 40 is default
tree_start = time.time()
tree = KDTree(x_padded, leaf_size=leaf_size)
print('made kdtree: {}'.format(time.time() - tree_start))

# csr
csr_test = scipy.sparse.csr_matrix((X.shape[0], N, N), [dtype='d'])
kgraph_csr = get_pcube_csr(x_padded, idx_list, N, 14)
