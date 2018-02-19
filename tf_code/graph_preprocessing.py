import numpy as np
from sklearn.neighbors import KDTree, kneighbors_graph, radius_neighbors_graph
import sklearn as skl
import time, os, code
import hickle as hkl


import tf_utils as utils
import tf_nn as nn

#=============================================================================
# load data
#=============================================================================
num_particles = 32
zX = 0.6
zY = 0.0
#X = utils.load_npy_data(num_particles, (zX, zY), normalize=True)
load_start_time = time.time()
X = utils.load_datum(num_particles, zX, normalize_data=True)[...,:3]
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


#
hkl.dump(tree, 'test.hkl', mode='w')
hkl.dump(tree, 'test_gzip.hkl', mode='w', compression='gzip')

print('uncompressed: {:.5f} bytes'.format(os.path.getsize('test.hkl') / 1e-6))
print('compressed: {:.5f} bytes'.format(os.path.getsize('test_gzip.hkl') / 1e-6))

tree_copy = hkl.load('test_gzip.hkl')
