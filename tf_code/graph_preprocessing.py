import numpy as np
from sklearn.neighbors import KDTree, kneighbors_graph, radius_neighbors_graph


import tf_utils as utils

#=============================================================================
# load data
#=============================================================================
num_particles = 32
zX = 0.6
zY = 0.0
X = utils.load_npy_data(num_particles, (zX, zY), normalize=True)
X_coo = X[...,:3]

X06 = X[0]
X00 = X[-1]

# batch
batch_size = 8

