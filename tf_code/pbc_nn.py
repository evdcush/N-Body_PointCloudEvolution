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
X_data = utils.load_datum(n_P, 0.6, normalize_data=True)
#X_data = np.load('X16_06.npy')
K = 14
x = X_data[:8]

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
# maps and clones init
#=============================================================================
boundary_threshold = 0.1 # this represents absolute distance from a boundary
N, D = x_in.shape # number of particles



face_cubes   = {}
edge_cubes   = {}
corner_cubes = {}
boundary_remap = {}

expanded_cube = np.zeros((1,D))


def map_to_cubes():
    return false


#=============================================================================
# map and clone boundary particles
#=============================================================================
'''
The computation flow:
 - original cube is cloned, all original indices preserved
 - for each particle original cube, if particle is boundary:
     - add particle to cloned cube for as many dimensions as it needs to be
        + corner particle: readjusted and added 7 times
          eg (Top-front-left)
           - opposite corner cube: (Top-front-left) to (Bottom-rear-right)
           - opposite edge1 cube:  (Front-left) to (Rear-right)
           - opposite edge2 cube:  (Top-left) to (Bottom-right)
           - opposite edge3 cube:  (Top-front) to (Bottom-rear)
           - opposite face1 cube: (Top) to (Bottom)
           - opposite face2 cube: (Left) to (Right)
           - opposite face3 cube: (Front) to (Rear)
        + edge particle: readjusted and added 3 times
          eg (Rear-right)
           - opposite edge cube: (Rear-right) to (Front-left)
           - opposite face1 cube: (Rear) to (Front)
           - opposite face2 cube: (Right) to (Left)
        + face particle: readjusted and added 1 time
          eg (Bottom)
           - opposite face cube: (Bottom) to (Top)
     - each time the particle coordinates are readjusted and added, the index
       to which they were are added are mapped back to their original location
 - kneighbors graph on expanded clone cube
 - for each particle within clone_cube[:N], if neighbors outside N, remap
'''


def shift(p):
    """ p is (3,) coordinate vector, x, y, z
    corner: all 3 coordinates in threshold
    edge: 2 coordinates in threshold
    face: 1 coord in threshold
    """
    x, y, z = p




def build_boundary_dict(boundary_threshold):
    """ Note: all cube location names, like 'Top' or 'right' are arbitrary
    """
    # get boundary range
    lower = boundary_threshold
    upper = 1 - boundary_threshold

    # X Y Z
    left_bottom_rear = (lower, lower, upper)

def shift(_p, boundary_threshold):
    """ p is (3,) coordinate vector, x, y, z
    corner: all 3 coordinates in threshold
    edge: 2 coordinates in threshold
    face: 1 coord in threshold
Just do the entire array at once for x<lower x>upper

    """
    p = np.copy(_p)
    lower = boundary_threshold
    upper = 1 - boundary_threshold

    # get boundaries
    boundary_upper = p >= upper
    boundary_lower = p <= lower
    b_upper_count = np.sum(boundary_upper)
    b_lower_count = np.sum(boundary_lower)
    num_boundary_dims = b_upper_count + b_lower_count

    # ret
    shifted = []
    wrap = {()}

    #if num_boundary_dims == 0: # not boundary particle
    if num_boundary_dims > 0:
        lower_wrap = np.where(boundary_lower, 1, 0)
        upper_wrap = np.where(boundary_upper, -1, 0)
        wrap = lower_wrap + upper_wrap
        if num_boundary_dims == 1:
            face = p + wrap
            shifted.append(face)
        elif num_boundary_dims == 2:
            for i in range(b_lower_count):
                face =

            face1 = p + lower_wrap
            face2 = p + upper_wrap
            edge  = p + wrap




        # corner: all coords change
        # (.05, .05, .95) -> (1.05, 1.05, -0.05): (+1, +1, -1)
        wrapped_upper = np.copy(p) + np.where(boundary_upper, -1, 0)
        wrapped_lower = np.copy(p) + np.where(boundary_lower, 1, 0)
        wrapped_corner = np.where(boundary_upper, wrapped_upper, wrapped_lower)

        # edge: 2 coords change per
        # (.05, .05, .95) -> (1.05, 1.05, .95): (+1, +1, 0)
        wrapped_edge1 = np.copy(p) +
        if b_upper_count > b_lower_count:
            wrapped_edge = np.where(boundary_upper, wrapped_upper, p)

        # faces: 1 coord change per



    return shifted


def pad_cube_boundaries(x, boundary_threshold=0.1):
    """
    """
    # get boundary range
    lower = boundary_threshold
    upper = 1 - boundary_threshold

    # clone original array
    N, D = x.shape
    clone = np.copy(x[...,:3])
    idx_map = np.arange(N)

    # iterate through array and add boundary particles to clone
    for i in range(N):
        p = clone[i:i+1] # (1, 3)




def add_particle_to_cube(particle, cube):
    """ Adds a boundary particle to expanded cube
    All particles added to expanded cube. Boundary particles are readded and
    their index in the expanded cube is added to map
    Args:
        particle (ndarray): nbody particle, a (1,6) vector
        cube: the expanded cube to add the particle, of shape (*, 6),
              initially None
    Returns: cube with particle concatenated
    """
    if cube is None:
        cube = particle
    else:
        cube = np.concatenate((cube, particle), axis=0)
    return cube

def add_boundary_particle_idx_to_map(bounded_idx, original_idx, bmap):
    """ maps bounded particle to it's original location
    """
    bmap[bounded_idx] = original_idx
    return bmap







# JUST WORK ON SINGLE CUBE FOR NOW:
# put things in map first
for i in range(N):
    x = x_in[i]
    x_coo = x[:3]


#=============================================================================
# nearest neighbors graph and remap
#=============================================================================
'''
def get_adjacency_list(X_in):
    """ search for K nneighbors, and return offsetted indices in adjacency list
    Args:
        X_in (numpy ndarray): input data of shape (mb_size, N, 6)
    """
    mb_size, N, D = X_in.shape
    adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
    for i in range(mb_size):
        # this returns indices of the nn
        graph_idx = kneighbors_graph(X_in[i, :, :3], K, include_self=True).indices
        graph_idx = graph_idx.reshape([N, K]) + (N * i)  # offset idx for batches
        adj_list[i] = graph_idx
    return adj_list
'''

graph = kneighbors_graph()