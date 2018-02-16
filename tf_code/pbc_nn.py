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
K = 30
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
boundary_threshold = 0.1 # this represents absolute distance from a boundary


face_cubes   = {}
edge_cubes   = {}
corner_cubes = {}
boundary_remap = {}

mb_size, N, D = x_in.shape # number of particles

# JUST WORK ON SINGLE CUBE FOR NOW:
for i in range(N):