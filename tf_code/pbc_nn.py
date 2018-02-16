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