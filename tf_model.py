import os, code, sys, time
import numpy as np
import tensorflow as tf
import chainer.functions as F
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import utils

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
#=============================================================================
# Data
#=============================================================================
params_seed = 98765
data_seed   = 12345
def seed_rng(s=data_seed):
    np.random.seed(s)
    tf.set_random_seed(s)
    print('seeded by {}'.format(s))

num_particles = 16 # defaults 16**3
zX = 0.6
zY = 0.0
rs_start = utils.REDSHIFTS.index(zX)
rs_target = utils.REDSHIFTS.index(zY)
X = utils.load_npy_data(num_particles) # (11, N, D, 6)
X = X[[rs_start, rs_target]] # (2, N, D, 6)
X = utils.normalize_fullrs(X)
seed_rng()
X_train, X_val = utils.multi_split_data_validation(X, num_val_samples=200)
X = None # reduce memory overhead

#=============================================================================
# inits
#=============================================================================
def init_weight(k_in, k_out, name, scale=1.0):
    std = scale * np.sqrt(2. / k_in)
    henorm = tf.random_normal((k_in, k_out), stddev=std)
    W = tf.Variable(henorm, name=name, dtype=tf.float32)
    return W

def init_bias(k_in, k_out, name):
    b_val = np.ones((k_out,)) * 1e-6
    b = tf.Variable(b_val, name=name, dtype=tf.float32)
    return b

#=============================================================================
# nn ops
#=============================================================================
def init_weight(k_in, k_out, name, scale=1.0):
    std = scale * np.sqrt(2. / k_in)
    henorm = tf.random_normal((k_in, k_out), stddev=std)
    W = tf.Variable(henorm, name=name, dtype=tf.float32)
    return W

def init_bias(k_in, k_out, name):
    b_val = np.ones((k_out,)) * 1e-6
    b = tf.Variable(b_val, name=name, dtype=tf.float32)
    return b

def left_mult(h, W):
    return tf.einsum('ijl,lq->ijq', h, W)

def linear_layer(h, W, b):
    """ permutation equivariant linear transformation
    Args:
        h: external input, of shape (mb_size, n_P, k_in)
        W: layer weight, of shape (k_in, k_out)
        b: bias, of shape (k_out,)
    """
    mu = tf.reduce_mean(h, axis=1, keepdims=True)
    h_out = h - mu
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    h_out = left_mult(h, W) + b
    return h_out

#=============================================================================
# periodic boundary conditions, loss
#=============================================================================

def get_readout(h_out):
    gt_one  = (tf.sign(h_out - 1) + 1) / 2
    ls_zero = -(tf.sign(h_out) - 1) / 2
    rest = 1 - gt_one - ls_zero
    readout = rest*h_out + gt_one*(h_out - 1) + ls_zero*(1 + h_out)
    return readout

def periodic_boundary_dist(readout, x_truth):
    x_truth_coo = x_truth[...,:3]
    dist = tf.minimum(tf.square(readout - x_truth_coo), tf.square(readout - (1 + x_truth_coo)))
    dist = tf.minimum(dist, tf.square((1 + readout) - x_truth_coo))
    return dist

def pbc_loss(readout, x_truth):
    pbc_dist = periodic_boundary_dist(readout, x_truth)
    pbc_error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1), name='loss')
    return pbc_error

#=============================================================================
# model setup
#=============================================================================
# var naming
WEIGHT_TAG = 'W_{}'
BIAS_TAG   = 'B_{}' # eg 'B_6'

# model params
channels = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 3]
kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
lr = 0.01
activation = tf.nn.relu

#=============================================================================
# disgusting hardcoded tensorflow boilerplate placeholder/tensor bullshit
# I mean "graph"
#=============================================================================
# external data
X_input = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_input')
X_truth = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_truth')

# network layers and hyper params, 12 layers
#============================================================================= H0
W0 = init_weight(*kdims[0], WEIGHT_TAG.format(0))
B0 = init_bias(  *kdims[0],   BIAS_TAG.format(0))
H0 = activation(linear_layer(X_input, W0, B0))

#============================================================================= H1
W1 = init_weight(*kdims[1], WEIGHT_TAG.format(1))
B1 = init_bias(  *kdims[1],   BIAS_TAG.format(1))
H1 = activation(linear_layer(H0, W1, B1))

#============================================================================= H2
W2 = init_weight(*kdims[2], WEIGHT_TAG.format(2))
B2 = init_bias(  *kdims[2],   BIAS_TAG.format(2))
H2 = activation(linear_layer(H1, W2, B2))

#============================================================================= H3
W3 = init_weight(*kdims[3], WEIGHT_TAG.format(3))
B3 = init_bias(  *kdims[3],   BIAS_TAG.format(3))
H3 = activation(linear_layer(H2, W3, B3))

#============================================================================= H4
W4 = init_weight(*kdims[4], WEIGHT_TAG.format(4))
B4 = init_bias(  *kdims[4],   BIAS_TAG.format(4))
H4 = activation(linear_layer(H3, W4, B4))

#============================================================================= H5
W5 = init_weight(*kdims[5], WEIGHT_TAG.format(5))
B5 = init_bias(  *kdims[5],   BIAS_TAG.format(5))
H5 = activation(linear_layer(H4, W5, B5))

#============================================================================= H6
W6 = init_weight(*kdims[6], WEIGHT_TAG.format(6))
B6 = init_bias(  *kdims[6],   BIAS_TAG.format(6))
H6 = activation(linear_layer(H5, W6, B6))

#============================================================================= H7
W7 = init_weight(*kdims[7], WEIGHT_TAG.format(7))
B7 = init_bias(  *kdims[7],   BIAS_TAG.format(7))
H7 = activation(linear_layer(H6, W7, B7))

#============================================================================= H8
W8 = init_weight(*kdims[8], WEIGHT_TAG.format(8))
B8 = init_bias(  *kdims[8],   BIAS_TAG.format(8))
H8 = activation(linear_layer(H7, W8, B8))

#============================================================================= H9
W9 = init_weight(*kdims[9], WEIGHT_TAG.format(9))
B9 = init_bias(  *kdims[9],   BIAS_TAG.format(9))
H9 = activation(linear_layer(H8, W9, B9))

#============================================================================= H10
W10 = init_weight(*kdims[10], WEIGHT_TAG.format(10))
B10 = init_bias(  *kdims[10],   BIAS_TAG.format(10))
H10 = activation(linear_layer(H9, W10, B10))

#============================================================================= H11
W11 = init_weight(*kdims[11], WEIGHT_TAG.format(11))
B11 = init_bias(  *kdims[11],   BIAS_TAG.format(11))
H11 = linear_layer(H10, W11, B11)

#============================================================================= output
readout = get_readout(H11)
loss  = pbc_loss(readout, X_truth)
train = tf.train.AdamOptimizer(lr).minimize(loss)

#=============================================================================
# session and training
#=============================================================================
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())