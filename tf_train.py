import os, code, sys, time
import numpy as np
import tensorflow as tf
#import chainer.functions as F
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

import utils
import tf_nn
from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, RS_TAGS

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
#=============================================================================
# network and model params
#=============================================================================
# model params
channels = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 3]
kdims    = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
num_layers = len(kdims)
lr = 0.01
activation = tf.nn.relu


#=============================================================================
# Data params
#=============================================================================
# nbody data params
num_particles = 16 # base, actual num_particles**3
zX = 0.6
zY = 0.0

# Load data
rs_start  = utils.REDSHIFTS.index(zX)
rs_target = utils.REDSHIFTS.index(zY)
X = utils.load_npy_data(num_particles) # (11, N, D, 6)
X = X[[rs_start, rs_target]] # (2, N, D, 6)
X = utils.normalize_fullrs(X)
X_train, X_val = utils.multi_split_data_validation(X, num_val_samples=200)
X = None # reduce memory overhead

#=============================================================================
# initialize graph
#=============================================================================
# init network params
utils.init_params(kdims)

# direct graph
X_input = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_input')
X_truth = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_truth')
X_pred  = tf_nn.network_fwd(X_input, num_layers)

# loss and optimizer
readout = tf_nn.get_readout(X_pred)
loss    = tf_nn.pbc_loss(readout, X_truth)
train   = tf.train.AdamOptimizer(lr).minimize(loss)

#=============================================================================
# Training and Session setup
#=============================================================================
# training params
batch_size = 32
num_iters = 500
loss_history = np.zeros((num_iters))
verbose = True

# Sess
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())


#=============================================================================
# Training
#=============================================================================
for i in range(num_iters):
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=True)
    x_in   = _x_batch[0]
    x_true = _x_batch[1]
    error = sess.run(loss, feed_dict={X_input: x_in, X_truth: x_true})
    loss_history[i] = error
    if verbose:
        if i % 10 == 0:
            print('{}: {:.6f}'.format(i, error))
    train.run(feed_dict={X_input: x_in, X_truth: x_true})

#gvars = tf.global_variables()
code.interact(local=dict(globals(), **locals())) # DEBUGGING-use