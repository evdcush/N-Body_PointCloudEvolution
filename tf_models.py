import os, code, sys, time
import numpy as np
import tensorflow as tf
#import chainer.functions as F
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

import utils
import tf_nn
from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, RS_TAGS
#from utils import init_weight, init_bias

'''
I don't know if train has to be defined here with all the placeholders,
or whether it can be defined in train script
'''
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
#=============================================================================
# model setup
#=============================================================================
# model params
channels = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 3]
kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
lr = 0.01
activation = tf.nn.relu
num_particles = 16


def network_fwd(x_in, num_layers, activation=tf.nn.relu):
    H = x_in
    for i in range(num_layers):
        H = tf_nn.linear_layer(H, i)
        if i != num_layers - 1:
            H = activation(H)
    return H
'''
Guess you ahve to do training here too?
it complains about not knowing X_input in tf_train.py
'''
X_input = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_input')
X_truth = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_truth')

#============================================================================= output
utils.init_params(kdims)
num_layers = len(kdims)
H_out = network_fwd(X_input, num_layers)
#H_out = tf_nn.network_fwd(X_input, num_layers)
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
readout = tf_nn.get_readout(H_out)
loss  = tf_nn.pbc_loss(readout, X_truth)
train = tf.train.AdamOptimizer(lr).minimize(loss)

# data
params_seed = 98765
data_seed   = 12345
def seed_rng(s=data_seed):
    np.random.seed(s)
    tf.set_random_seed(s)
    print('seeded by {}'.format(s))

num_particles = 16 # defaults 16**3
zX = 0.6
zY = 0.0
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
rs_start = utils.REDSHIFTS.index(zX)
rs_target = utils.REDSHIFTS.index(zY)
X = utils.load_npy_data(num_particles) # (11, N, D, 6)
X = X[[rs_start, rs_target]] # (2, N, D, 6)
X = utils.normalize_fullrs(X)
seed_rng()
X_train, X_val = utils.multi_split_data_validation(X, num_val_samples=200)
X = None # reduce memory overhead

# training params
batch_size = 32
num_iters = 500


# Sess
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

loss_history = np.zeros((num_iters))
verbose = True

for i in range(num_iters):
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=True)
    x_in   = _x_batch[0]
    x_true = _x_batch[1]

    if verbose:
        error = sess.run(loss, feed_dict={X_input: x_in, X_truth: x_true})
        loss_history[i] = error
        if i % 10 == 0:
            print('{}: {:.6f}'.format(i, error))
    train.run(feed_dict={X_input: x_in, X_truth: x_true})

gvars = tf.global_variables()
code.interact(local=dict(globals(), **locals())) # DEBUGGING-use