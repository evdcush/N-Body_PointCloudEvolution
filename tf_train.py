import os, code, sys, time
import numpy as np
import tensorflow as tf
#import chainer.functions as F
import matplotlib.pyplot as plt
import utils
from tf_models import train, loss

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
num_iters = 1000


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