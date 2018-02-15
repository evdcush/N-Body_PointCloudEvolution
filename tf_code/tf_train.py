import os, code, sys, time
import numpy as np
import tensorflow as tf
#import chainer.functions as F
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

import tf_utils as utils
import tf_nn as nn
from tf_utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, RS_TAGS

parser = argparse.ArgumentParser()
# argparse not handle bools well so 0,1 used instead
parser.add_argument('--particles', '-p', default=16,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--redshifts', '-z', default=[0.6, 0.0], nargs='+', type=float, help='redshift tuple, predict z[1] from z[0]')
#parser.add_argument('--model_type','-m', default=0,          type=int,  help='model type')
parser.add_argument('--resume',    '-r', default=0,          type=int,  help='resume training from serialized params')
#parser.add_argument('--multi_step','-s', default=0,          type=int, help='use multi-step redshift model')
parser.add_argument('--num_iters', '-i', default=1000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=32,          type=int,  help='training batch size')
parser.add_argument('--model_dir', '-d', default='./Model/', type=str,  help='directory where model parameters are saved')
parser.add_argument('--save_prefix','-n', default='',         type=str,  help='model name prefix')
parser.add_argument('--vel_coeff', '-c', default=0,          type=int, help='use timestep coefficient on velocity')
parser.add_argument('--verbose',   '-v', default=1,          type=int, help='verbose prints training progress')
pargs = vars(parser.parse_args())
start_time = time.time()

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

# model path


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

# vel_coeff
#vel_coeff = None
vel_coeff = utils.load_velocity_coefficients(num_particles)[(zX, zY)]

#=============================================================================
# initialize graph
#=============================================================================
# init network params
utils.init_params(kdims)

# direct graph
X_input = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_input')
X_truth = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_truth')
X_pred  = nn.network_fwd(X_input, num_layers, vel_coeff=vel_coeff)

# loss and optimizer
readout = nn.get_readout(X_pred)
loss    = nn.pbc_loss(readout, X_truth)
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
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

# Save
saver = tf.train.Saver()
model_save_path = './Models/'
#utils.make_dirs([model_save_path])
model_name = model_save_path + 'test_model'
saver.save(sess, model_name)


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
        print('{}: {:.6f}'.format(i, error))
    train.run(feed_dict={X_input: x_in, X_truth: x_true})
    if i % 100 == 0 and i != 0:
        #saver.save(sess, model_name, global_step=i)
        saver.save(sess, model_name, global_step=i, write_meta_graph=False)