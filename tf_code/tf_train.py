import os, code, sys, time, argparse
import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_utils as utils
import tf_nn as nn
from tf_utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, RS_TAGS

parser = argparse.ArgumentParser()
# argparse not handle bools well so 0,1 used instead
parser.add_argument('--particles', '-p', default=16,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--redshifts', '-z', default=[0.6, 0.0], nargs='+', type=float, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--model_type','-m', default=0,          type=int,  help='model type')
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
# nbody Data
#=============================================================================
# nbody data params
num_particles = pargs['particles']
zX, zY = pargs['redshifts']
nbody_params = (num_particles, (zX, zY))

# Load data
X = utils.load_npy_data(*nbody_params, normalize=True)
X_train, X_val = utils.split_data_validation_combined(X, num_val_samples=200)
X = None # reduce memory overhead

# velocity coefficient
vel_coeff = None
if pargs['vel_coeff']:
    vel_coeff = utils.load_velocity_coefficients(num_particles)[(zX, zY)]


#=============================================================================
# network and model params
#=============================================================================
# model vars
model_type = pargs['model_type'] # 0: set, 1: graph, but only set implemented at moment
channels = utils.NBODY_MODELS[model_type]['channels']
num_layers = len(channels) - 1
learning_rate = LEARNING_RATE # 0.01


#=============================================================================
# Session save parameters
#=============================================================================
# model name
mname_args = [nbody_params, model_type, vel_coeff, pargs['save_prefix']]
model_name = utils.get_model_name(*mname_args)

# save paths
paths = utils.make_save_dirs(pargs['model_dir'], model_name)
model_path, loss_path, cube_path = paths


#=============================================================================
# initialize graph
#=============================================================================
# init network params
utils.init_params(channels)

# direct graph
X_input = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_input')
X_truth = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_truth')
X_pred  = nn.network_fwd(X_input, num_layers, vel_coeff=vel_coeff, mtype_key=model_type)

# loss and optimizer
readout = nn.get_readout(X_pred)
loss    = nn.pbc_loss(readout, X_truth)
train   = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#=============================================================================
# Training and Session setup
#=============================================================================
# training params
batch_size = pargs['batch_size']
num_iters  = pargs['num_iters']
verbose    = pargs['verbose']

# Sess
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

# Save
loss_history = np.zeros((num_iters)).astype(np.float32)
saver = tf.train.Saver()
saver.save(sess, model_path)
save_checkpoint = lambda step: step % 100 == 0 and step != 0


#=============================================================================
# Training
#=============================================================================
for step in range(num_iters):
    # data
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=True)
    x_in   = _x_batch[0]
    x_true = _x_batch[1]

    # save error
    error = sess.run(loss, feed_dict={X_input: x_in, X_truth: x_true})
    loss_history[step] = error
    if verbose: print('{}: {:.6f}'.format(step, error))

    # cycle through graph
    train.run(feed_dict={X_input: x_in, X_truth: x_true})
    if save_checkpoint(step):
        saver.save(sess, model_path, global_step=step, write_meta_graph=False)

'''
>>> _y_batch = X_val[:, :8]
>>> y_in = _y_batch[0]
>>> y_true = _y_batch[1]
>>> y_pred = sess.run(X_pred, feed_dict={X_input:y_in})
>>> y_pred.shape
(8, 4096, 3)
'''

code.interact(local=dict(globals(), **locals())) # DEBUGGING-use