import os, code, sys, time, argparse

import numpy as np
import tensorflow as tf

import utils
import nn
from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, RS_TAGS

parser = argparse.ArgumentParser()
# argparse not handle bools well so 0,1 used instead
parser.add_argument('--particles', '-p', default=32,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--redshifts', '-z', default=[18,19], nargs='+', type=int, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--model_type','-m', default=0,          type=int,  help='model type')
parser.add_argument('--graph_var', '-k', default=0.08,       type=float, help='search parameter for graph model')
parser.add_argument('--restore',   '-r', default=0,          type=int,  help='resume training from serialized params')
parser.add_argument('--num_iters', '-i', default=1000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=8,          type=int,  help='training batch size')
parser.add_argument('--model_dir', '-d', default='./Model/', type=str,  help='directory where model parameters are saved')
parser.add_argument('--save_prefix','-n', default='',        type=str,  help='model name prefix')
parser.add_argument('--vel_coeff', '-c', default=0,          type=int,  help='use timestep coefficient on velocity')
parser.add_argument('--verbose',   '-v', default=0,          type=int,  help='verbose prints training progress')
pargs = vars(parser.parse_args())
start_time = time.time()

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
#=============================================================================
# nbody Data
#=============================================================================
# nbody data params
num_particles = 32 #pargs['particles']
redshift_steps = pargs['redshifts']
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1

# Load data
num_val_samples = 200
X = utils.load_rs_npy_data(redshift_steps, norm_coo=True, )#old_dataset=True)
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=num_val_samples)
X = None # reduce memory overhead


#=============================================================================
# network and model params
#=============================================================================
# model vars
model_type = pargs['model_type'] # 0: set, 1: graph
model_vars = utils.NBODY_MODELS[model_type]

# network kernel sizes and depth
#channels = model_vars['channels'] # OOM with sparse graph
channels = [6, 32, 16, 8, 3]#, 8, 16, 12, 16, 8, 3]
#channels = [6, 8, 12, 8, 4, 8, 12, 16, 8, 3]#, 8, 16, 12, 16, 8, 3]
channels[-1] = 3
channels[0] = 7
num_layers = len(channels) - 1

# model features
use_graph = model_type == 1
vcoeff = pargs['vel_coeff'] == 1

# hyperparameters
learning_rate = LEARNING_RATE # 0.01
G = pargs['graph_var']
threshold = 0.08

#=============================================================================
# Session save parameters
#=============================================================================
# model name
zX = redshift_steps[0]  # starting redshift
zY = redshift_steps[-1] # target redshift
model_name = utils.get_zuni_model_name(model_type, zX, zY, pargs['save_prefix'])

# save paths
paths = utils.make_save_dirs(pargs['model_dir'], model_name)
model_path, loss_path, cube_path = paths

# restore
restore = pargs['restore'] == 1

# save test data
#utils.save_test_cube(X_test, cube_path, (zX, zY), prediction=False)


#=============================================================================
# initialize graph and placeholders
#=============================================================================
# init network params
vscope = utils.VAR_SCOPE_SINGLE_MULTI.format(zX, zY)
tf.set_random_seed(utils.PARAMS_SEED)
utils.init_params(channels, var_scope=vscope, vel_coeff=vcoeff, restore=restore)

# INPUTS
#data_shape = (None, num_particles**3, 6)
data_shape = (None, num_particles**3, 7)
X_input = tf.placeholder(tf.float32, shape=data_shape, name='X_input')
X_truth = tf.placeholder(tf.float32, shape=data_shape, name='X_truth')

# RAD SPARSE TENSOR
Sparse_in = tf.sparse_placeholder(tf.float32)

def graph_get_func(h_in): # for tf.py_func
    #return nn.alist_to_indexlist(nn.get_pbc_kneighbors(h_in, K, threshold))
    return nn.get_radius_graph_input(h_in, G)

#=============================================================================
# Model predictions and optimizer
#=============================================================================
# model fwd dispatch args
if use_graph:
    margs = (X_input, num_layers, Sparse_in)
else:
    margs = (X_input, num_layers)

# network out
H_out  = nn.model_fwd(*margs, vel_coeff=vcoeff, var_scope=vscope)
X_pred = nn.get_readout(H_out)

# error and optimizer
error = nn.pbc_loss(X_pred, X_truth[...,:-1], )#vel=True)
train = tf.train.AdamOptimizer(learning_rate).minimize(error)
val_error = nn.pbc_loss(X_pred, X_truth[...,:-1]) # since training loss fn not always same
ground_truth_error = nn.pbc_loss(X_input[...,:-1], X_truth[...,:-1])


#=============================================================================
# Session and Train setup
#=============================================================================
# training params
batch_size = pargs['batch_size']
num_iters  = pargs['num_iters']
verbose    = pargs['verbose']

# Sess
gpu_frac = 0.9
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
if restore:
    utils.load_graph(sess, model_path)
# Save
train_loss_history = np.zeros((num_iters)).astype(np.float32)
saver = tf.train.Saver()
saver.save(sess, model_path + model_name)
checkpoint = 500
save_checkpoint = lambda step: (step+1) % checkpoint == 0 and step != 0

#=============================================================================
# TRAINING
#=============================================================================
start_time = time.time()
np.random.seed(utils.DATASET_SEED)
print('\nTraining:\n==============================================================================')
# START
for step in range(num_iters):
    # data batching
    _x_batch = utils.next_zuni_minibatch(X_train, batch_size, data_aug=True)
    #_x_batch = utils.next_minibatch(X_train, batch_size, data_aug=True)
    x_in    = np.copy(_x_batch[0])
    x_truth = _x_batch[1]
    fdict = {X_input: x_in, X_truth: x_truth}

    # feed graph model data
    if use_graph:
        #alist = nn.alist_to_indexlist(nn.get_kneighbor_alist(x_in, K))
        #alist = nn.alist_to_indexlist(nn.get_pbc_kneighbors(x_in, K, threshold))
        sparse_attribs = graph_get_func(x_in)
        fdict[Sparse_in] = sparse_attribs

    # training pass
    train.run(feed_dict=fdict)

    # save checkpoint
    if save_checkpoint(step):
        tr_error = sess.run(error, feed_dict=fdict)
        print('checkpoint {:>5}: {}'.format(step+1, tr_error))
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)
# END
print('elapsed time: {}'.format(time.time() - start_time))

# save
saver.save(sess, model_path + model_name, global_step=num_iters, write_meta_graph=True)
#if verbose: utils.save_loss(loss_path + model_name, train_loss_history)
X_train = None # reduce memory overhead

#=============================================================================
# EVALUATION
#=============================================================================
# data containers
num_test_samples = X_test.shape[1]
#test_predictions  = np.zeros(X_test.shape[1:]).astype(np.float32)
test_predictions  = np.zeros(X_test.shape[1:-1] + (channels[-1],)).astype(np.float32)
test_loss_history = np.zeros((num_test_samples)).astype(np.float32)
inputs_loss_history = np.zeros((num_test_samples)).astype(np.float32)

#print('\nEvaluation:\n==============================================================================')
for j in range(X_test.shape[1]):
    # validation inputs
    x_in   = np.copy(X_test[0, j:j+1]) # (1, n_P, 6)
    x_true = X_test[1, j:j+1]
    fdict = {X_input: x_in, X_truth: x_true}

    # feed graph data inputs
    if use_graph:
        sparse_attribs = graph_get_func(x_in)
        fdict[Sparse_in] = sparse_attribs

    # validation outputs
    vals = sess.run([X_pred, val_error, ground_truth_error], feed_dict=fdict)
    x_pred, v_error, truth_error = vals
    test_loss_history[j] = v_error
    inputs_loss_history[j] = truth_error
    test_predictions[j] = x_pred[0]
    #print('{:>3d}: {:.6f} | {:.6f}'.format(j, v_error, truth_error))

# median test error
test_median = np.median(test_loss_history)
inputs_median = np.median(inputs_loss_history)
#print('test median: {}'.format(test_median))
print('{:<18} median: {:.9f}, {:.9f}'.format(model_name, test_median, inputs_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss_history, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use