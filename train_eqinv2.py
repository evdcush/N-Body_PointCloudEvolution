import os, code, sys, time, argparse

import numpy as np
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf

import utils
import nn
from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, RS_TAGS

parser = argparse.ArgumentParser()
# argparse not handle bools well so 0,1 used instead
parser.add_argument('--particles', '-p', default=32,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--redshifts', '-z', default=[18,19], nargs='+', type=int, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--model_type','-m', default=0,          type=int,  help='model type')
parser.add_argument('--graph_var', '-k', default=14,       type=int, help='search parameter for graph model')
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
N = num_particles**3
redshift_steps = pargs['redshifts']
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1

# Load data
num_val_samples = 200
X = utils.load_zuni_npy_data(redshifts=redshift_steps, norm_coo=True)[...,:-1]
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
channels = [6, 32, 16, 8, 3]
channels[0]  = 9
#channels[0]  = 6
channels[-1] = 3
num_layers = len(channels) - 1

# model features
use_graph = model_type == 1
vcoeff = pargs['vel_coeff'] == 1

# hyperparameters
learning_rate = LEARNING_RATE # 0.01
M = pargs['graph_var']
threshold = 0.03

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
utils.init_sinv_params(channels, var_scope=vscope, restore=restore)

# INPUTS
#X_input_edges = tf.placeholder(tf.float32, shape=(N, M, 3, None))
#X_input_nodes = tf.placeholder(tf.float32, shape=(N, 3, None))
#X_truth       = tf.placeholder(tf.float32, shape=(N, 6, None))
X_input_edges = tf.placeholder(tf.float32, shape=(None, N, M, 3))
X_input_nodes = tf.placeholder(tf.float32, shape=(None, N, 3))
X_truth       = tf.placeholder(tf.float32, shape=(None, N, 6))

# ADJ LIST
#Graph_input = tf.placeholder(tf.int32, shape=(N, M, None))
Graph_input = tf.placeholder(tf.int32, shape=(None, 2))

def graph_get_func(h_in): # for tf.py_func
    return nn.get_kneighbor_alist(h_in, M, offset_idx=False, )#inc_self=False) # offset idx for batches

#=============================================================================
# Model predictions and optimizer
#=============================================================================
H_out  = nn.sinv_model_fwd(num_layers, X_input_edges, X_input_nodes, Graph_input, var_scope=vscope) # (b, N, M, 3)
H_pooled = tf.reduce_mean(H_out, axis=2)
X_pred = nn.get_readout(H_pooled)

# error and opt
error = nn.pbc_loss(X_pred, X_truth, vel=False)
train = tf.train.AdamOptimizer(learning_rate).minimize(error)

#=============================================================================
# Session and Train setup
#=============================================================================
# training params
batch_size = pargs['batch_size']
num_iters  = pargs['num_iters']
verbose    = pargs['verbose']

# Sess
gpu_frac = 0.85
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
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=False) # shape (2, b, N, 6)

    # split data
    x_in    = _x_batch[0] # (b, N, 6)
    x_truth = _x_batch[1] # (b, N, 6)

    # get adj_list
    adj_list = graph_get_func(x_in) # (b, N, M)

    # get edges (relative dists) and nodes (velocities)
    x_in_edges = nn.get_input_edge_features(x_in, adj_list) # (b, N, M, 3)
    x_in_nodes = nn.get_input_node_features(x_in) # (b, N, 3)

    # format input dims
    #x_in_edges = nn.sinv_dim_change(x_in_edges) # simply np.moveaxis
    #x_in_nodes = nn.sinv_dim_change(x_in_nodes) # simply np.moveaxis
    #x_truth  = nn.sinv_dim_changes(x_truth)
    #adj_list = nn.sinv_dim_changes(adj_list)

    # get idx list for tf.gather_nd
    idx_list = nn.alist_to_indexlist(adj_list)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use


    fdict = {X_input_edges: x_in_edges,
             X_input_nodes: x_in_nodes,
             X_truth: x_truth,
             Graph_input: idx_list}

    # training pass
    train.run(feed_dict=fdict)

    # save checkpoint
    if save_checkpoint(step):
        tr_error = sess.run(error, feed_dict=fdict)
        print('checkpoint {:>5}: {}'.format(step, tr_error))
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
    x_in    = X_test[0, j:j+1] # (1, n_P, 6)
    x_truth = X_test[1, j:j+1]

    # get adj_list
    adj_list = graph_get_func(x_in) # (b, N, M)

    # get edges (relative dists) and nodes (velocities)
    x_in_edges = nn.get_input_edge_features(x_in, adj_list) # (b, N, M, 3)
    x_in_nodes = nn.get_input_node_features(x_in) # (b, N, 3)

    # get idx list for tf.gather_nd
    idx_list = nn.alist_to_indexlist(adj_list)

    fdict = {X_input_edges: x_in_edges,
             X_input_nodes: x_in_nodes,
             X_truth: x_truth,
             Graph_input: idx_list}

    # validation outputs
    #vals = sess.run([X_pred, val_error, ground_truth_error], feed_dict=fdict)
    x_pred, v_error = sess.run([X_pred, error], feed_dict=fdict)
    code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    #x_pred, v_error, truth_error = vals
    test_loss_history[j] = v_error
    #inputs_loss_history[j] = truth_error
    test_predictions[j] = x_pred[0]
    print('{:>3d}: {:.6f}'.format(j, v_error))

# median test error
test_median = np.median(test_loss_history)
#inputs_median = np.median(inputs_loss_history)
print('{:<18} median: {:.9f}'.format(model_name, test_median))
#print('{:<18} median: {:.9f}, {:.9f}'.format(model_name, test_median, inputs_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss_history, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
