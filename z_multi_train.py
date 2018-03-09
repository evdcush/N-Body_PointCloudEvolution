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
parser.add_argument('--knn',       '-k', default=14,          type=int, help='number of nearest neighbors for graph model')
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
redshifts = pargs['redshifts']
zX = redshifts[0]
zY = redshifts[-1]

#zX, zY = 9.0, 0.0
nbody_params = (num_particles, (zX, zY))

# redshifts
num_rs = len(redshifts)
num_rs_layers = num_rs - 1
rs_tups = [(redshifts[i], redshifts[i+1]) for i in range(num_rs_layers)]

# Load data
X = utils.load_zuni_npy_data(redshifts=[zX, zY], norm_coo=True, norm_vel=True) # normalize only rescales coo for now
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=200)
X = None # reduce memory overhead
#print('{}: X.shape = {}'.format(nbody_params, X_train.shape))
#=============================================================================
# network and model params
#=============================================================================
# model vars
model_type = pargs['model_type'] # 0: set, 1: graph
use_graph  = model_type == 1
model_vars = utils.NBODY_MODELS[model_type]
channels = model_vars['channels']
channels[-1] = 6
num_layers = len(channels) - 1
vcoeff = pargs['vel_coeff'] == 1
#print('model_type: {}\nuse_graph: {}\nchannels:{}'.format(model_type, use_graph, channels))

# hyperparameters
learning_rate = LEARNING_RATE # 0.01
K = pargs['knn']
threshold = 0.08

#=============================================================================
# Session save parameters
#=============================================================================
# model name
#mname_args = [nbody_params, model_type, vel_coeff, pargs['save_prefix']]
#model_name = utils.get_model_name(*mname_args)
#model_name = utils.get_zuni_model_name(zX, zY, pargs['save_prefix'])
model_name = utils.get_zuni_model_name(model_type, zX, zY, pargs['save_prefix'])


# save paths
paths = utils.make_save_dirs(pargs['model_dir'], model_name)
model_path, loss_path, cube_path = paths

# restore
restore = pargs['restore'] == 1

# save test data
#utils.save_test_cube(X_test, cube_path, (zX, zY), prediction=False)


#=============================================================================
# initialize graph
#=============================================================================
# init network params
tf.set_random_seed(utils.PARAMS_SEED)
#vscope = utils.VAR_SCOPE_MULTI.format('{}_{}'.format(zX, zY))
#vscopes = ['params_{}_{}'.format(i, i+1) for i in range(zX, zY)]
vscopes = [utils.VAR_SCOPE_SINGLE_MULTI.format(tup[0], tup[1]) for tup in rs_tups]
utils.init_zuni_params_multi(channels, vscopes, graph_model=use_graph, restore=restore)

#var_scopes = [utils.VAR_SCOPE_MULTI.format(j) for j in range(num_rs_layers)]


# INPUTS
#data_shape = (num_rs, None, num_particles**3, 6)
data_shape = (None, num_particles**3, 6)
X_input = tf.placeholder(tf.float32, shape=data_shape, name='X_input')
X_truth = tf.placeholder(tf.float32, shape=data_shape, name='X_truth')

# ADJACENCY LIST
#alist_shape = (num_rs_layers, None, 2) # output shape
#alist_shape = (None, 2)
#adj_list = tf.placeholder(tf.int32, shape=alist_shape, name='adj_list')

# loss scaling weights
#scale_weights = tf.placeholder(tf.float32, shape=(num_rs_layers,), name='scale_weights')

# scheduled sampling probs
#sampling_probs = tf.placeholder(tf.bool, shape=(num_rs_layers-1,), name='sampling_probs')

def alist_func(h_in): # for tf.py_func
    #return nn.alist_to_indexlist(nn.get_pbc_kneighbors(h_in, K, threshold))
    return nn.alist_to_indexlist(nn.get_kneighbor_alist(h_in, K))


# model pred
if use_graph:
    margs = (X_input, vscopes, num_layers, alist_func, K)
else:
    margs = (X_input, vscopes, num_layers)

X_pred = nn.zuni_multi_single_model_fwd(*margs)

def vel_mse_fn(a, b):
    v_diff = tf.squared_difference(a[...,3:], b[...,3:])
    v_mse  = tf.reduce_mean(tf.reduce_sum(v_diff, axis=-1))
    return v_mse


# loss and optimizer
error = nn.pbc_loss(X_pred, X_truth)

train = tf.train.AdamOptimizer(learning_rate, name='AdamMulti').minimize(error)
ground_truth_error = nn.pbc_loss(X_input, X_truth)


#=============================================================================
# Session and Train setup
#=============================================================================
# training params
batch_size = pargs['batch_size']
num_iters  = pargs['num_iters']
verbose    = pargs['verbose']

# Sess
gpu_frac = 0.8
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
if restore:
    mp = './Model/ZG_{}-{}/Session/'
    mpaths = [mp.format(tup[0], tup[1]) for tup in rs_tups]
    utils.load_multi_graph(sess, vscopes, num_layers, mpaths, use_graph=use_graph)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
# Save
#train_loss_history = np.zeros((num_iters)).astype(np.float32)
#train = tf.train.AdamOptimizer(learning_rate, name='AdamMulti').minimize(error)
saver = tf.train.Saver()
saver.save(sess, model_path + model_name)
checkpoint = 200
save_checkpoint = lambda step: step % checkpoint == 0 and step != 0
#=============================================================================
# TRAINING
#=============================================================================
start_time = time.time()
np.random.seed(utils.DATASET_SEED)
for step in range(num_iters):
    # data
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=True)
    x_in    = _x_batch[0]
    x_truth = _x_batch[1]
    fdict = {X_input: x_in, X_truth: x_truth}#, adj_list: alist}
    # cycle through graph
    #train_vel.run(feed_dict=fdict)
    train.run(feed_dict=fdict)
    if save_checkpoint(step):
        train_error = sess.run(error, feed_dict=fdict)
        print('checkpoint {:>5}: {}'.format(step, error))
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)

print('elapsed time: {}'.format(time.time() - start_time))

# save
saver.save(sess, model_path + model_name, global_step=num_iters, write_meta_graph=True)
X_train = None # reduce memory overhead

#=============================================================================
# TESTING
#=============================================================================

# data containers
num_test_samples = X_test.shape[1]
test_predictions  = np.zeros(X_test.shape[1:]).astype(np.float32)
test_loss_history = np.zeros((num_test_samples)).astype(np.float32)
test_vel = np.zeros((num_test_samples)).astype(np.float32)

print('\nTesting:\n==============================================================================')
for j in range(X_test.shape[1]):
    # data
    x_in   = X_test[0, j:j+1] # (1, n_P, 6)
    x_true = X_test[1, j:j+1]
    fdict = {X_input: x_in, X_truth: x_true}#, adj_list: alist}

    # validation error
    vals = sess.run([X_pred, error, ground_truth_error], feed_dict=fdict)
    x_pred, val_error, truth_error = vals
    test_loss_history[j] = val_error
    #test_vel[j] = vel_error
    print('{:>3d}: {:.6f} | {:.6f}'.format(j, val_error, truth_error))
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

    # prediction
    test_predictions[j] = x_pred[0]

# median test error
test_median = np.median(test_loss_history)
#test_vel_median = np.median(test_vel)
print('test median: {}'.format(test_median))
#print('test median: {}\nvel  median: {}'.format(test_median, test_vel_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss_history, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use