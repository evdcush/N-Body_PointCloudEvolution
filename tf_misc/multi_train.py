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
parser.add_argument('--redshifts', '-z', default=[0.6, 0.0], nargs='+', type=float, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--model_type','-m', default=0,          type=int,  help='model type')
parser.add_argument('--knn',       '-k', default=14,          type=int, help='number of nearest neighbors for graph model')
#parser.add_argument('--resume',    '-r', default=0,          type=int,  help='resume training from serialized params')
parser.add_argument('--num_iters', '-i', default=1000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=4,          type=int,  help='training batch size')
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
zX, zY = pargs['redshifts'] # strictly for model naming
nbody_params = (num_particles, (zX, zY))

# redshifts
redshift_steps = [6.0, 1.5, 1.0, 0.4, 0.0] # true redshifts
redshift_idx   = [utils.REDSHIFTS.index(rs) for rs in redshift_steps]
num_rs = len(redshift_idx)
num_rs_layers = num_rs - 1

# Load data
X = utils.load_npy_data(num_particles, normalize=True)[redshift_idx]
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=200)
X = None # reduce memory overhead
#print('{}: X.shape = {}'.format(nbody_params, X_train.shape))

# velocity coefficient
vel_coeff = None
if pargs['vel_coeff']:
    vel_coeff = utils.load_velocity_coefficients(num_particles)[(zX, zY)]

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
#print('model_type: {}\nuse_graph: {}\nchannels:{}'.format(model_type, use_graph, channels))

# hyperparameters
learning_rate = LEARNING_RATE # 0.01
K = pargs['knn']
threshold = 0.08

#=============================================================================
# Session save parameters
#=============================================================================
# model name
mname_args = [nbody_params, model_type, vel_coeff, pargs['save_prefix']]
model_name = utils.get_model_name(*mname_args)

# save paths
paths = utils.make_save_dirs(pargs['model_dir'], model_name)
model_path, loss_path, cube_path = paths

# save test data
#utils.save_test_cube(X_test, cube_path, (zX, zY), prediction=False)


#=============================================================================
# initialize graph
#=============================================================================
# init network params
tf.set_random_seed(utils.PARAMS_SEED)
utils.init_params_multi(channels, num_rs_layers, graph_model=use_graph, seed=utils.PARAMS_SEED)
# restore
'''
with tf.Session() as sess:
    restore_model = tf.train.import_meta_graph('./Model/Multi_G_32_06-00_')
'''
var_scopes = [utils.VAR_SCOPE_MULTI.format(j) for j in range(num_rs_layers)]


# INPUTS
data_shape = (num_rs, None, num_particles**3, 6)
X_input = tf.placeholder(tf.float32, shape=data_shape, name='X_input')

# ADJACENCY LIST
#alist_shape = (num_rs_layers, None, 2) # output shape
#adj_list = tf.placeholder(tf.int32, shape=alist_shape, name='adj_list')

# loss scaling weights
#scale_weights = tf.placeholder(tf.float32, shape=(num_rs_layers,), name='scale_weights')

# scheduled sampling probs
#sampling_probs = tf.placeholder(tf.bool, shape=(num_rs_layers-1,), name='sampling_probs')

def alist_func(h_in): # for tf.py_func
    return nn.alist_to_indexlist(nn.get_pbc_kneighbors(h_in, K, threshold))

#X_pred, loss = nn.multi_model_fwd_sampling(X_input, var_scopes, num_layers, adj_list, K, sampling_probs)
#X_pred, loss = nn.multi_model_fwd(X_input, var_scopes, num_layers, adj_list, K)
#X_pred_val, loss_val = nn.multi_func_model_fwd(X_input, var_scopes, num_layers, alist_func, K)

# for vel
#X_pred, loss = nn.multi_model_vel_fwd(X_input, var_scopes, num_layers, adj_list, K)
#X_pred_val, loss_val = nn.multi_func_model_vel_fwd(X_input, var_scopes, num_layers, alist_func, K)

# SET
X_pred, loss = nn.multi_model_fwd_set(X_input, var_scopes, num_layers)

# loss and optimizer
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
est_error = nn.pbc_loss(X_pred, X_input[-1]) # this just for evaluation


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

# Save
train_loss_history = np.zeros((num_iters)).astype(np.float32)
saver = tf.train.Saver()
saver.save(sess, model_path + model_name)
checkpoint = 500
save_checkpoint = lambda step: step % checkpoint == 0 and step != 0

#=============================================================================
# TRAINING
#=============================================================================
start_time = time.time()
np.random.seed(utils.DATASET_SEED)
for step in range(num_iters):
    # data
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=True)
    x_in = _x_batch
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    #sweights = nn.error_scales(np.copy(x_in))
    #sprobs = np.random.sample(num_rs_layers-1) < (step / num_iters)
    #alist = [nn.alist_to_indexlist(nn.get_pbc_kneighbors(x_in[j], K, threshold)) for j in range(num_rs_layers)]
    #fdict = {X_input: x_in, adj_list: alist, }#sampling_probs:sprobs, }#scale_weights:sweights}
    fdict = {X_input: x_in}

    if verbose:
        error = sess.run(loss, feed_dict=fdict)
        train_loss_history[step] = error
        print('{}: {:.8f}'.format(step, error))

    # cycle through graph
    train.run(feed_dict=fdict)
    if save_checkpoint(step):
        error = sess.run(loss, feed_dict=fdict)
        print('checkpoint {:>5}: {}'.format(step, error))
        #wr_meta = step == checkpoint # only write on first checkpoint
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)

print('elapsed time: {}'.format(time.time() - start_time))

# save
saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=False)
if verbose: utils.save_loss(loss_path + model_name, train_loss_history)
X_train = None # reduce memory overhead

#=============================================================================
# TESTING
#=============================================================================

# data containers
num_test_samples = X_test.shape[1]
test_predictions  = np.zeros(X_test.shape[1:]).astype(np.float32)
test_loss_history = np.zeros((num_test_samples)).astype(np.float32)

print('\nTesting:\n==============================================================================')
for j in range(X_test.shape[1]):
    # data
    #x_in   = X_test[0, j:j+1] # (1, n_P, 6)
    #x_true = X_test[1, j:j+1]
    x_in = X_test[:,j:j+1]
    #alist = [nn.alist_to_indexlist(nn.get_pbc_kneighbors(x_in[j], K, threshold)) for j in range(0, num_rs_layers)]
    #alist = nn.alist_to_indexlist(nn.get_pbc_kneighbors(x_in[0], K, threshold))
    fdict = {X_input: x_in}#adj_list: alist}

    # validation error
    #error = sess.run(loss, feed_dict=fdict)
    error = sess.run(est_error, feed_dict=fdict)
    test_loss_history[j] = error
    print('{}: {:.6f}'.format(j, error))

    # prediction
    x_pred = sess.run(X_pred, feed_dict=fdict)
    test_predictions[j] = x_pred[0]

# median test error
test_median = np.median(test_loss_history)
print('test median: {}'.format(test_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss_history, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use