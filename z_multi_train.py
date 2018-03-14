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
num_particles = 16 #pargs['particles']
#redshift_steps = pargs['redshifts']
#              0    1    2    3    4    5    6    7    8    9   10
#REDSHIFTS = [6.0, 4.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
redshift_steps = [1, 4, 7, 10]
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1

# Load data
num_val_samples = 200
X = utils.thinkpadx201_load_npy(redshift_steps, norm_coo=True)
#X = utils.load_rs_npy_data(redshift_steps, norm_coo=True, old_dataset=False)
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=num_val_samples)
X = None # reduce memory overhead


#=============================================================================
# network and model params
#=============================================================================
# model vars
model_type = pargs['model_type'] # 0: set, 1: graph
model_vars = utils.NBODY_MODELS[model_type]

# network kernel sizes and depth
channels = model_vars['channels']
channels[-1] = 6
channels[0] = 7
num_layers = len(channels) - 1

# model features
use_graph = model_type == 1
vcoeff = pargs['vel_coeff'] == 1

# hyperparameters
learning_rate = LEARNING_RATE # 0.01
K = pargs['knn'] # 14
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
utils.init_params(channels, graph_model=use_graph, var_scope=vscope, restore=restore)


# INPUTS
#data_shape = (None, num_particles**3, 6)
data_shape = (None, num_particles**3, 7)
X_input = tf.placeholder(tf.float32, shape=data_shape, name='X_input')
X_truth = tf.placeholder(tf.float32, shape=data_shape, name='X_truth')

# ADJACENCY LIST
alist_shape = (None, 2)
adj_list = tf.placeholder(tf.int32, shape=alist_shape, name='adj_list')

# loss scaling weights
#scale_weights = tf.placeholder(tf.float32, shape=(num_rs_layers,), name='scale_weights')

# scheduled sampling probs
#sampling_probs = tf.placeholder(tf.bool, shape=(num_rs_layers-1,), name='sampling_probs')

def alist_func(h_in): # for tf.py_func
    return nn.alist_to_indexlist(nn.get_pbc_kneighbors(h_in, K, threshold))
    #return nn.alist_to_indexlist(nn.get_kneighbor_alist(h_in, K))



rs_adj_shape = (None, None, 2) # output shape
#alist_shape = (None, 2)
rs_adj = tf.placeholder(tf.int32, shape=rs_adj_shape, name='rs_adj')

full_rs_shape = (None, None, num_particles**3, 7)
X_rs = tf.placeholder(tf.float32, shape=full_rs_shape, name='X_rs')
sampling_probs = tf.placeholder(tf.bool, shape=(num_rs,), name='sampling_probs') # only needs to be num_rs -1, but for convenience

def multi_model_fwd_sampling(x_in, sampling_probs, adj):
    """
    Args:
        x_in: (11, mb_size, ...) full rs data
    """
    rs_depth = x_in.get_shape().as_list()[0]
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    h = nn.get_readout_vel(nn.zuni_model_fwd(x_in[0], num_layers, adj[0], K, var_scope=vscope))
    if rs_depth is None: return h
    for i in range(1, rs_depth): # NONETYPE ERROR HERE, FIGURE OUT
        h_in = tf.where(sampling_probs[i], tf.concat((h, x_in[i,:,:,-1]), axis=-1), x_in[i])
        h = nn.get_readout_vel(nn.zuni_model_fwd(h_in, num_layers, adj[i], K, var_scope=vscope))
    return h

'''
def sampling_fwd_body():
    return False

def sampling_fwd(h, x_in, adj, sampling_prob):
    h_in = tf.where(sampling_prob, tf.concat((h, x_in[...,-1]), axis=-1), x_in)
    h = nn.get_readout_vel(nn.zuni_model_fwd(h_in, num_layers, adj, K, var_scope=vscope))
    return h
'''

X_pred_train = multi_model_fwd_sampling(X_rs, sampling_probs, rs_adj)


#=============================================================================
# Model predictions and optimizer
#=============================================================================
# model fwd dispatch args
if use_graph:
    margs = (X_input, num_layers, adj_list, K)
else:
    margs = (X_input, num_layers)

# network out
H_out  = nn.zuni_model_fwd(*margs, vel_coeff=vcoeff, var_scope=vscope)
X_pred = nn.get_readout_vel(H_out)

# error and optimizer
#error = nn.pbc_loss(X_pred, X_truth[...,:-1])
error = nn.pbc_loss(X_pred_train, X_rs[-1, :,:, :-1])
#error = nn.pbc_loss_vel(X_pred, X_truth[...,:-1])
train = tf.train.AdamOptimizer(learning_rate).minimize(error)
ground_truth_error = nn.pbc_loss(X_input[...,:-1], X_truth[...,:-1])

# evaluation error (multi-step)
Val_pred = tf.placeholder(tf.float32, shape=(None, num_particles**3, 6), name='Val_pred')
multi_error = nn.pbc_loss(Val_pred, X_truth[...,:-1])


#=============================================================================
# Session and Train setup
#=============================================================================
# training params
batch_size = pargs['batch_size']
num_iters  = pargs['num_iters']
verbose    = pargs['verbose']

# Sess
#gpu_frac = 0.8
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
if restore:
    utils.load_graph(sess, model_path)
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
# Save
train_loss_history = np.zeros((num_iters)).astype(np.float32)
saver = tf.train.Saver()
saver.save(sess, model_path + model_name)
checkpoint = 500
save_checkpoint = lambda step: step % checkpoint == 0 and step != 0

#=============================================================================
# TRAINING
#=============================================================================
print('\nTraining:\n==============================================================================')
start_time = time.time()
rs_tups = [(i, i+1) for i in range(num_rs_layers)]
np.random.seed(utils.DATASET_SEED)
time_per_depth = np.zeros((num_iters, num_rs)).astype(np.float32)
# START
for step in range(num_iters):
    # data batching
    #print('STEP: {}'.format(step))
    _x_batch = utils.next_zuni_minibatch(X_train, batch_size, data_aug=True)
    '''
    np.random.shuffle(rs_tups)
    for idx_in, idx_out in rs_tups:
        # data inputs
        x_in    = _x_batch[idx_in]
        x_truth = _x_batch[idx_out]
        fdict = {X_input: x_in, X_truth: x_truth}

        # feed graph model data
        if use_graph:
            alist = alist_func(x_in)
            fdict[adj_list] = alist

        # training pass
        train.run(feed_dict=fdict)
    '''
    samp_probs = np.random.rand(num_rs) <= step / float(num_iters)
    samp_probs[0] = True # first input always ground truth
    pred_depth = np.sum(samp_probs)

    # feed data
    pdepths = np.arange(1, num_rs)
    np.random.shuffle(pdepths)
    for pred_depth in pdepths:
        time_in = time.time()
        x_in = _x_batch[:pred_depth+1]
        adj_lists = np.array([alist_func(x_in[i]) for i in range(pred_depth)])
        fdict = {X_rs: x_in, rs_adj:adj_lists, sampling_probs:samp_probs}
        train.run(feed_dict=fdict)
        time_elapsed = time.time() - time_in
        time_per_depth[step, pred_depth] = time_elapsed
        print('pred_depth: {}, time: {}'.format(pred_depth, time_elapsed))


    # save checkpoint
    if save_checkpoint(step):
        #tr_error = sess.run(error, feed_dict=fdict)
        #print('checkpoint {:>5}: {}'.format(step, tr_error))
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)
# END
print('elapsed time: {}'.format(time.time() - start_time))
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
def print_depth_times(times):
    meds = np.median(times[:,1:], axis=0)
    print('TIME PER DEPTH:')
    print(meds)

print_depth_times(time_per_depth)

# save
saver.save(sess, model_path + model_name, global_step=num_iters, write_meta_graph=True)
#if verbose: utils.save_loss(loss_path + model_name, train_loss_history)
X_train = None # reduce memory overhead
"""
#=============================================================================
# EVALUATION
#=============================================================================
print('\nEvaluation:\n==============================================================================')
# data containers
num_test_samples = X_test.shape[1]
#test_predictions  = np.zeros(X_test.shape[1:]).astype(np.float32)
test_predictions  = np.zeros(X_test.shape[1:-1] + (6,)).astype(np.float32)
test_loss_history = np.zeros((num_test_samples)).astype(np.float32)

rs_tups = [(i, i+1) for i in range(num_rs_layers)] # DO NOT SHUFFLE!

for j in range(X_test.shape[1]):
    # first pass
    x_in    = X_test[0, j:j+1] # (1, n_P, 6)
    z_next  = X_test[1, j:j+1, :, -1:] # redshifts
    fdict = {X_input: x_in}
    if use_graph:
        alist = alist_func(x_in)
        fdict[adj_list] = alist
    x_pred = sess.run(X_pred, feed_dict=fdict)

    # subsequent pass receive previous prediction
    for i in range(1, num_rs_layers):
        x_in = np.concatenate((x_pred, z_next), axis=-1) #(...,6) -> (...,7)
        z_next = X_test[i+1, j:j+1, :, -1:]
        fdict = {X_input: x_in}
        if use_graph:
            alist = alist_func(x_in)
            fdict[adj_list] = alist
        x_pred = sess.run(X_pred, feed_dict=fdict)

    # save prediction
    test_predictions[j] = x_pred[0]

    # feeding for multi-step loss info
    x_in   = X_test[-2, j:j+1]
    x_true = X_test[-1, j:j+1]
    fdict = {Val_pred: x_pred, X_input: x_in, X_truth: x_true}

    # loss data
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    v_error, truth_error = sess.run([multi_error, ground_truth_error], feed_dict=fdict)
    test_loss_history[j] = v_error
    '''
    Ideally, the multi_error would be, at the very least, better than
    the ground_truth_error.
    '''
    print('{:>3d}: {:.6f} | {:.6f}'.format(j, v_error, truth_error))


# median test error
test_median = np.median(test_loss_history)
print('test median: {}'.format(test_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss_history, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)
"""
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use