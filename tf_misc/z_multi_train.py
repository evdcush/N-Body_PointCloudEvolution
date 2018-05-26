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
redshift_steps = pargs['redshifts']
#              0    1    2    3    4    5    6    7    8    9   10
#REDSHIFTS = [6.0, 4.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0] # old
#redshift_steps = [0, 3, 5, 7, 10]
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1

# Load data
num_val_samples = 200
#X = utils.load_rs_npy_data(redshift_steps, norm_coo=True, old_dataset=False)
X = utils.load_zuni_npy_data_eqvar(redshifts=redshift_steps, norm_coo=True, rs=True)
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=num_val_samples)
X = None # reduce memory overhead


#=============================================================================
# network and model params
#=============================================================================
# model vars
model_type = pargs['model_type'] # 0: set, 1: graph
model_vars = utils.NBODY_MODELS[model_type]

# network kernel sizes and depth
#channels = model_vars['channels']
#channels[-1] = 6
#channels[0] = 7
channels = [6, 8, 16, 4, 3]
channels[-1] = 2
channels[0] = 3


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
utils.save_test_cube(X_test, cube_path, (zX, zY), prediction=False)


#=============================================================================
# initialize graph and placeholders
#=============================================================================
# init network params
vscope = utils.VAR_SCOPE_SINGLE_MULTI.format(zX, zY)
tf.set_random_seed(utils.PARAMS_SEED)
#utils.init_params(channels, graph_model=use_graph, var_scope=vscope, restore=restore)
utils.init_eqvar_params(channels, var_scope=vscope, vel_coeff=vcoeff, restore=restore)


# INPUTS
#data_shape = (None, num_particles**3, 6)
#data_shape = (None, num_particles**3, 7)
data_shape = (None, None, num_particles**3, 3, 3)
X_input = tf.placeholder(tf.float32, shape=data_shape, name='X_input')
#X_truth = tf.placeholder(tf.float32, shape=data_shape, name='X_truth')

# ADJACENCY LIST
#alist_shape = (None, 2)
adj_list = tf.placeholder(tf.int32, shape=(None, 2), name='adj_list')
adj_list2 = tf.placeholder(tf.int32, shape=(num_rs_layers, None, 2), name='adj_list2')

def alist_func(h_in): # for tf.py_func
    #return nn.alist_to_indexlist(nn.get_pbc_kneighbors(h_in, K, threshold))
    return nn.alist_to_indexlist(nn.get_kneighbor_alist_eqvar(h_in, K))


#=============================================================================
# Model predictions and optimizer
#=============================================================================
# model fwd dispatch args
if use_graph:
    margs = (X_input[0], num_layers, adj_list, K)
    multi_args_val = (X_input, num_rs_layers, num_layers, alist_func, K)
    multi_args = (X_input, num_rs_layers, num_layers, adj_list2, K)
else:
    margs = (X_input[0], num_layers)
    multi_args_val = (X_input, num_rs_layers, num_layers)
    multi_args = multi_args_val

# network out
#H_out  = nn.zuni_model_fwd(*margs, vel_coeff=vcoeff, var_scope=vscope)
#X_pred = nn.get_readout(H_out)
H_out  = nn.eqvar_model_fwd(*margs, vel_coeff=vcoeff, var_scope=vscope)
X_pred = nn.get_readout_eqvar(H_out)
X_pred2 = nn.multi_eqvar_model_fwd(*multi_args, var_scope=vscope)
X_pred_val = nn.multi_eqvar_model_fwd_val(*multi_args_val, var_scope=vscope)

# training error
error1 = nn.pbc_loss_eqvar(X_pred,  X_input[-1])
error2 = nn.pbc_loss_eqvar(X_pred2, X_input[-1])
error3 = nn.pbc_loss_eqvar(X_pred_val, X_input[-1])

# optimizer
opt = tf.train.AdamOptimizer(learning_rate)
train1 = opt.minimize(error1)
train2 = opt.minimize(error2)


#
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
#sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
if restore:
    utils.load_graph(sess, model_path)

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
# START
for step in range(num_iters):
    # data batching
    #print('STEP: {}'.format(step))
    _x_batch = utils.next_zuni_minibatch(X_train, batch_size, data_aug=False)
    #_x_batch = _x_batch[...,:-1]
    np.random.shuffle(rs_tups)
    for tup in rs_tups:
        # data inputs
        x_in    = _x_batch[[tup]]
        fdict = {X_input: x_in}

        # feed graph model data
        if use_graph:
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            alist = alist_func(x_in[0])
            fdict[adj_list] = alist

        # training pass
        train1.run(feed_dict=fdict)

    # save checkpoint
    if save_checkpoint(step):
        tr_error = sess.run(error1, feed_dict=fdict)
        print('checkpoint {:>5}: {}'.format(step, tr_error))
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)
print('FINISHED SINGLE-STEP TRAINING')


for step in range(num_iters):
    #print('SCHED STEP: {}'.format(step))

    # data batching: (num_rs, mb_size, N, 7)
    _x_batch = utils.next_zuni_minibatch(X_train, batch_size, data_aug=False)
    #_x_batch = _x_batch[...,:-1]

    # feed data
    x_in = _x_batch
    fdict = {X_input: x_in}

    if use_graph:
        adj_lists = np.array([alist_func(x_in[i]) for i in range(num_rs_layers)])
        fdict[adj_list2] = adj_lists

    train2.run(feed_dict=fdict)

    # save checkpoint
    if save_checkpoint(step):
        tr_error = sess.run(error2, feed_dict=fdict)
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
print('\nEvaluation:\n==============================================================================')
# data containers
num_test_samples = X_test.shape[1]
#test_predictions  = np.zeros((num_rs_layers,) + X_test.shape[1:-1] + (6,)).astype(np.float32)
test_predictions  = np.zeros(X_test.shape[1:-1] + (2,)).astype(np.float32)
test_loss_history = np.zeros((num_test_samples)).astype(np.float32)
inputs_loss_history = np.zeros((num_test_samples)).astype(np.float32)

'''
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
    test_predictions[0, j] = x_pred[0]

    # subsequent pass receive previous prediction
    for i in range(1, num_rs_layers):
        x_in = np.concatenate((x_pred, z_next), axis=-1) #(...,6) -> (...,7)
        z_next = X_test[i+1, j:j+1, :, -1:]
        fdict = {X_input: x_in}
        if use_graph:
            alist = alist_func(x_in)
            fdict[adj_list] = alist
        x_pred = sess.run(X_pred, feed_dict=fdict)
        test_predictions[i, j] = x_pred[0]

    # save prediction
    #test_predictions[j] = x_pred[0]

    # feeding for multi-step loss info
    x_in   = X_test[-2, j:j+1]
    x_true = X_test[-1, j:j+1]
    fdict = {X_pred_val: x_pred, X_input: x_in, X_truth: x_true}

    # loss data
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    v_error, truth_error = sess.run([val_error, ground_truth_error], feed_dict=fdict)
    test_loss_history[j] = v_error
    inputs_loss_history[j] = truth_error
    print('{:>3d}: {:.6f} | {:.6f}'.format(j, v_error, truth_error))
'''
for j in range(X_test.shape[1]):
    # validation inputs
    x_in   = X_test[:, j:j+1] # (1, n_P, 6)
    #x_true = X_test[1, j:j+1]
    fdict = {X_input: x_in}

    x_pred, v_error = sess.run([X_pred_val, error3], feed_dict=fdict)
    test_loss_history[j] = v_error
    test_predictions[j] = x_pred[0]



# median test error
test_median = np.median(test_loss_history)
#inputs_median = np.median(inputs_loss_history)
#print('test median: {}, input median: {}'.format(test_median, inputs_median))
print('test median: {}'.format(test_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss_history, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

