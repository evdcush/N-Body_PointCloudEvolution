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
parser.add_argument('--graph_var', '-k', default=0.03,          type=float, help='number of nearest neighbors for graph model')
parser.add_argument('--restore_single', '-rs', default=0,          type=int,  help='resume training from serialized params')
parser.add_argument('--restore_agg',    '-ra', default=0,          type=int,  help='resume training from serialized params')
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
rs_tups = [(redshift_steps[i], redshift_steps[i+1]) for i in range(num_rs_layers)]

# Load data
num_val_samples = 200
X = utils.load_zuni_npy_data(redshifts=redshift_steps, norm_coo=True)
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=num_val_samples)
X = None # reduce memory overhead


#=============================================================================
# network and model params
#=============================================================================
# model vars
model_type = pargs['model_type'] # 0: set, 1: graph
model_vars = utils.NBODY_MODELS[model_type]

# network kernel sizes and depth
rad_channels = [6, 32, 16, 8, 3]
channels = rad_channels #model_vars['channels']
channels[-1] = 6
channels[0] = 7
num_layers = len(channels) - 1

# model features
use_graph = model_type == 1
vcoeff = pargs['vel_coeff'] == 1

# hyperparameters
learning_rate = LEARNING_RATE # 0.01
G = pargs['graph_var'] # 14
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
#restore = pargs['restore'] == 1
restore_agg = pargs['restore_agg'] == 1
restore_single = pargs['restore_single'] == 1
restore = restore_agg or restore_single

# save test data
#utils.save_test_cube(X_test, cube_path, (zX, zY), prediction=False)

init_use_graph = False # disgusting, dirty fix. do it right
#=============================================================================
# initialize graph and placeholders
#=============================================================================
# init network params
#vscope = utils.VAR_SCOPE_SINGLE_MULTI.format(zX, zY)
tf.set_random_seed(utils.PARAMS_SEED)
vscopes = [utils.VAR_SCOPE_SINGLE_MULTI.format(tup[0], tup[1]) for tup in rs_tups]
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
#z_idx = -3
#utils.init_zuni_params_multi(channels, vscopes[z_idx:], graph_model=use_graph, restore=True, vel_coeff=vcoeff)
#utils.init_zuni_params_multi(channels, vscopes[0:z_idx], graph_model=use_graph, restore=False, vel_coeff=vcoeff)
utils.init_zuni_params_multi(channels, vscopes, graph_model=init_use_graph, restore=restore, vel_coeff=vcoeff)


# INPUTS
#data_shape = (None, num_particles**3, 6)
data_shape = (None, num_particles**3, 7)
X_rs = tf.placeholder(tf.float32, shape=(num_rs,) + data_shape)

# ADJACENCY LIST
#alist_shape = (None, 2)
#rs_adj_list = tf.placeholder(tf.int32, shape=(num_rs_layers,) + alist_shape)
#graph_in = tf.sparse_placeholder(tf.float32, shape=(num_rs_layers, None, None)) # indexing no-no for sparse
graph_in = [tf.sparse_placeholder(tf.float32) for _ in range(num_rs_layers)]

def graph_func(h_in): # for tf.py_func
    #return nn.alist_to_indexlist(nn.get_pbc_kneighbors(h_in, K, threshold))
    #return nn.alist_to_indexlist(nn.get_kneighbor_alist(h_in, K))
    return nn.get_radius_graph_input(h_in, G)

''' # Cannot use tf.py_func to wrap function that returns attributes for sparse tensor
def graph_to_tensor_func(h_in):
    return tf.SparseTensor(nn.get_radius_graph_input(h_in, G))
'''

#=============================================================================
# Model predictions and optimizer
#=============================================================================
# ==== Network outputs
 # Train
X_pred, error = nn.aggregate_multiStep_fwd(X_rs, num_layers, vscopes, graph_in, vel_coeff=vcoeff)

 # Validation
val_x_in    = tf.placeholder(tf.float32, shape=data_shape)
val_x_pred  = tf.placeholder(tf.float32, shape=data_shape[:-1] + (6,))
val_x_truth = tf.placeholder(tf.float32, shape=data_shape)

val_g_in = tf.sparse_placeholder(tf.float32)

def val_func(i):
    h_out = nn.model_fwd(val_x_in, num_layers, val_g_in, vel_coeff=vcoeff, var_scope=vscopes[i])
    val_pred = nn.get_readout(h_out)
    return val_pred

#X_pred_val = val_func()
val_error = nn.pbc_loss(val_x_pred, val_x_truth[...,:-1])

# ==== Error and optimizer
 # Training
#error = nn.pbc_loss(X_pred, X_rs[-1,:,:,:-1])
#error = nn.pbc_loss_vel(X_pred, X_rs[-1,:,:,:-1])
train = tf.train.AdamOptimizer(learning_rate, name='AdamMulti').minimize(error)

 # Validation
#val_error = nn.pbc_loss(X_pred_val, X_rs[-1,:,:,:-1]) # since training loss fn not always same
#val_error = nn.pbc_loss(X_pred_val[-1], X_rs[-1,:,:,:-1]) # since training loss fn not always same
#ground_truth_error = nn.pbc_loss(X_rs[-2,:,:,:-1], X_rs[-1,:,:,:-1])


#=============================================================================
# Session and Train setup
#=============================================================================
# training params
batch_size = pargs['batch_size']
num_iters  = pargs['num_iters']
verbose    = pargs['verbose']

# Sess
gpu_frac = 0.95
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

# RESTORE
 # restore individually trained params for new aggregate model
if restore_single: #
    mp = './Model/Rad_short_coo_ZG_{}-{}/Session/'
    print('restore from: {}'.format(mp))
    mpaths = [mp.format(tup[0], tup[1]) for tup in rs_tups]
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    utils.load_multi_graph(sess, vscopes, num_layers, mpaths, use_graph=init_use_graph, vel_coeff=vcoeff)
 # restore previously trained aggregate model
elif restore_agg:
    utils.load_graph(sess, model_path)


#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
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
for step in range(num_iters):
    _x_batch = utils.next_zuni_minibatch(X_train, batch_size, data_aug=True)

    # Feed data
    x_in = _x_batch

    # feed graph data
    #adj_lists = np.array([alist_func(x_in[i]) for i in range(num_rs_layers)])
    graph_list = [graph_func(x_in[i]) for i in range(num_rs_layers)]
    fdict = {tensor: sparse_attribs for tensor, sparse_attribs in zip(graph_in, graph_list)}
    fdict[X_rs] = x_in

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
#test_predictions  = np.zeros(X_test.shape[1:-1] + (6,)).astype(np.float32)
test_predictions  = np.zeros((num_rs_layers,) + X_test.shape[1:-1] + (6,)).astype(np.float32)
test_loss_history = np.zeros((num_test_samples)).astype(np.float32)
#inputs_loss_history = np.zeros((num_test_samples)).astype(np.float32)

print('\nEvaluation:\n==============================================================================')
for j in range(X_test.shape[1]):

    # validation inputs
    x_in   = X_test[0, j:j+1] # (1, n_P, 6)
    x_true = X_test[-1,j:j+1]

    # validation outputs
    '''
    vals = sess.run([X_pred_val, val_error, ground_truth_error], feed_dict=fdict)
    x_pred, v_error, truth_error = vals
    for idx, x in enumerate(x_pred):
        test_predictions[idx,j] = x[0]
    test_loss_history[j] = v_error
    inputs_loss_history[j] = truth_error
    #test_predictions[j] = x_pred[0]
    '''
    '''
    def concat_rs(h, i):
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        cat = np.concatenate((h, X_test[i, j:j+1, :, -1:]), axis=-1)
        return cat
    '''
    concat_rs = lambda h, i: np.concatenate((h, X_test[i, j:j+1, :, -1:]), axis=-1)

    val_graph = graph_func(x_in)
    x_pred = sess.run(val_func(0), feed_dict={val_x_in: x_in, val_g_in: val_graph})
    test_predictions[0, j] = x_pred[0]

    for z in range(1, num_rs_layers):
        #print('Val scope: {}'.format(vscopes[z]))
        x_in = concat_rs(x_pred, z)
        val_graph = graph_func(x_in)
        x_pred = sess.run(val_func(z), feed_dict={val_x_in: x_in, val_g_in: val_graph})
        test_predictions[z, j] = x_pred[0]

    v_error = sess.run(val_error, feed_dict={val_x_pred: x_pred, val_x_truth: x_true})
    test_loss_history[j] = v_error

    print('{:>3d}: {:.6f}'.format(j, v_error))




    #print('{:>3d}: {:.6f} | {:.6f}'.format(j, v_error, truth_error))

# median test error
test_median = np.median(test_loss_history)
#inputs_median = np.median(inputs_loss_history)
print('test median: {}'.format(test_median))
#print('test median: {}, input median: {}'.format(test_median, inputs_median))
#print('{:<12} median: {}, {}'.format(model_name, test_median, inputs_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss_history, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
