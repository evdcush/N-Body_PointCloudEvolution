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
parser.add_argument('--particles', '-p', default=32,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--redshifts', '-z', default=[0.6, 0.0], nargs='+', type=float, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--model_type','-m', default=0,          type=int,  help='model type')
#parser.add_argument('--resume',    '-r', default=0,          type=int,  help='resume training from serialized params')
#parser.add_argument('--multi_step','-s', default=0,          type=int, help='use multi-step redshift model')
parser.add_argument('--num_iters', '-i', default=1000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=8,          type=int,  help='training batch size')
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
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=200)
X = None # reduce memory overhead

# velocity coefficient
vel_coeff = None
if pargs['vel_coeff']:
    vel_coeff = utils.load_velocity_coefficients(num_particles)[(zX, zY)]


#=============================================================================
# network and model params
#=============================================================================
# model vars
model_type = 1#pargs['model_type'] # 0: set, 1: graph, but only set implemented at moment
graph_model = True
#model_vars = utils.NBODY_MODELS[model_type]
channels   = utils.GRAPH_CHANNELS#model_vars['channels']
num_layers = len(channels) - 1

# hyperparameters
#init_params = model_vars['init_params']
init_params = utils.init_params_graph
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

# save test data
utils.save_test_cube(X_test, cube_path, (zX, zY), prediction=False)


#=============================================================================
# initialize graph
#=============================================================================
# init network params
init_params(channels)
K = 14

# direct graph
X_input = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_input')
X_truth = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_truth')
#adj_list = None
#if graph_model: # graph
adj_list = tf.placeholder(tf.int32, shape=[None, 2], name='adj_list')
X_pred = nn.network_graph_fwd(X_input, num_layers, adj_list, activation=tf.nn.relu, add=True, vel_coeff=None, K=14)
# network_fwd will dispatch on mtype_key
#X_pred  = nn.network_fwd(X_input, num_layers, adj_list, vel_coeff=vel_coeff, mtype_key=model_type, K=14)



# loss and optimizer
readout = nn.get_readout(X_pred)
loss    = nn.pbc_loss(readout, X_truth)
train   = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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
save_checkpoint = lambda step: step % 500 == 0 and step != 0
num_checkpoints_saved = 0

#=============================================================================
# TRAINING
#=============================================================================
start_time = time.time()
for step in range(num_iters):
    # data
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=True)
    x_in   = _x_batch[0]
    x_true = _x_batch[1]
    alist = nn.get_kneighbor_alist(x_in, K)

    if verbose:
        error = sess.run(loss, feed_dict={X_input: x_in, X_truth: x_true, adj_list: alist})
        train_loss_history[step] = error
        print('{}: {:.6f}'.format(step, error))

    # cycle through graph
    train.run(feed_dict={X_input: x_in, X_truth: x_true, adj_list: alist})
    if save_checkpoint(step):
        wr_meta = num_checkpoints_saved == 0
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=wr_meta)
        num_checkpoints_saved += 1

print('elapsed time: {}'.format(time.time() - start_time))
# elapsed time: 55.703558683395386 #  'error = sess.run()' on each iter: little over 10sec/run
# elapsed time: 41.57636308670044 # with no sess.run, under 10sec/run
# save
saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=False)
if verbose: utils.save_loss(loss_path + model_name, train_loss_history)
X_train = None # reduce memory overhead

#=============================================================================
# TESTING
#=============================================================================
# data containers
num_test_samples = X_test.shape[1]
test_predictions  = np.zeros((X_test.shape[1:-1] + (channels[-1],))).astype(np.float32)
test_loss_history = np.zeros((num_test_samples)).astype(np.float32)

print('\nTesting:\n==============================================================================')
for j in range(X_test.shape[1]):
    # data
    x_in   = X_test[0, j:j+1] # (1, n_P, 6)
    x_true = X_test[1, j:j+1]
    alist = nn.get_kneighbor_alist(x_in, K)

    # validation error
    error = sess.run(loss, feed_dict={X_input: x_in, X_truth: x_true, adj_list: alist})
    test_loss_history[j] = error
    print('{}: {:.6f}'.format(j, error))

    # prediction
    x_pred = sess.run(X_pred, feed_dict={X_input: x_in, adj_list: alist})
    test_predictions[j] = x_pred[0]

# median test error
test_median = np.median(test_loss_history)
print('test median: {}'.format(test_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss_history, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use