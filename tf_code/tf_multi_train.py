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
parser.add_argument('--knn',       '-k', default=14,          type=int, help='number of nearest neighbors for graph model')
#parser.add_argument('--resume',    '-r', default=0,          type=int,  help='resume training from serialized params')
#parser.add_argument('--multi_step','-s', default=0,          type=int, help='use multi-step redshift model')
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
zX, zY = pargs['redshifts']
nbody_params = (num_particles, (zX, zY))
num_rs = len(utils.REDSHIFTS)
num_rs_layers = num_rs - 1

# Load data
X = utils.load_npy_data(num_particles, normalize=True)
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
channels   = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 6]#model_vars['channels']
num_layers = len(channels) - 1
#print('model_type: {}\nuse_graph: {}\nchannels:{}'.format(model_type, use_graph, channels))

# hyperparameters
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
#utils.save_test_cube(X_test, cube_path, (zX, zY), prediction=False)


#=============================================================================
# initialize graph
#=============================================================================
# init network params
#tf.set_random_seed(utils.PARAMS_SEED)
utils.init_params_multi(channels, num_rs_layers, graph_model=use_graph, seed=utils.PARAMS_SEED)



'''
It would not be necessary to have all these placeholders if you were able to
use tf ops for neighbor search instead of sklearn
'''
# direct graph
X60 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X60')
X40 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X40')
X20 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X20')
X15 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X15')
X12 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X12')
X10 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X10')
X08 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X08')
X06 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X06')
X04 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X04')
X02 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X02')
X00 = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X00')
'''
X_input = tf.placeholder(tf.float32, shape=[num_rs, None, num_particles**3, 6], name='X_input')
X60 = X_input[0]
X40 = X_input[1]
X20 = X_input[2]
X15 = X_input[3]
X12 = X_input[4]
X10 = X_input[5]
X08 = X_input[6]
X06 = X_input[7]
X04 = X_input[8]
X02 = X_input[9]
X00 = X_input[10]
'''

X60_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X60_alist')
X40_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X40_alist')
X20_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X20_alist')
X15_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X15_alist')
X12_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X12_alist')
X10_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X10_alist')
X08_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X08_alist')
X06_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X06_alist')
X04_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X04_alist')
X02_alist = tf.placeholder(tf.int32, shape=[None, 2], name='X02_alist')
K = 14
X40_hat = nn.get_readout_vel(nn.model_fwd(X60,     num_layers, X60_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(0)))
X20_hat = nn.get_readout_vel(nn.model_fwd(X40_hat, num_layers, X40_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(1)))
X15_hat = nn.get_readout_vel(nn.model_fwd(X20_hat, num_layers, X20_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(2)))
X12_hat = nn.get_readout_vel(nn.model_fwd(X15_hat, num_layers, X15_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(3)))
X10_hat = nn.get_readout_vel(nn.model_fwd(X12_hat, num_layers, X12_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(4)))
X08_hat = nn.get_readout_vel(nn.model_fwd(X10_hat, num_layers, X10_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(5)))
X06_hat = nn.get_readout_vel(nn.model_fwd(X08_hat, num_layers, X08_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(6)))
X04_hat = nn.get_readout_vel(nn.model_fwd(X06_hat, num_layers, X06_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(7)))
X02_hat = nn.get_readout_vel(nn.model_fwd(X04_hat, num_layers, X04_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(8)))
X00_hat = nn.get_readout_vel(nn.model_fwd(X02_hat, num_layers, X02_alist, K, var_scope=utils.VAR_SCOPE_MULTI.format(9)))


# losses
loss = nn.pbc_loss(X40_hat, X40)
loss += nn.pbc_loss(X20_hat, X20)
loss += nn.pbc_loss(X15_hat, X15)
loss += nn.pbc_loss(X12_hat, X12)
loss += nn.pbc_loss(X10_hat, X10)
loss += nn.pbc_loss(X08_hat, X08)
loss += nn.pbc_loss(X06_hat, X06)
loss += nn.pbc_loss(X04_hat, X04)
loss += nn.pbc_loss(X02_hat, X02)
loss += nn.pbc_loss(X00_hat, X00)


# loss and optimizer
#readout = nn.get_readout(X_pred)
#loss    = nn.pbc_loss(readout, X_truth)
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
checkpoint = 500
save_checkpoint = lambda step: step % checkpoint == 0 and step != 0

#=============================================================================
# TRAINING
#=============================================================================
start_time = time.time()
for step in range(num_iters):
    # data
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=True)
    x_in   = _x_batch
    x60 = x_in[0]


    #using non pbc neighbor for now
    nn_fn = nn.get_kneighbor_alist
    '''
    I'm almost certain you can use lists of placeholders, but I guess lets just
    see if this would even work for now
    The current training routine with the progressive additions of adj lists
    literally sounds bad. My machine makes bad noises, sounds like a seeking
    hard disk (ttttttTa tttttttTa ...), which is suspiciously similar to this
    routine.
    WHAT IS THE SOLUTION?
    is there any way around this ridiculous recursive alist thing?
    - we can use lists, which would clean up the hardcode, but that's not a solution
      The real problem is that we cannot pass tensors to the nearest neighbor search
    - originally tried to .eval() on the layer preds within the network fwd
      function, but the problem was that placeholders were being eval'd and
      sent to kneighbor search when it needs real data
    '''

    # pred
    alist60 = nn.alist_to_indexlist(nn_fn(x60, K))
    fdict   = {X60: x60, X60_alist: alist60}
    #=====================================
    x40_hat = sess.run(X40_hat, feed_dict=fdict)
    fdict[X40_alist] = nn.alist_to_indexlist(nn_fn(x40_hat, K))
    #=====================================
    x20_hat = sess.run(X20_hat, feed_dict=fdict)
    fdict[X20_alist] = nn.alist_to_indexlist(nn_fn(x20_hat, K))
    #=====================================
    x15_hat = sess.run(X15_hat, feed_dict=fdict)
    fdict[X15_alist] = nn.alist_to_indexlist(nn_fn(x15_hat, K))
    #=====================================
    x12_hat = sess.run(X12_hat, feed_dict=fdict)
    fdict[X12_alist] = nn.alist_to_indexlist(nn_fn(x12_hat, K))
    #=====================================
    x10_hat = sess.run(X10_hat, feed_dict=fdict)
    fdict[X10_alist] = nn.alist_to_indexlist(nn_fn(x10_hat, K))
    #=====================================
    x08_hat = sess.run(X08_hat, feed_dict=fdict)
    fdict[X08_alist] = nn.alist_to_indexlist(nn_fn(x08_hat, K))
    #=====================================
    x06_hat = sess.run(X06_hat, feed_dict=fdict)
    fdict[X06_alist] = nn.alist_to_indexlist(nn_fn(x06_hat, K))
    #=====================================
    x04_hat = sess.run(X04_hat, feed_dict=fdict)
    fdict[X04_alist] = nn.alist_to_indexlist(nn_fn(x04_hat, K))
    #=====================================
    x02_hat = sess.run(X02_hat, feed_dict=fdict)
    fdict[X02_alist] = nn.alist_to_indexlist(nn_fn(x02_hat, K))
    #=====================================
    #x00_hat = sess.run(X00_hat, feed_dict=fdict)
    fdict[X40] = x_in[1]
    fdict[X20] = x_in[2]
    fdict[X15] = x_in[3]
    fdict[X12] = x_in[4]
    fdict[X10] = x_in[5]
    fdict[X08] = x_in[6]
    fdict[X06] = x_in[7]
    fdict[X04] = x_in[8]
    fdict[X02] = x_in[9]
    fdict[X00] = x_in[10]

    if verbose:
        error = sess.run(loss, feed_dict=fdict)
        train_loss_history[step] = error
        print('{}: {:.8f}'.format(step, error))

    # cycle through graph
    train.run(feed_dict=fdict)
    if save_checkpoint(step):
        error = sess.run(loss, feed_dict=fdict)
        print('checkpoint {:>5}: {}'.format(step, error))
        wr_meta = step == checkpoint # only write on first checkpoint
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=wr_meta)

print('elapsed time: {}'.format(time.time() - start_time))

# save
saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=False)
if verbose: utils.save_loss(loss_path + model_name, train_loss_history)
X_train = None # reduce memory overhead

#=============================================================================
# TESTING
#=============================================================================
'''
# data containers
num_test_samples = X_test.shape[1]
test_predictions  = np.zeros((X_test.shape[1:-1] + (channels[-1],))).astype(np.float32)
test_loss_history = np.zeros((num_test_samples)).astype(np.float32)

print('\nTesting:\n==============================================================================')
for j in range(X_test.shape[1]):
    # data
    x_in   = X_test[0, j:j+1] # (1, n_P, 6)
    x_true = X_test[1, j:j+1]
    fdict = {X_input: x_in, X_truth: x_true}
    if use_graph:
        neighbors = nn.get_pbc_kneighbors(x_in, K, boundary_threshold)
        alist = nn.alist_to_indexlist(neighbors)
        fdict[adj_list] = alist

    # validation error
    error = sess.run(loss, feed_dict=fdict)
    test_loss_history[j] = error
    print('{}: {:.6f}'.format(j, error))

    # prediction
    x_pred = sess.run(readout, feed_dict=fdict)
    test_predictions[j] = x_pred[0]

# median test error
test_median = np.median(test_loss_history)
print('test median: {}'.format(test_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss_history, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)
'''
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use