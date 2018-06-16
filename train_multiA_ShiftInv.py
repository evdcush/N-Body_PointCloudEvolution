import os, code, sys, time, argparse
import numpy as np
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf
import nn
import utils
from utils import REDSHIFTS_ZUNI, PARAMS_SEED, LEARNING_RATE, RS_TAGS, NUM_VAL_SAMPLES

parser = argparse.ArgumentParser()
# argparse not handle bools well so 0,1 used instead
parser.add_argument('--redshifts', '-z', default=[18,19], nargs='+', type=int, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--particles', '-p', default=32,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--model_type','-m', default=1,          type=int,  help='model type')
parser.add_argument('--graph_var', '-k', default=14,         type=int, help='search parameter for graph model')
parser.add_argument('--restore',   '-r', default=0,          type=int,  help='resume training from serialized params')
parser.add_argument('--num_iters', '-i', default=1000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=8,          type=int,  help='training batch size')
parser.add_argument('--model_dir', '-d', default='./Model/', type=str,  help='directory where model parameters are saved')
parser.add_argument('--vcoeff',    '-c', default=0,          type=int,  help='use timestep coefficient on velocity')
parser.add_argument('--save_prefix','-n', default='',        type=str,  help='model name prefix')
parser.add_argument('--variable',   '-q', default=0.1,  type=float, help='multi-purpose variable argument')
pargs = vars(parser.parse_args())
start_time = time.time()


#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
#=============================================================================
# NBODY Data
#=============================================================================
# Nbody data specs
# ----------------
num_particles = pargs['particles'] # 32
N = num_particles**3
redshift_steps = pargs['redshifts']
redshifts = [REDSHIFTS_ZUNI[i] for i in redshift_steps]
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1
print('redshifts: {}'.format(redshifts))

# Load data
# ----------------
X = utils.load_zuni_npy_data(redshift_steps, norm_coo=True)[...,:-1]
#X = utils.load_rs_npy_data(redshift_steps, norm_coo=True, old_dataset=True)[...,:-1]
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=NUM_VAL_SAMPLES)
X = None # reduce memory overhead


#=============================================================================
# Model and network features
#=============================================================================
# Model features
# ----------------
model_type = pargs['model_type'] # 0: set, 1: graph
model_vars = utils.NBODY_MODELS[model_type]
use_coeff  = pargs['vcoeff'] == 1

# Network depth and channel sizes
# ----------------
#channels = model_vars['channels'] # OOM with sparse graph
channels = [6, 32, 16, 8, 3]
channels[0]  = 10
channels[-1] = 6
num_layers = len(channels) - 1
M = pargs['graph_var']

# Training hyperparameters
# ----------------
learning_rate = LEARNING_RATE # 0.01
#threshold = 0.03 # for PBC kneighbor search, currently not supported
batch_size = pargs['batch_size']
num_iters  = pargs['num_iters']


#=============================================================================
# Session save parameters
#=============================================================================
# Model name and paths
# ----------------
zX = redshift_steps[0]  # starting redshift
zY = redshift_steps[-1] # target redshift
model_name = utils.get_zuni_model_name(model_type, zX, zY, pargs['save_prefix'])
paths = utils.make_save_dirs(pargs['model_dir'], model_name)
model_path, loss_path, cube_path = paths

# restore
restore = pargs['restore'] == 1

# save test data
utils.save_test_cube(X_test, cube_path, (zX, zY), prediction=False)


#=============================================================================
# INITIALIZE model parameters and placeholders
#=============================================================================
# Init model params
# ----------------
vscope = utils.VAR_SCOPE.format(zX, zY)
tf.set_random_seed(utils.PARAMS_SEED)
utils.init_ShiftInv_params(channels, vscope, restore=restore, vcoeff=use_coeff)

# CUBE DATA
# ----------------
X_input = tf.placeholder(tf.float32, shape=(None, N, 6))
X_truth = tf.placeholder(tf.float32, shape=(None, N, 6))
RS_in = tf.placeholder(tf.float32,   shape=(None, 1))

# NEIGHBOR GRAPH DATA
# ----------------
# these shapes must be concrete for unsorted_segment_mean
m_coo_shape = (num_rs_layers, 3, batch_size*N*M,)
COO_seg_single = tf.placeholder(tf.int32, shape=(3, batch_size*N*M,))
COO_seg_multi  = tf.placeholder(tf.int32, shape=m_coo_shape)
COO_seg_val = tf.placeholder(tf.int32, shape=(3, N*M,))

# COEFFS
# ----------------
with tf.variable_scope(vscope):
    utils.init_coeff_multi(num_rs_layers)

Cidx = tf.placeholder(tf.int32)

#=============================================================================
# MODEL output and optimization
#=============================================================================
# helper for kneighbor search
def get_list_csr(h_in):
    return nn.get_kneighbor_list(h_in, M, inc_self=False, )#pbc=True)

# Model static func args
# ----------------
train_args = nn.ModelFuncArgs(num_layers, vscope, dims=[batch_size,N,M],)
val_args   = nn.ModelFuncArgs(num_layers, vscope, dims=[1,N,M], )


# Model outputs
# ----------------
# Train
X_pred_single = nn.ShiftInv_single_model_func(X_input, COO_seg_single,     RS_in, train_args)
X_pred_multi  = nn.ShiftInv_multi_model_func( X_input, COO_seg_multi,  redshifts, train_args, use_coeff=use_coeff)

# Validation
X_pred_val = nn.ShiftInv_single_model_func(X_input, COO_seg_val, RS_in, val_args, coeffs=None)


# Loss
# ----------------
# Optimizer
opt = tf.train.AdamOptimizer(learning_rate)

# Training error
s_error = nn.pbc_loss(X_pred_single, X_truth, vel=False)
m_error = nn.pbc_loss(X_pred_multi,  X_truth, vel=False)

# Backprop on loss
s_train = opt.minimize(s_error)
m_train = opt.minimize(m_error)

# Validation error
val_error   = nn.pbc_loss(X_pred_val, X_truth, vel=False)
inputs_diff = nn.pbc_loss(X_input,    X_truth, vel=False)


#=============================================================================
# Session setup
#=============================================================================
# Sess
# ----------------
gpu_frac = 0.9
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# initialize variables
sess.run(tf.global_variables_initializer())
if restore:
    utils.load_graph(sess, model_path)

#theta = utils.get_vcoeff(vscope).eval()
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

# Session saver
# ----------------
saver = tf.train.Saver()
saver.save(sess, model_path + model_name)
checkpoint = 500
save_checkpoint = lambda step: (step+1) % checkpoint == 0


#=============================================================================
# TRAINING
#=============================================================================
# SINGLE-STEP
# ----------------
rs_tups = [(i,i+1) for i in range(num_rs_layers)]
print('\nTraining Single-step:\n{}'.format('='*78))
np.random.seed(utils.DATASET_SEED)
for step in range(num_iters):
    # Data batching
    # ----------------
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=False) # shape (3, b, N, 6)

    # Train on redshift pairs
    # ----------------
    np.random.shuffle(rs_tups)
    for zx, zy in rs_tups:
        # redshift
        rs_in = np.full([batch_size*N*M, 1], redshifts[zx], dtype=np.float32)

        # split data
        x_in    = _x_batch[zx] # (b, N, 6)
        x_truth = _x_batch[zy] # (b, N, 6)

        # Graph data
        # ----------------
        csr_list = get_list_csr(x_in) # len b list of (N,N) csrs
        coo_segs = nn.to_coo_batch(csr_list)

        # Feed data to tensors
        # ----------------
        fdict = {X_input: x_in,
                 X_truth: x_truth,
                 COO_seg_single: coo_segs,
                 RS_in: rs_in,
                 Cidx: zx,
                 }

        # training pass
        s_train.run(feed_dict=fdict)

        # Checkpoint
        # ----------------
        # Track error
        #"""
        if (step + 1) % 5 == 0:
            e = sess.run(s_error, feed_dict=fdict)
            print('{:>5}: {}'.format(step+1, e))
        #"""

    # Save
    if save_checkpoint(step):
        tr_error = sess.run(s_error, feed_dict=fdict)
        print('checkpoint {:>5}: {}'.format(step, tr_error))
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)

# MULTI-STEP
# ----------------
def get_multi_coo(x):
    multi_coo = np.zeros((m_coo_shape)).astype(np.int32)
    for i in range(num_rs_layers):
        multi_coo[i] = nn.to_coo_batch(get_list_csr(x[i]))
    return multi_coo

print('\nTraining Multi-step:\n{}'.format('='*78))
#np.random.seed(utils.DATASET_SEED)
for step in range(num_iters):
    # Data batching
    # ----------------
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=False) # shape (3, b, N, 6)

    # split data
    x_in    = _x_batch[0]  # (b, N, 6)
    x_truth = _x_batch[-1] # (b, N, 6)

    # Graph data
    # ----------------
    coo_segs = get_multi_coo(_x_batch[:-1]) # (num_rs_layers, 3, c)

    # Feed data to tensors
    # ----------------
    fdict = {X_input: x_in,
             X_truth: x_truth,
             COO_seg_multi: coo_segs,
             }

    # training pass
    m_train.run(feed_dict=fdict)

    # Checkpoint
    # ----------------
    # Track error
    #"""
    if (step + 1) % 5 == 0:
        e = sess.run(m_error, feed_dict=fdict)
        print('{:>5}: {}'.format(step+1, e))
    #"""

    # Save
    if save_checkpoint(step):
        tr_error = sess.run(m_error, feed_dict=fdict)
        print('checkpoint {:>5}: {}'.format(step, tr_error))
        saver.save(sess, model_path + model_name, global_step=step+num_iters, write_meta_graph=True)



# END training
# ========================================
print('elapsed time: {}'.format(time.time() - start_time))

# Save trained variables and session
saver.save(sess, model_path + model_name, global_step=2*num_iters, write_meta_graph=True)
X_train = None # reduce memory overhead

'''
#=============================================================================
# EVALUATION
#=============================================================================
# Eval data containers
# ----------------
test_predictions  = np.zeros(X_test.shape[1:-1] + (channels[-1],)).astype(np.float32)
test_loss   = np.zeros((NUM_VAL_SAMPLES)).astype(np.float32)
inputs_loss = np.zeros((NUM_VAL_SAMPLES)).astype(np.float32)

print('\nEvaluation:\n{}'.format('='*78))
for j in range(X_test.shape[1]):
    # Validation cubes
    # ----------------
    x_in    = X_test[0, j:j+1] # (1, n_P, 6)
    x_truth = X_test[1, j:j+1]

    # Graph data
    # ----------------
    csr_list = get_list_csr(x_in) # len b list of (N,N) csrs

    # get edges (relative dists) and nodes (velocities)
    x_in_edges = nn.get_input_edge_features_batch(x_in, csr_list, M) # (b*N*M, 3)
    x_in_nodes = nn.get_input_node_features(x_in) # (b*N, 3)

    # get coo features
    COO_feats = nn.to_coo_batch(csr_list)

    # Feed data to tensors
    # ----------------
    fdict = {X_input: x_in,
             X_truth: x_truth,
             X_input_edges: x_in_edges,
             X_input_nodes: x_in_nodes,
             COO_features_val: COO_feats,}

    # Validation output
    # ----------------
    x_pred, v_error, ins_diff = sess.run([X_pred_val, val_error, inputs_diff], feed_dict=fdict)
    test_predictions[j] = x_pred[0]
    test_loss[j]   = v_error
    inputs_loss[j] = ins_diff
    #print('{:>3d}: {:.6f}'.format(j, v_error))

# END Validation
# ========================================
# median error
test_median = np.median(test_loss)
inputs_median = np.median(inputs_loss)
#print('{:<18} median: {:.9f}'.format(model_name, test_median))
print('{:<30} median: {:.9f}, {:.9f}'.format(model_name, test_median, inputs_median))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
'''
