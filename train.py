import os, code, sys, time, argparse
import numpy as np
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf
import nn
import utils

from shift_inv import get_symmetrized_idx, model_func_ShiftInv_symm, get_input_features_ShiftInv_numpy
#from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, NUM_VAL_SAMPLES, MODEL_BASENAME
#from utils import REDSHIFTS, AttrDict


""" TODO:

Would-be-nice:
 X done - have some other script handle the parse-args, instead of polluting here

"""
start_time = time.time()
#==============================================================================
# Data & Session config
#==============================================================================

# Arg parser
# ========================================
parser = utils.Parser()
args = parser.parse_args()
parser.print_args()
rs_idx = args.rs_idx = [10, 19] # MUST BE FIXED TO USE CACHED DATA
args.layer_type = 'shift-inv-symm'

# Dimensionality
# ========================================
num_redshifts = len(args.redshifts)
N = 32**3
M = args.graph_var = 14 # not used, but cached data was for M = 14
batch_size = 4 #args.batch_size # MUST BE FIXED TO USE CACHED DATA

# Training variables
# ========================================
learning_rate = args.learn_rate
checkpoint = 250 #args.checkpoint
num_iters  = args.num_iters
num_val_batches = 50 #args.num_test // batch_size
save_checkpoint = lambda step: (step+1) % checkpoint == 0





# Network config
# ========================================
network_features = utils.AttrDict()
network_features.var_scope = args.var_scope
network_features.num_layers = len(args.channels) - 1
network_features.dims = [batch_size, N, M]
network_features.activation = tf.nn.relu
loss_func = nn.pbc_loss # NOW SCALED 1e5 BY DEFAULT ! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#==============================================================================
# Setup computational graph
#==============================================================================

# Initialize model parameters
# ========================================
utils.initialize_model_params(args)


# Inputs
# ========================================
# Placeholders
in_shape = (None, 32**3, 6)
X_input = tf.placeholder(tf.float32, shape=in_shape)
X_truth = tf.placeholder(tf.float32, shape=in_shape)
#RS_in   = tf.placeholder(tf.float32, shape=(None, 1))
X_input_features = tf.placeholder(tf.float32, shape=(None, 9))

# Adjacency indices, symmetrized
# =======================================
row_in = tf.placeholder(tf.int32, shape=(None,))
col_in = tf.placeholder(tf.int32, shape=(None,))
all_in = tf.placeholder(tf.int32, shape=(None,))
tra_in = tf.placeholder(tf.int32, shape=(None,))
dia_in = tf.placeholder(tf.int32, shape=(None,))
dal_in = tf.placeholder(tf.int32, shape=(None,))

# Insert adj into dict
adj_symm_in = dict(row=row_in, col=col_in, all=all_in,
                   tra=tra_in, dia=dia_in, dal=dal_in)
#get_input_features_ShiftInv_numpy(X_in, A, N, redshift)
# Input args
# ShiftInv_layer_AdjTensor(H_in, adj, bN, layer_id, is_last=False):
model_args = (X_input, X_input_features, adj_symm_in, network_features)
rs = RS_in if args.cat_rs else None


# Outputs
# ========================================
#X_pred = nn.model_func_ShiftInv(*model_args, redshift=rs)
X_pred = model_func_ShiftInv_symm(*model_args, redshift=rs)


# Optimization
# ========================================
optimizer = tf.train.AdamOptimizer(learning_rate)
#loss_args = (X_pred,) + loss_args
error = loss_func(X_pred, X_truth)
train = optimizer.minimize(error)



# Initialize session and variables
# ========================================
# Session
gpu_frac = 0.85
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# Variables
sess.run(tf.global_variables_initializer())
train_saver = utils.TrainSaver(args)
if args.restore:
    train_saver.restore_model_parameters(sess)



#==============================================================================
# Load data
#==============================================================================

# Data load utils for cached data
# ========================================
CACHED_FEATURES_PATH  = './CachedData/Features/X_10-19_features_{}_batch.npy'
CACHED_ADJACENCY_PATH = './CachedData/Adjacency/X_10-19_adjacency_{}_batch.npy'
TRAINING_BATCHES_IDX = np.arange(201)

def load_cached_batch(batch_num):
    #==== Get file names
    feat_fname = CACHED_FEATURES_PATH.format(batch_num)
    adj_fname  = CACHED_ADJACENCY_PATH.format(batch_num)
    #==== Load files
    x_features  = np.load(feat_fname)
    x_adjacency = np.load(adj_fname)
    return x_features, x_adjacency

# Batch ground truth utils
# ========================================
# PREPROCESSING
def batch_ground_truth(X_in):
    """ Reshape ground truth into batches of size 4 to match cached data

    Params
    ------
    X_in : ndarray; shape (2, 1000, N, 6)
        Full ground truth of input and target

    Returns
    -------
    X_out : ndarray; shape (2, 250, 4, N, 6)
        batched ground truth
        X_out[:, :200] : training data (X_train, 200 batches)
        X_out[:, 200:] : test data (X_test, 50 batches)
    """
    num_batches = X_in.shape[1] // batch_size # 250
    assert num_batches == 250

    # Shape is FIXED to batch_size = 4
    X_out = np.zeros((2, num_batches, batch_size, N, 6), np.float32)

    # Batch
    for batch_num in range(num_batches):
        i, j = (batch_num * batch_size, (batch_num + 1) * batch_size)
        X_out[:, batch_num] = X_in[:, i:j]

    return X_out

# PROCESSED INPUT
def get_batch(X_in, eval_idx=None):
    """ Get a batch from the ground truth dataset and cached shift-inv data

    # Batching ground truth
    # ---------------------
    TRAIN: If training (eval_idx=None), then select a batch at random
    TEST:  If testing (eval_idx=<int>), then select that batch

    # Loading cached data
    # -------------------
    All cached data (features and adjacencies) filenames include their
    batch number, so we simply need a batch_number to load the proper data

    Params
    ------
    X_in : ndarray; (2, 250, 4, N, 6)
        The batched ground truth data
        X_in[:,:200] ---> Training set
        X_in[:,200:] --->  Testing set
    eval_idx : int
        if evaluation is defined, it is an int representing an index of one of
        the 50 test batches

    Returns
    -------
    X_out : ndarray; (2, 4, N, 6)
        ground truth batch
    X_features : ndarray; (S*, 9)
        features batch
    X_adjacency : list(ndarray); (6,)
        The symmetrical adjacency indices for shiftinv ops
            (row, col, all, tra, dia, dal)
    """
    #==== Training case: choose random
    if eval_idx is None:
        batch_num = np.random.choice(TRAINING_BATCHES_IDX) # num in [0...250]

    #==== Testing case: select batch from index alegbra
    else:
        assert isinstance(eval_idx, int) # input integrity check
        batch_num = 200 + eval_idx
        #print(f'In else clause for eval_idx: {eval_idx}\n  Batch_num = {batch_num}')

    # Get batch data
    X_out = np.copy(X_in[:, batch_num])
    X_features  = np.load(CACHED_FEATURES_PATH.format(batch_num))
    X_adjacency = np.load(CACHED_ADJACENCY_PATH.format(batch_num))
    return X_out, X_features, X_adjacency



# Load cubes
X = utils.normalize(utils.load_simulation_data(rs_idx))
#X = utils.load_simulation_data(rs_idx)
#X_train, X_test = utils.split_data_validation(X, num_val=args.num_test)
#X = None # reduce memory overhead

X = batch_ground_truth(X) # (2, 250, 4, N, 6)
assert X.shape == (2, 250, 4, N, 6)
# KEEP X AS FULL SHAPE, simpler..
#X_train = X[:, :200] # 200 batches (800 samples)
#X_test  = X[:, 200:] #  50 batches (200 samples)


# Get test containers
#test_pred_shape = X_test.shape[1:-1] + (args.channels[-1],)
#test_predictions  = np.zeros(test_pred_shape).astype(np.float32)
#test_loss = np.zeros((num_val_batches,)).astype(np.float32)
test_pred_shape = (200, N,) + (args.channels[-1],) # (200, N, 3)
test_predictions  = np.zeros(test_pred_shape).astype(np.float32)
test_loss = np.zeros((50,)).astype(np.float32)

# Save session data
#train_saver.save_model_cube(X_test, ground_truth=True)
train_saver.save_model_files()
train_saver.save_model_params(sess, 0)


#=============================================================================
# TRAINING
#=============================================================================
print('\nTraining:\n{}'.format('='*78))
np.random.seed(utils.DATASET_SEED)
for step in range(num_iters):
    # Data batching
    # ----------------
    #_x_batch = utils.next_minibatch(X_train, batch_size) # shape (2, b, N, 6)
    _x_batch, x_in_feats, symm_idx = get_batch(X) # (2, 4, N, 6)

    # split data
    x_in    = _x_batch[0] # (b, N, 6)
    x_truth = _x_batch[1] # (b, N, 6)


    # UNNECESSARY WITH CACHED DATA
    """
    # Graph data
    # ----------------
    csr_list = nn.get_graph_csr_list(x_in, args)
    #csr_list = get_graph_csr(x_in) # len b list of (N,N) csrs
    #coo_feats = nn.to_coo_batch(csr_list)


    x_in_feats = get_input_features_ShiftInv_numpy(np.copy(x_in),
                                                   csr_list,
                                                   N,
                                                   None)
    symm_idx = get_symmetrized_idx(csr_list)
    """
    fdict = {
        X_input : x_in,
        X_input_features: x_in_feats,
        X_truth: x_truth,
        row_in : symm_idx[0],
        col_in : symm_idx[1],
        all_in : symm_idx[2],
        tra_in : symm_idx[3],
        dia_in : symm_idx[4],
        dal_in : symm_idx[5],
    }

    # Train
    train.run(feed_dict=fdict)

    # Save
    if save_checkpoint(step):
        err = sess.run(error, feed_dict=fdict)
        utils.print_checkpoint(step, err)
        train_saver.save_model_params(sess, step)


# END training
# ========================================
print('elapsed time: {}'.format(time.time() - start_time))

# Save trained variables and session
train_saver.save_model_params(sess, num_iters)
#X_train = None # reduce memory overhead

eval_time = time.time()
#=============================================================================
# EVALUATION
#=============================================================================
print('\nEvaluation:\n{}'.format('='*78))
for j in range(num_val_batches): # ---> range(50) for b = 4
    # Validation cubes
    # ----------------
    p, q = batch_size*j, batch_size*(j+1)
    _x_test_batch, x_in_feats, symm_idx = get_batch(X, eval_idx=j)

    #x_in    = _x_test_batch[0, p:q]
    #x_truth = _x_test_batch[1, p:q]
    x_in = _x_test_batch[0]
    x_truth = _x_test_batch[1]


    """ # UNNECSSARY WITH CACHED DATA
    # Graph data
    # ----------------
    csr_list = nn.get_graph_csr_list(x_in, args)
    #csr_list = get_graph_csr(x_in) # len b list of (N,N) csrs
    #coo_feats = nn.to_coo_batch(csr_list)
    x_in_feats = get_input_features_ShiftInv_numpy(np.copy(x_in),
                                                   csr_list,
                                                   N,
                                                   None)
    symm_idx = get_symmetrized_idx(csr_list)
    """

    fdict = {
        X_input : x_in,
        X_input_features: x_in_feats,
        X_truth: x_truth,
        row_in : symm_idx[0],
        col_in : symm_idx[1],
        all_in : symm_idx[2],
        tra_in : symm_idx[3],
        dia_in : symm_idx[4],
        dal_in : symm_idx[5],
    }


    # Validation output
    # ----------------
    x_pred_val, v_error = sess.run([X_pred, error], feed_dict=fdict)
    test_predictions[p:q] = x_pred_val
    test_loss[j] = v_error
    print(f'val_err, {j} : {v_error}')



# END Validation
# ========================================
utils.print_median_validation_loss(rs_idx, test_loss)
train_saver.save_model_error(test_loss)
train_saver.save_model_cube(test_predictions, ground_truth=False)
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
print(f'elapsed eval time: {time.time() - eval_time}')
