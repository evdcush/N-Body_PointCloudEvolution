import os, code, sys, time, argparse
import numpy as np
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf
import nn
import utils
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

# Dimensionality
# ========================================
num_redshifts = len(args.redshifts)
N = 32**3
M = args.graph_var
batch_size = args.batch_size

# Training variables
# ========================================
learning_rate = args.learn_rate
checkpoint = 100 #args.checkpoint
num_iters  = args.num_iters
num_val_batches = args.num_test // batch_size
save_checkpoint = lambda step: (step+1) % checkpoint == 0
get_graph_csr   = lambda h: nn.get_graph_csr_list(h, args)

# Network config
# ========================================
network_features = utils.AttrDict()
network_features.var_scope = args.var_scope
network_features.num_layers = len(args.channels) - 1
network_features.dims = [batch_size, N, M]
network_features.activation = tf.nn.relu
loss_func = nn.get_loss_func(args)


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
RS_in   = tf.placeholder(tf.float32, shape=(None, 1))

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

# Input args
# ShiftInv_layer_AdjTensor(H_in, adj, bN, layer_id, is_last=False):
model_args = (X_input, adj_symm_in, network_features)
loss_args = (X_truth, X_input) if multi_step else (X_truth,)
rs = RS_in if args.cat_rs else None


# Outputs
# ========================================
#X_pred = nn.model_func_ShiftInv(*model_args, redshift=rs)
X_pred = shift_inv.model_func_ShiftInv(*model_args, redshift=rs)


# Optimization
# ========================================
optimizer = tf.train.AdamOptimizer(learning_rate)
loss_args = (X_pred,) + loss_args
error = loss_func(*loss_args)
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
# Load cubes
rs_idx = args.rs_idx
#X = utils.normalize(utils.load_simulation_data(rs_idx))
X = utils.load_simulation_data(rs_idx)
X_train, X_test = utils.split_data_validation(X, num_val=args.num_test)
X = None # reduce memory overhead

# Get test containers
test_pred_shape = X_test.shape[1:-1] + (args.channels[-1],)
test_predictions  = np.zeros(test_pred_shape).astype(np.float32)
test_loss = np.zeros((num_val_batches,)).astype(np.float32)

# Save session data
train_saver.save_model_cube(X_test, ground_truth=True)
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
    _x_batch = utils.next_minibatch(X_train, batch_size) # shape (2, b, N, 6)

    # split data
    x_in    = _x_batch[0] # (b, N, 6)
    x_truth = _x_batch[1] # (b, N, 6)

    # Graph data
    # ----------------
    csr_list = get_graph_csr(x_in) # len b list of (N,N) csrs
    coo_feats = nn.to_coo_batch(csr_list)

    # Feed data to tensors
    # ----------------
    fdict = {X_input: x_in,
             X_truth: x_truth,
             COO_feats: coo_feats,
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
X_train = None # reduce memory overhead


#=============================================================================
# EVALUATION
#=============================================================================
print('\nEvaluation:\n{}'.format('='*78))
for j in range(num_val_batches):
    # Validation cubes
    # ----------------
    p, q = batch_size*j, batch_size*(j+1)
    x_in    = X_test[0, p:q]
    x_truth = X_test[1, p:q]

    # Graph data
    # ----------------
    csr_list = get_graph_csr(x_in) # len b list of (N,N) csrs
    coo_feats = nn.to_coo_batch(csr_list)

    # Feed data to tensors
    # ----------------
    fdict = {X_input: x_in,
             X_truth: x_truth,
             COO_feats: coo_feats,
             }

    # Validation output
    # ----------------
    x_pred_val, v_error = sess.run([X_pred, error], feed_dict=fdict)
    test_predictions[p:q] = x_pred_val
    test_loss[j] = v_error



# END Validation
# ========================================
utils.print_median_validation_loss(rs_idx, test_loss)
train_saver.save_model_error(test_loss)
train_saver.save_model_cube(test_predictions, ground_truth=False)
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
