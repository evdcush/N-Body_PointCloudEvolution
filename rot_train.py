import os, code, sys, time, argparse
import numpy as np
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf
import nn
import utils
#from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, NUM_VAL_SAMPLES, MODEL_BASENAME
from utils import REDSHIFTS, AttrDict, SINGLE_STEP, MULTI_STEP


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
args.channels[0] = 10
args.channels[-1] = 3
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
multi_step = args.model_type == MULTI_STEP
checkpoint = 100 #args.checkpoint
num_iters  = args.num_iters
num_val_batches = args.num_test // batch_size
save_checkpoint = lambda step: (step+1) % checkpoint == 0
get_graph_csr = lambda h: nn.get_graph_csr_list(h, args)

# Network config
# ========================================
network_features = AttrDict()
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

#              RotInv
# ----------------------------------------
'''
e : N*(M-1)*(M-2), num of edges in 3D adjacency


#==== nn.model_func_RotInv
X_input : (b, N, 3)

edges : (b, e, 10)
    |----- # get_RotInv_input_edges(X, V, lst_csrs, M)
                X, V : (b,N,3) location, velocities
                lst_csrs : (b,)-len list of csrs
                M : num neighbors

segID_3D : (b, 7, e)
    |----- # get_batch_3D_segmentID(lst_csrs, M)
                lst_csrs : (b,)-len list of csrs

segID_2D : (2, b*N*(M-1), 2)
    |----- # get_batch_2D_segmentID(lst_csrs)
                lst_csrs : (b,)-len list of csrs


#==== Variable
model_specs
 var_scope :
num_layers :
      dims : (b, N, M)
activation :
'''


# Placeholders
#==== Data cube
X_input = tf.placeholder(tf.float32, shape=(None, N, 3))
X_truth = tf.placeholder(tf.float32, shape=(None, N, 6))

#==== Graph
#e = batch_size * (M-1) * (M-2)
edges_in = tf.placeholder(tf.float32, shape=(None, None, 10))
segID_3D = tf.placeholder(tf.int32, shape=(None, 7, None))
segID_2D = tf.placeholder(tf.int32, shape=(2, None, 2)) # (2, b*N*(M-1), 2)
#RS_in   = tf.placeholder(tf.float32, shape=(None, 1))


# Input args
#model_args = (X_input, COO_feats, network_features)
model_args = (X_input, edges_in, segID_3D, segID_2D, network_features)
loss_args  = (X_truth,)
#loss_args = (X_truth, X_input) if multi_step else (X_truth,)
#rs = RS_in if args.cat_rs else None


# Outputs
# ========================================
#X_pred = nn.model_func_ShiftInv(*model_args, redshift=None)#rs)
X_pred = nn.model_func_RotInv(*model_args, redshift=None)#rs)


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
X = utils.normalize(utils.load_simulation_data(rs_idx))
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

    # Split data
    # ----------------
    # input and truth
    x_in    = _x_batch[0] # (b, N, 6)
    x_truth = _x_batch[1] # (b, N, 6)

    # Graph data
    # ----------------
    csr_list = get_graph_csr(x_in) # len b list of (N,N) csrs

    #==== Rot inv pre-processing
    seg_id_2D = nn.get_batch_2D_segmentID(csr_list)
    seg_id_3D = nn.get_batch_3D_segmentID(csr_list, M)
    edges = nn.get_RotInv_input_edges(x_in, csr_list, M)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

    # Feed data to tensors
    # ----------------
    fdict = {X_input: x_in[...,:3],
             X_truth: x_truth,
             edges_in : edges,
             segID_2D : seg_id_2D,
             segID_3D : seg_id_3D,
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

    #==== Rot inv pre-processing
    seg_id_2D = nn.get_batch_2D_segmentID(csr_list)
    seg_id_3D = nn.get_batch_3D_segmentID(csr_list, M)
    edges = nn.get_RotInv_input_edges(x_in, csr_list, M)


    # Feed data to tensors
    # ----------------
    fdict = {X_input: x_in[...,:3],
             X_truth: x_truth,
             edges_in : edges,
             segID_2D : seg_id_2D,
             segID_3D : seg_id_3D,
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
