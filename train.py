import os, code, sys, time, argparse
import numpy as np
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf
import nn
import utils
#from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, NUM_VAL_SAMPLES, MODEL_BASENAME
from utils import REDSHIFTS


""" TODO:

Would-be-nice:
 X done - have some other script handle the parse-args, instead of polluting here

"""
start_time = time.time()
#==============================================================================
# Model/Session config
#==============================================================================
# Arg parser
# ========================================
parser = utils.Parser()
args = parser.parse_args()

#------------------------------------------------------------------------------
# Data features
#------------------------------------------------------------------------------
# Dataset vars
# ========================================
redshift_idx = args.redshifts
redshifts = [REDSHIFTS[i] for i in redshift_idx]
zx, zy = redshifts[0], redshifts[-1]


# Dimensionality
# ========================================
num_redshifts = len(redshifts)
N = 32**3
M = args.graph_var
batch_size = args.batch_size
num_test_samples = args.num_test
#cube_dims = [batch_size, N, M, ]

# Network features
# ========================================
channels   = args.channels
layer_type = args.layer_type
network_depth = len(channels) - 1



# Training parameters
# ========================================

#------------------------------------------------------------------------------
# Model structure
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Session features
#------------------------------------------------------------------------------
# Session variables
# ========================================
var_scope  = args.var_scope.format(zx, zy)





#==============================================================================
# Setup computational graph
#==============================================================================
#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------
X = utils.normalize(utils.load_simulation_data(redshift_idx))
X_train, X_test = utils.split_data_validation(X, num_val=num_test_samples)
X = None # reduce memory overhead

#------------------------------------------------------------------------------
# Specify graph nodes
#------------------------------------------------------------------------------
# Inputs
# ========================================
X_input = tf.placeholder(tf.float32, shape=(None, N, 6))
X_truth = tf.placeholder(tf.float32, shape=(None, N, 6))
COO_feats = tf.placeholder(tf.int32, shape=(3, None,))


# Output
# ========================================
dims = [batch_size,N,M]
model_specs = utils.AttrDict()
model_specs.num_layers = num_layers
model_specs.var_scope = vscope
model_specs.activation_func = tf.nn.relu
model_specs.dims = dims
X_pred = nn.model_func_ShiftInv(X_input, COO_feats, model_specs)

# Optimization
# ========================================
optimizer = tf.train.AdamOptimizer(learning_rate)

error = nn.pbc_loss(X_pred, X_truth, vel=False)
train = optimizer.minimize(error)


#------------------------------------------------------------------------------
# Model variables
#------------------------------------------------------------------------------
# Data features
# ========================================
dims = [batch_size,N,M]
model_specs = utils.AttrDict()
model_specs.num_layers = num_layers
model_specs.var_scope = vscope
model_specs.activation_func = tf.nn.relu
model_specs.dims = dims

# Init model params
# ----------------
vscope = utils.VARIABLE_SCOPE.format(zX, zY)
utils.initialize_model_params(ltype, channels, vscope, restore=restore)


# Model static func args
# ----------------
#model_specs = nn.ModelFuncArgs(num_layers, vscope, dims=[batch_size,N,M])
dims = [batch_size,N,M]
model_specs = utils.AttrDict()
model_specs.num_layers = num_layers
model_specs.var_scope = vscope
model_specs.activation_func = tf.nn.relu
model_specs.dims = dims
_X_edges, _X_nodes = nn.get_input_features_ShiftInv(X_input, COO_feats, dims)

# Model outputs
# ----------------
# Train
if compute_edges_nodes:
    #X_edges, X_nodes = nn.get_input_features_ShiftInv(X_input, COO_feats, dims)
    X_pred = nn.model_func_ShiftInv_preprocess_assumption(X_input, X_edges, X_nodes, COO_feats, model_specs)
else:
    X_pred = nn.model_func_ShiftInv(X_input, COO_feats, model_specs)



# Loss
# ----------------
optimizer = tf.train.AdamOptimizer(learning_rate)

error = nn.pbc_loss(X_pred, X_truth, vel=False)
train = optimizer.minimize(error)
#sc_error = nn.pbc_loss_scaled(X_input, X_pred, X_truth, vel=False)
#train = tf.train.AdamOptimizer(learning_rate).minimize(sc_error)

# Validation error
#val_error   = nn.pbc_loss(X_pred_val, X_truth, vel=False)
#inputs_diff = nn.pbc_loss(X_input,    X_truth, vel=False)


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
train_saver.initialize_saver()
if restore:
    train_saver.restore_model_parameters(sess)
    #utils.load_graph(sess, model_path)


#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
#=============================================================================
# NBODY Data
#=============================================================================
# Nbody data specs
# ----------------
num_particles = pargs['particles'] # 32
N = num_particles**3
redshift_steps = pargs['redshifts']
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1
# restore
restore = pargs['restore'] == 1

# Load data
# ----------------
X = utils.normalize(utils.load_simulation_data(redshift_steps))
X_train, X_test = utils.split_data_validation(X)
X = None # reduce memory overhead


#=============================================================================
# Model and network features
#=============================================================================
# Model features
# ----------------
learning_rate = LEARNING_RATE # 0.01
#learning_rate = 0.001

# Network depth and channel sizes
# ----------------
channels = [6, 32, 16, 8, 3]
channels[0]  = 9
channels[-1] = 3
#channels[-1] = 6
num_layers = len(channels) - 1
M = pargs['graph_var']
compute_edges_nodes = pargs['variable'] != 0

# Training hyperparameters
# ----------------
#threshold = 0.03 # for PBC kneighbor search, currently not supported
batch_size = pargs['batch_size']
num_iters  = pargs['num_iters']


#=============================================================================
# Session save parameters
#=============================================================================
# Model name and paths
# ----------------
ltype = 'shift-inv'
mtype = 'single-step'
if pargs['model_name'] != MODEL_BASENAME:
    model_name = pargs['model_name']
else:
    model_name = utils.get_model_name(mtype, ltype, redshift_steps, suffix=pargs['save_suffix'])

#(self, mname, num_iters, always_write_meta=False, restore=False)
train_saver = utils.TrainSaver(model_name, num_iters, always_write_meta=True, restore=restore)
zX = redshift_steps[0]  # starting redshift
zY = redshift_steps[-1] # target redshift


# save test data
train_saver.save_model_cube(X_test, redshift_steps, ground_truth=True)
train_saver.save_model_files()



#=============================================================================
# INITIALIZE model parameters and placeholders
#=============================================================================
# Init model params
# ----------------
vscope = utils.VARIABLE_SCOPE.format(zX, zY)
utils.initialize_model_params(ltype, channels, vscope, restore=restore)


# CUBE DATA
# ----------------
X_input = tf.placeholder(tf.float32, shape=(None, N, 6))
X_truth = tf.placeholder(tf.float32, shape=(None, N, 6))
X_edges = tf.placeholder(tf.float32, shape=(batch_size*N*M,3))
X_nodes = tf.placeholder(tf.float32, shape=(batch_size*N,3))



# NEIGHBOR GRAPH DATA
# ----------------
# these shapes must be concrete for unsorted_segment_mean
#COO_feats = tf.placeholder(tf.int32, shape=(3, batch_size*N*M,))
COO_feats = tf.placeholder(tf.int32, shape=(3, None,))


#=============================================================================
# MODEL output and optimization
#=============================================================================
# helper for kneighbor search
def get_list_csr(h_in):
    return nn.get_kneighbor_list(h_in, M, inc_self=False)
    # thresh 0.03 on (0,19) about 6.1 % of total num of particles for one samp
    #return nn.get_pbc_kneighbors_csr(h_in, M, boundary_threshold=0.05, include_self=False)







# Session saver
# ----------------
#saver = tf.train.Saver()
#saver.save(sess, model_path + model_name)
train_saver.save_model_params(sess, 0)
checkpoint = 100
save_checkpoint = lambda step: (step+1) % checkpoint == 0


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
    csr_list = get_list_csr(x_in) # len b list of (N,N) csrs
    coo_feats = nn.to_coo_batch(csr_list)

    # Feed data to tensors
    # ----------------
    fdict = {X_input: x_in,
             X_truth: x_truth,
             COO_feats: coo_feats,
             }

    if compute_edges_nodes:
        edges, nodes = sess.run([_X_edges, _X_nodes], feed_dict=fdict)
        fdict[X_edges] = edges
        fdict[X_nodes] = nodes

    # Train
    #err = sess.run(error, feed_dict=fdict, options=run_opts)
    train.run(feed_dict=fdict)

    # Checkpoint
    # ----------------
    # Track error
    """
    if (step + 1) % 5 == 0:
        e = sess.run(error, feed_dict=fdict)
        print('{:>5}: {}'.format(step+1, e))
    """


    # Save
    if save_checkpoint(step):
        err = sess.run(error, feed_dict=fdict)
        utils.print_checkpoint(step, err)
        #saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)
        #save_model_params(self, session, cur_iter)
        train_saver.save_model_params(sess, step)


# END training
# ========================================
print('elapsed time: {}'.format(time.time() - start_time))

# Save trained variables and session
#saver.save(sess, model_path + model_name, global_step=num_iters, write_meta_graph=True)
train_saver.save_model_params(sess, num_iters)
X_train = None # reduce memory overhead


#=============================================================================
# EVALUATION
#=============================================================================
# Eval data containers
# ----------------
num_val_batches = NUM_VAL_SAMPLES // batch_size
test_predictions  = np.zeros(X_test.shape[1:-1] + (channels[-1],)).astype(np.float32)
#test_predictions  = np.zeros(X_test.shape[1:-1] + (6,)).astype(np.float32)
test_loss = np.zeros((num_val_batches,)).astype(np.float32)
#test_loss_sc = np.zeros((num_val_batches,)).astype(np.float32)

print('\nEvaluation:\n{}'.format('='*78))
#for j in range(X_test.shape[1]):
for j in range(num_val_batches):
    # Validation cubes
    # ----------------
    p, q = batch_size*j, batch_size*(j+1)
    x_in    = X_test[0, p:q]
    x_truth = X_test[1, p:q]

    # Graph data
    # ----------------
    csr_list = get_list_csr(x_in) # len b list of (N,N) csrs
    coo_feats = nn.to_coo_batch(csr_list)

    # Feed data to tensors
    # ----------------
    fdict = {X_input: x_in,
             X_truth: x_truth,
             COO_feats: coo_feats,
             }

    if compute_edges_nodes:
        edges, nodes = sess.run([_X_edges, _X_nodes], feed_dict=fdict)
        fdict[X_edges] = edges
        fdict[X_nodes] = nodes

    # Validation output
    # ----------------
    x_pred_val, v_error = sess.run([X_pred, error], feed_dict=fdict)
    #x_pred_val, v_error, v_sc_error = sess.run([X_pred, error, sc_error], feed_dict=fdict)
    test_predictions[p:q] = x_pred_val
    test_loss[j] = v_error
    #test_loss_sc[j] = v_sc_error
    #print('{:>3d} = LOC: {:.6f}'.format(j, v_error))
    #print('{:>3d} = LOC: {:.8f}, SCA: {:.6f}'.format(j, v_error, v_sc_error))


# END Validation
# ========================================
utils.print_median_validation_loss(redshift_steps, test_loss)

train_saver.save_model_error(test_loss)
train_saver.save_model_cube(test_predictions, (zX, zY), ground_truth=False)


#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

