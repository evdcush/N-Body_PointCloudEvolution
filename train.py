import code
import numpy as np
import tensorflow as tf
from utils import data_loader, initializer, saver, parser
import nn
#=============================================================================#
#                        _____    ___    ____     ___                         #
#                       |_   _|  / _ \  |  _ \   / _ \                        #
#                         | |   | | | | | | | | | | | |                       #
#                         | |   | |_| | | |_| | | |_| |                       #
#                         |_|    \___/  |____/   \___/                        #
#                                                                             #
"""===========================================================================#

# 1
- TRY CURRENT GRAPH-BASED UPDATES TO ZA MODEL

# 2
IF THINGS WORK: update to correct shift_inv ops (15 ops)

# 3 encoding
graph model with different encoding
- instead of placeing za on diagonal
  ---> define a set of neighbors as the union of grid pos and final ZA pos
       (where ZA_final_pos is just za_init + za_disp)
- each edge will contain the diff between corresponding grid pos,
  and the diff between corresponding final pos

# 4 SET:
(this task should be easy), thus we shouldnt need all this expensive machinery
with graph
- Instead have a set of N particles (vanilla), so just nodes
  - node features:
    - ZA_disp
    - positions of init_particles
- drastically reduced ops
  - just vanilla: affine transformation with pooling op (mean, max, etc..)


#==========================================================================="""




#==============================================================================
# Data & Session config
#==============================================================================

# Arg parser
# ========================================
arg_parser = parser.Parser()
args = arg_parser.parse()

saver = saver.ModelSaver(args)
sess_mgr = initializer.Initializer(args)
dataset  = data_loader.get_dataset(args)
#dataset.normalize()


# Dimensionality
# ========================================
N = 32**3
M = args.neighbors
batch_size = args.batch_size

# Training variables
# ========================================
learning_rate = 0.01
checkpoint = 250 #args.checkpoint
num_iters  = args.num_iters
num_val_batches = args.num_eval_samples // batch_size
save_checkpoint = lambda step: (step+1) % checkpoint == 0


# Network config
# ========================================
dims = [batch_size, N, M]
#loss_func = nn.pbc_loss
loss_func = nn.loss_ZA if args.dataset_type == 'ZA' else nn.pbc_loss

# Initialize model parameters
# ========================================
sess_mgr.initialize_params()

# Inputs
# ========================================
# Placeholders
#in_shape = (None, N, 6)
in_shape = (None, N, 3)
X_input = tf.placeholder(tf.float32, shape=in_shape, name='X_input')
#X_truth = tf.placeholder(tf.float32, shape=in_shape)
true_error  = tf.placeholder(tf.float32,shape=in_shape, name='true_error')
coo_feats   = tf.placeholder(tf.int32,  shape=(3, batch_size*N*M), name='coo_feats')
za_displacement = tf.placeholder(tf.float32, shape=in_shape, name='za_displacement')
#za_diagonal = tf.placeholder(tf.int32,  shape=(batch_size*N*M))
za_diagonal = tf.placeholder(tf.int32,  shape=(batch_size*N), name='za_diagonal')


# Outputs
# ========================================
#X_pred = nn.model_func_shift_inv(X_input, coo_feats, sess_mgr, dims)
model_args = (X_input, coo_feats, za_displacement, za_diagonal, sess_mgr, dims)
pred_error = nn.model_func_shift_inv_za(*model_args)

# Optimizer and loss
# ========================================
optimizer = tf.train.AdamOptimizer(learning_rate)
#error = loss_func(X_pred, X_truth, scale_error=False)
error = nn.loss_ZA(pred_error, true_error)
train = optimizer.minimize(error)


# Initialize session and variables
# ========================================
# Session
sess = sess_mgr()

# Variables
sess_mgr.initialize_graph()
saver.init_sess_saver()


test_pred_shape = (200, N,) + (args.channels[-1],) # (200, N, 3)
test_predictions = np.zeros(test_pred_shape).astype(np.float32)
test_loss = np.zeros((50,)).astype(np.float32)

#=============================================================================
# TRAINING
#=============================================================================
print(f'\nTraining:\n{"="*78}')
for step in range(num_iters):
    # Data batching
    # ----------------
    _x_batch = dataset.get_minibatch() # (2, 4, N, 6)

    # split data
    x_za  = _x_batch[0] # (b, N, 6)
    x_fpm = _x_batch[1] # (b, N, 6)

    # displacements
    x_za_disp  = x_za[...,:3]
    x_fpm_disp = x_fpm[...,:3]

    # get initial displacement
    init_pos = nn.get_init_pos(x_za_disp)

    # calculate true_error
    true_err = x_fpm_disp - x_za_disp

    # Graph data
    # ----------------
    csr_list  = nn.get_kneighbor_list(init_pos, M)
    #coo_batch = nn.to_coo_batch(csr_list)
    coo_batch, diag = nn.to_coo_batch_ZA_diag(csr_list)

    # Feed data and Train
    fdict = {
        X_input : init_pos,
        #X_truth : x_truth,
        true_error : true_err,
        za_displacement : x_za_disp,
        za_diagonal : diag,
        coo_feats : coo_batch,
    }
    train.run(feed_dict=fdict)

    # Save
    if save_checkpoint(step):
        err = sess.run(error, feed_dict=fdict)
        saver.save_model_params(step, sess)
        saver.print_checkpoint(step, err)


# Save trained variables and session
saver.save_model_params(num_iters, sess)

#=============================================================================
# EVALUATION
#=============================================================================
print(f'\nEvaluation:\n{"="*78}')
X_test = dataset.X_test
for j in range(num_val_batches): # ---> range(50) for b = 4
    # Validation cubes
    # ----------------
    p, q = batch_size*j, batch_size*(j+1)
    _x_batch = X_test[:, p:q]

    # split data
    x_za  = _x_batch[0] # (b, N, 6)
    x_fpm = _x_batch[1] # (b, N, 6)

    # displacements
    x_za_disp  = x_za[...,:3]
    x_fpm_disp = x_fpm[...,:3]

    # get initial displacement
    init_pos = nn.get_init_pos(x_za_disp)

    # calculate true_error
    true_err = x_fpm_disp - x_za_disp

    # Graph data
    # ----------------
    csr_list  = nn.get_kneighbor_list(init_pos, M)
    #coo_batch = nn.to_coo_batch(csr_list)
    coo_batch, diag = nn.to_coo_batch_ZA_diag(csr_list)

    # Feed data and Train
    fdict = {
        X_input : init_pos,
        #X_truth : x_truth,
        true_error : true_err,
        za_displacement : x_za_disp,
        za_diagonal : diag,
        coo_feats : coo_batch,
    }

    # Validation output
    # ----------------
    p_error, v_error = sess.run([pred_error, error], feed_dict=fdict)
    #test_predictions[p:q] = x_pred_val
    test_predictions[p:q] = p_error
    test_loss[j] = v_error
    print(f'val_err, {j} : {v_error}')



# END Validation
# ========================================
saver.save_model_cube(test_predictions)
saver.save_model_error(test_loss)
saver.print_evaluation_results(test_loss)
