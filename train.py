import sys
import code
import time
import numpy as np
import tensorflow as tf

import nn
import utils
from utils import PARSER, Dataset, Saver


#-----------------------------------------------------------------------------#
#                           Training & Model Config                           #
#-----------------------------------------------------------------------------#

# Parse args
args = PARSER.parse_args()

# Training
# ========
num_iters  = args.num_iters
batch_size = args.batch_size
data_idx   = args.data_idx
num_test   = args.num_test
model_name = args.name
saver = Saver(data_idx, model_tag=model_name)

# train loop
checkpoint = 250
save_checkpoint = lambda step: (step+1) % checkpoint == 0

# Data
# ====
num_test_batches = num_test // batch_size
num_particles = 32**3
dataset = Dataset(data_idx, num_test)

# Model
# =====
lr = args.learnrate
channels = args.channels
num_layers = len(channels) - 1
params_seed = args.seed
var_scope = utils.VAR_SCOPE
get_layer_vars = lambda i: utils.get_params(i, vscope=var_scope)
activation = tf.nn.relu
model_vars = utils.ModelVars(num_layers, get_layer_vars, activation)
#kneighbors = args.kneighbors  # focusing on set


#-----------------------------------------------------------------------------#
#                               Session Config                                #
#-----------------------------------------------------------------------------#

# Initialize params
utils.initialize_params(channels, vscope=var_scope, seed=params_seed)

# Inputs
# ======
in_shape = (None, num_particles, 3)
X_input = tf.placeholder(tf.float32, shape=in_shape[:-1] + (6,))
true_error = tf.placeholder(tf.float32, shape=in_shape)

# Outputs
# =======
pred_error = nn.model_func_set(X_input, model_vars)

# Optimizer and loss
# ==================
optimizer = tf.train.AdamOptimizer(lr)
error = nn.loss_ZA(pred_error, true_error)
train = optimizer.minimize(error)

# Initialize session and variables
sess = utils.initialize_session()
utils.initialize_graph(sess)
saver.init_sess_saver()

#=============================================================================#
#                                    Training                                 #
#=============================================================================#
#num_chkpts = num_iters // checkpoint
#training_error = np.zeros((2, num_chkpts, batch_size, N, 3)).astype(np.float32)
tstart = time.time()

print(f'\nTraining:\n{"="*78}')
for step in range(num_iters):
    # Data batching
    # ----------------
    _x_batch = dataset.get_minibatch() # (2, 4, N, 6)

    # split data
    #x_za  = _x_batch[0] # (b, N, 6)
    #x_fpm = _x_batch[1] # (b, N, 6)
    x_za = _x_batch[...,:6]
    x_fpm = _x_batch[...,6:]

    # displacements
    #x_za_disp  = x_za[...,:3]
    #x_fpm_disp = x_fpm[...,:3]

    # get initial positions
    #init_pos = nn.get_init_pos(x_za_disp)

    # calculate true_error
    #true_err = x_fpm_disp - x_za_disp

    # Feed data and Train
    #code.interact(local=dict(globals(), **locals()))
    fdict = {
        X_input : x_za,
        true_error : x_fpm, #true_err,
    }
    train.run(feed_dict=fdict)

    # Save
    if save_checkpoint(step):
        err, pred_err = sess.run([error, pred_error], feed_dict=fdict)
        saver.save_model(step, sess)
        saver.print_checkpoint(step, err)

tfin = time.time()
est_time = (tfin - tstart) / 60  # minutes
print(f"Training finished!\n\tElapsed time: {est_time:.2f}m")
# Save trained variables and session
saver.save_model(num_iters, sess, write_meta=True)


# Test results
# ============
test_error = np.zeros((num_test_batches,), dtype=np.float32)
test_predictions = np.zeros((2, num_test, num_particles, channels[-1]), dtype=np.float32)

#=============================================================================#
#                                 Evaluation                                  #
#=============================================================================#

print(f'\nEvaluation:\n{"="*78}')
X_test = dataset.X_test
for j in range(num_test_batches): # ---> range(50) for b = 4
    # Validation cubes
    # ----------------
    p, q = batch_size*j, batch_size*(j+1)
    #_x_batch = X_test[:, p:q]
    _x_batch = X_test[p:q]

    # split data
    #x_za  = _x_batch[0] # (b, N, 6)
    #x_fpm = _x_batch[1] # (b, N, 6)
    x_za = _x_batch[...,:6]
    x_fpm = _x_batch[...,6:]


    # displacements
    #x_za_disp  = x_za[...,:3]
    #x_fpm_disp = x_fpm[...,:3]

    # calculate true_error
    #true_err = x_fpm_disp - x_za_disp

    # Feed data and Train
    fdict = {
        X_input : x_za,
        true_error : x_fpm, #true_err,
    }

    # Validation output
    # ----------------
    p_error, v_error = sess.run([pred_error, error], feed_dict=fdict)
    #test_predictions[0, p:q] = true_err
    test_predictions[0, p:q] = x_fpm
    test_predictions[1, p:q] = p_error
    test_error[j] = v_error
    print(f'val_err, {j} : {v_error}')



# END Validation
# ========================================
saver.save_cube(test_predictions)
saver.save_error(test_error)
saver.print_evaluation_results(test_error)


