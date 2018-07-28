import os, code, sys, time, argparse
import numpy as np
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf
import nn
import utils
from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, RS_TAGS, NUM_VAL_SAMPLES

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
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1

# Load data
# ----------------
X = utils.load_zuni_npy_data(redshift_steps, norm_coo=True)[...,:-1]
#X = utils.load_rs_npy_data(redshift_steps, norm_coo=True, old_dataset=True)[...,:-1]
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=NUM_VAL_SAMPLES)
timestep = utils.get_timestep(X[0], X[1])
print('timestep = {}'.format(timestep))
X = None # reduce memory overhead


#=============================================================================
# Model and network features
#=============================================================================
# Model features
# ----------------
model_type = pargs['model_type'] # 0: set, 1: graph
model_vars = utils.NBODY_MODELS[model_type]
use_coeff  = pargs['vcoeff'] == 1
var_timestep = False

# Network depth and channel sizes
# ----------------
#channels = model_vars['channels'] # OOM with sparse graph
channels = [6, 32, 16, 8, 3]
channels[0]  = 9
channels[-1] = 3
#channels[-1] = 6
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
utils.save_pyfiles(model_path)


#=============================================================================
# INITIALIZE model parameters and placeholders
#=============================================================================
# Init model params
# ----------------
vscope = utils.VAR_SCOPE.format(zX, zY)
tf.set_random_seed(utils.PARAMS_SEED)
utils.init_ShiftInv_params(channels, vscope, restore=restore)

if use_coeff:
    if not var_timestep:
        scalar_tag = utils.init_coeff(vscope, redshift_steps, restore=restore)
        print('Scalar tag: {}'.format(scalar_tag))
    else: # CLEAN THIS UP, SHOULDN'T SCOPE INIT HERE
        with tf.variable_scope(vscope):
            #utils.init_coeff_multi(num_rs_layers)
            utils.init_coeff_multi2(num_rs_layers, restore=restore)
            #utils.init_coeff_agg(num_rs_layers, restore=restore)



# CUBE DATA
# ----------------
X_input = tf.placeholder(tf.float32, shape=(None, N, 6))
X_truth = tf.placeholder(tf.float32, shape=(None, N, 6))

# NEIGHBOR GRAPH DATA
# ----------------
# these shapes must be concrete for unsorted_segment_mean
COO_feats = tf.placeholder(tf.int32, shape=(3, batch_size*N*M,))


#=============================================================================
# MODEL output and optimization
#=============================================================================
# helper for kneighbor search
def get_list_csr(h_in):
    return nn.get_kneighbor_list(h_in, M, inc_self=False, )#pbc=True)


# Model static func args
# ----------------
model_specs = nn.ModelFuncArgs(num_layers, vscope, dims=[batch_size,N,M])

# Model outputs
# ----------------
# Train
#X_pred = nn.ShiftInv_single_model_func_v1(X_input, COO_feats, model_specs, coeff_idx=0)

# Static timestep model:
X_pred = nn.ShiftInv_model_func_timestep(X_input, COO_feats, model_specs, timestep, scalar_tag=scalar_tag)


# Loss
# ----------------
# Training error and Optimizer
error = nn.pbc_loss(X_pred, X_truth, vel=False)
train = tf.train.AdamOptimizer(learning_rate).minimize(error)
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
if restore:
    utils.load_graph(sess, model_path)



# Session saver
# ----------------
saver = tf.train.Saver()
saver.save(sess, model_path + model_name)
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
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=False) # shape (2, b, N, 6)

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

    # Train
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
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)


# END training
# ========================================
print('elapsed time: {}'.format(time.time() - start_time))

# Save trained variables and session
saver.save(sess, model_path + model_name, global_step=num_iters, write_meta_graph=True)
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
#zx, zy = redshift_steps
#print('# LOCATION LOSS:')
#print('  {:>2} --> {:>2}: {:.9f}'.format(zx, zy, test_median))


#MCOEFFTAG = 'coeff_{}'
#VEL_COEFF_TAG = 'V'
#t0 = utils.get_var('coeff_{}_{}'.format(0,0))[0]
#t1 = utils.get_var('coeff_{}_{}'.format(0,1))[0]
#print(' TIMESTEP, final value: {:.6f}'.format(t1))
#print('LOCSCALAR, final value: {:.6f}'.format(t0))


# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss, validation=True)
#utils.save_loss(loss_path + model_name + 'SC', test_loss_sc, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
