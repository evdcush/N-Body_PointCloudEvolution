import os, code, sys, time, argparse
import numpy as np
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf
#from tensorflow.core.protobuf import config_pb2
import nn
import utils
from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, NUM_VAL_SAMPLES, MODEL_BASENAME


parser = argparse.ArgumentParser()
# argparse not handle bools well so 0,1 used instead
parser.add_argument('--seed', '-s', default=PARAMS_SEED,     type=int, help='initial parameter seed')
parser.add_argument('--redshifts', '-z', default=[18,19], nargs='+', type=int, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--particles', '-p', default=32,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
#parser.add_argument('--model_type','-m', default=1,          type=int,  help='model type')
parser.add_argument('--graph_var', '-k', default=14,         type=int, help='search parameter for graph model')
parser.add_argument('--restore',   '-r', default=0,          type=int,  help='resume training from serialized params')
parser.add_argument('--num_iters', '-i', default=1000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=8,          type=int,  help='training batch size')
parser.add_argument('--model_name', '-m', default=MODEL_BASENAME, type=str,  help='directory where model parameters are saved')
parser.add_argument('--save_suffix','-n', default='',        type=str,  help='model name suffix')
parser.add_argument('--variable',   '-q', default=0,  type=int, help='multi-purpose variable argument')
parser.add_argument('--variable2',  '-c', default=0,          type=int,  help='multi-purpose variable argument2')
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
#model_type = pargs['model_type'] # 0: set, 1: graph
#model_vars = utils.NBODY_MODELS[model_type]
var_timestep = False
learning_rate = LEARNING_RATE # 0.01

# Network depth and channel sizes
# ----------------
channels = [6, 32, 16, 8, 3]
channels[0]  = 9
channels[-1] = 3
#channels[-1] = 6
num_layers = len(channels) - 1
M = pargs['graph_var']

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
#model_name = utils.get_zuni_model_name(model_type, zX, zY, pargs['save_prefix'])
#paths = utils.make_save_dirs(pargs['model_dir'], model_name)
#model_path, loss_path, cube_path = paths

'''
restore_model_parameters(self, sess):
save_model_files(self):
save_model_cube(self, cube, rs, save_path=self.result_path, ground_truth=False):
save_model_error(self, error, save_path=self.result_path, training=False):
save_model_params(self, session, cur_iter):
'''

# save test data
train_saver.save_model_cube(X_test, redshift_steps, ground_truth=True)
train_saver.save_model_files()
#utils.save_test_cube(X_test, cube_path, (zX, zY), prediction=False)
#utils.save_pyfiles(model_path)


#=============================================================================
# INITIALIZE model parameters and placeholders
#=============================================================================
# Init model params
# ----------------
'''
def initialize_model_params(layer_type, channels, scope=VAR_SCOPE,
                            seed=PARAMS_SEED, restore=False, **kwargs)
'''
#vscope = utils.VAR_SCOPE
vscope = utils.VARIABLE_SCOPE.format(zX, zY)
utils.initialize_model_params(ltype, channels, vscope, restore=restore)

#seed = pargs['seed']
#if seed != PARAMS_SEED:
#    print('\n\n\n USING DIFFERENT RANDOM SEED: {}\n\n\n'.format(seed))
#tf.set_random_seed(seed)
#utils.init_ShiftInv_params(channels, vscope, restore=restore)
#with tf.variable_scope(vscope):
#    utils.init_coeff_single(restore=restore)
#    #utils.init_coeff_agg(num_rs_layers, restore=restore)


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
X_pred = nn.ShiftInv_model_func(X_input, COO_feats, model_specs)


# Loss
# ----------------
# Training error and Optimizer
# https://www.tensorflow.org/api_docs/python/tf/contrib/opt/LazyAdamOptimizer
'''
if pargs['variable'] != 0:
    optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate)
    print('\nLazy Adam\n')
else:
    optimizer = tf.train.AdamOptimizer(learning_rate)
    print('\nregular Adam\n')
'''
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

#run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
#run_opts = config_pb2.RunOptions(report_tensor_allocations_upon_oom=True)

# initialize variables
sess.run(tf.global_variables_initializer())
train_saver.initialize_saver()
if restore:
    train_saver.restore_model_parameters(sess)
    #utils.load_graph(sess, model_path)



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
#utils.save_loss(loss_path + model_name, test_loss, validation=True)
#utils.save_loss(loss_path + model_name + 'SC', test_loss_sc, validation=True)
#utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#restore_model_parameters(self, sess):
#save_model_files(self):
#save_model_cube(self, cube, rs, save_path=self.result_path, ground_truth=False):
#save_model_error(self, error, save_path=self.result_path, training=False):
#save_model_params(self, session, cur_iter):

train_saver.save_model_error(test_loss)
train_saver.save_model_cube(test_predictions, (zX, zY), ground_truth=False)


#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
