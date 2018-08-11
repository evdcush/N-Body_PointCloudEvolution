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
parser.add_argument('--seed', '-s', default=PARAMS_SEED,     type=int, help='initial parameter seed')
parser.add_argument('--particles', '-p', default=32,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--model_type','-m', default=1,          type=int,  help='model type')
parser.add_argument('--graph_var', '-k', default=14,         type=int, help='search parameter for graph model')
parser.add_argument('--restore',   '-r', default=0,          type=int,  help='resume training from serialized params')
parser.add_argument('--num_iters', '-i', default=1000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=8,          type=int,  help='training batch size')
parser.add_argument('--model_dir', '-d', default='./Model/', type=str,  help='directory where model parameters are saved')
parser.add_argument('--vcoeff',    '-c', default=0,          type=int,  help='use timestep coefficient on velocity')
parser.add_argument('--save_prefix','-n', default='',        type=str,  help='model name prefix')
parser.add_argument('--variable',   '-q', default=0.0,       type=float, help='multi-purpose variable argument')
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
rs_tups = [(i,i+1) for i in range(num_rs_layers)]
#rs_tups.append((0, 2))
print('redshifts: {}'.format(redshifts))

# Load data
# ----------------
X = utils.load_zuni_npy_data(redshift_steps, norm_coo=True)
#X = utils.load_rs_npy_data(redshift_steps, norm_coo=True, old_dataset=True)[...,:-1]
X_train, X_test = utils.split_data_validation_combined(X, num_val_samples=NUM_VAL_SAMPLES)
#timesteps = [utils.get_timestep(X[i], X[i+1]) for i in range(num_rs_layers)]
#print('timesteps = {}'.format(timesteps))
#for i, tstep in enumerate(timesteps): print('timesteps[{}] = {:>2} --> {:>2} = {:.6f}'.format(i, redshift_steps[i], redshift_steps[i+1], tstep))

X = None # reduce memory overhead


#=============================================================================
# Model and network features
#=============================================================================
# Model features
# ----------------
model_type = pargs['model_type'] # 0: set, 1: graph
model_vars = utils.NBODY_MODELS[model_type]
use_coeff  = pargs['vcoeff'] == 1
p_variable = pargs['variable']
print('noise: {}'.format(p_variable))
noisy_inputs = p_variable != 0.0
#var_timestep = False

# Network depth and channel sizes
# ----------------
#channels = model_vars['channels'] # OOM with sparse graph
channels = [9, 32, 16, 8, 6]
channels[0]  = 10
#channels[0]  = 11
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
seed = pargs['seed']
if seed != PARAMS_SEED:
    print('\n\n\n USING DIFFERENT RANDOM SEED: {}\n\n\n'.format(seed))
tf.set_random_seed(seed)
#print('\n\n\n USING DIFFERENT RANDOM SEED: {}\n\n\n'.format(rng_seed))
#print('\n\nOFFSETTING LAYER INPUT DIMS FOR CONCAT REDSHIFTS\n\n')
utils.init_ShiftInv_params(channels, vscope, restore=restore, const_init=0.01)#rs_ccat=True)

# CUBE DATA
# ----------------
X_input = tf.placeholder(tf.float32, shape=(None, N, 6))
X_truth = tf.placeholder(tf.float32, shape=(None, N, 6))
RS_in = tf.placeholder(tf.float32,   shape=(None, 1))
#RS_in = tf.placeholder(tf.float32,   shape=(None, 2))

# NEIGHBOR GRAPH DATA
# ----------------
# these shapes must be concrete for unsorted_segment_mean
COO_seg = tf.placeholder(tf.int32, shape=(3, batch_size*N*M,))
COO_seg_val = tf.placeholder(tf.int32, shape=(3, N*M,))

# COEFFS
# ----------------
with tf.variable_scope(vscope):
    #utils.init_coeff_multi(num_rs_layers)
    #utils.init_coeff_multi2(num_rs_layers, restore=restore)
    utils.init_coeff_single(restore=restore)
    #utils.init_coeff_agg(num_rs_layers, restore=restore)


#=============================================================================
# MODEL output and optimization
#=============================================================================
# helper for kneighbor search
def get_list_csr(h_in):
    return nn.get_kneighbor_list(h_in, M, inc_self=False, )#pbc=True)
    #return nn.get_pbc_kneighbors_csr(h_in, M, boundary_threshold=0.03, include_self=False)

# Model static func args
# ----------------
train_args = nn.ModelFuncArgs(num_layers, vscope, dims=[batch_size,N,M],)

# Model outputs
# ----------------
# Train
#pred_in = (X_input, COO_seg, train_args, RS_in)
pred_in = (X_input, COO_seg, train_args)
#X_preds = {i: nn.ShiftInv_single_model_func(*pred_in, timesteps[i], scalar_tags[i]) for i in range(num_rs_layers)}
#X_preds = {i: nn.ShiftInv_model_func_timestep(*pred_in, i, RS_in) for i in range(num_rs_layers)}
#X_pred = nn.ShiftInv_model_func_timestep(X_input, COO_feats, model_specs, timestep, scalar_tag=scalar_tag)
X_pred = nn.ShiftInv_model_func_timestep(*pred_in, RS_in)

# Loss
# ----------------
# Optimizer
opt = tf.train.AdamOptimizer(learning_rate)

# Training error
#errors = {i: nn.pbc_loss_scaled(X_input, X_pred, X_truth) for i, X_pred in X_preds.items()}
error = nn.pbc_loss_scaled(X_input, X_pred, X_truth)

# Backprop on loss
#trains = {i: opt.minimize(error) for i, error in errors.items()}
train = opt.minimize(error)

# Validation error
X_pred_val = tf.placeholder(tf.float32, shape=(None, N, 6))
val_error = nn.pbc_loss(X_pred_val, X_truth, vel=False)
inputs_diff = nn.pbc_loss(X_input, X_truth, vel=False)

# Noisy Inputs probabilities
# ----------------
# Determines whether input will be ground truth or previous prediction
noise_threshold = p_variable
NOISY = False # stays False if not noisy_inputs!
if noisy_inputs:
    # sample probabilities for inserting noise from random uniform distr
    noise = np.random.sample((num_iters, num_rs_layers-1)) <= noise_threshold
    #keys, values = np.unique(np.sum(noise, axis=1), return_counts)
    #total_noise = 0
    noisy_inputs_ratio = (np.sum(noise) / noise.shape[1]) / num_iters
    print('\nRatio of noisy input: {:3f}\n'.format(noisy_inputs_ratio))

def insert_noise_next(step, z):
    """ # determines if we need the model prediction (X_pred) to use
    as a noisy input for the next redshift, and updates the x_pred_step
    state variable
    NB: z is assumed as next redshift, due to noise dims offset
    Args:
        step (int): the current training step
           z (int): the current redshift index
    Outputs: (bool) based on whether the next model input will be prev model output
      - mutates x_pred_step
    """
    # don't get model prediction if we aren't inserting noise at all
    if not noisy_inputs:
        return False
    # don't get model prediction if on last redshift
    #  since first (0th) rs layer always ground truth input (and step will be different)
    is_last_redshift_index = z == num_rs_layers - 1
    global NOISY # kind of not good to mutate state var like this??
    NOISY = False if is_last_redshift_index else noise[step, z]
    return NOISY


#=============================================================================
# Session setup
#=============================================================================
# Sess
# ----------------
gpu_frac = 0.8
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# initialize variables
sess.run(tf.global_variables_initializer())
def get_var(tag):
    with tf.variable_scope(vscope, reuse=True):
        return tf.get_variable(tag).eval()

if restore:
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    utils.load_graph(sess, model_path)


# Session saver
# ----------------
saver = tf.train.Saver()
saver.save(sess, model_path + model_name)
checkpoint = 200
save_checkpoint = lambda step: (step+1) % checkpoint == 0
#=============================================================================
# TRAINING
#=============================================================================
# SINGLE-STEP
# ----------------
print('\nTraining Single-step:\n{}'.format('='*78))
np.random.seed(utils.DATASET_SEED)
for step in range(num_iters):
    checkpoint_reached = save_checkpoint(step)
    if checkpoint_reached:
        print('CHECKPOINT {:>4}:'.format(step+1))
        saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)

    # Data batching
    # ----------------
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=False) # shape (3, b, N, 6)

    # Train on redshift pairs
    # ----------------
    #np.random.shuffle(rs_tups)
    for rsi, tup in enumerate(rs_tups):
        #print('Step {:>5}, rsi {}, tup {}'.format(step, rsi, tup))
        zx, zy = tup
        # redshift
        rs_in = np.full([batch_size*N*M, 1], redshifts[zy], dtype=np.float32)
        #rs_Xin = np.full([batch_size*N*M, 1], redshifts[zx], dtype=np.float32)
        #rs_Yin = np.full([batch_size*N*M, 1], redshifts[zy], dtype=np.float32)
        #rs_in = np.concatenate([rs_Xin, rs_Yin], axis=-1)

        # split data
        #x_in    = _x_batch[zx] # (b, N, 6)
        x_in = _x_batch[zx] if not NOISY else x_pred
        x_truth = _x_batch[zy] # (b, N, 6)


        # Graph data
        # ----------------
        csr_list = get_list_csr(x_in) # len b list of (N,N) csrs
        coo_segs = nn.to_coo_batch(csr_list)

        # Feed data to tensors
        # ----------------
        fdict = {X_input: x_in,
                 X_truth: x_truth,
                 COO_seg: coo_segs,
                 RS_in: rs_in,
                 }
        # Train
        #trains[rsi].run(feed_dict=fdict)
        train.run(feed_dict=fdict)

        if insert_noise_next(step, rsi):
            x_pred = sess.run(X_pred, feed_dict=fdict)
        else:
            x_pred = None # not necessary, just extra safety

        # Save
        if checkpoint_reached:
            checkpoint_error = sess.run(error, feed_dict=fdict)
            print('  Error {} --> {} = {:.8f}'.format(zx, zy, checkpoint_error))
            #saver.save(sess, model_path + model_name, global_step=step, write_meta_graph=True)

# Save trained variables and session
saver.save(sess, model_path + model_name, global_step=num_iters, write_meta_graph=True)
# END training
# ========================================
print('elapsed time: {}'.format(time.time() - start_time))
X_train = None # reduce memory overhead


#=============================================================================
# EVALUATION
#=============================================================================
# Eval data containers
# ----------------
num_val_batches = NUM_VAL_SAMPLES // batch_size
test_predictions  = np.zeros((num_rs_layers,) + X_test.shape[1:-1] + (channels[-1],)).astype(np.float32)
test_loss = np.zeros((num_val_batches, num_rs_layers)).astype(np.float32)
test_loss_loc = np.zeros((num_val_batches, num_rs_layers)).astype(np.float32)
print('\nEvaluation:\n{}'.format('='*78))
#for j in range(X_test.shape[1]):
for j in range(num_val_batches):
    # Validation cubes
    # ----------------
    #x_in    = X_test[ 0, j:j+1] # (1, n_P, 6)
    #x_truth = X_test[-1, j:j+1]
    p, q = batch_size*j, batch_size*(j+1)

    for z in range(num_rs_layers):
        # Validation cubes
        # ----------------
        x_in    = X_test[z,   p:q] if z == 0 else x_pred
        x_truth = X_test[z+1, p:q]
        rs_in = np.full([batch_size*N*M, 1], redshifts[z+1], dtype=np.float32)
        #rs_Xin = np.full([batch_size*N*M, 1], redshifts[z], dtype=np.float32)
        #rs_Yin = np.full([batch_size*N*M, 1], redshifts[z+1], dtype=np.float32)
        #rs_in = np.concatenate([rs_Xin, rs_Yin], axis=-1)

        # Graph data
        # ----------------
        csr_list = get_list_csr(x_in) # len b list of (N,N) csrs

        # get coo features
        coo_segs = nn.to_coo_batch(csr_list)

        # Feed data to tensors
        # ----------------
        fdict = {X_input: x_in,
                 X_truth: x_truth,
                 COO_seg: coo_segs,
                 RS_in: rs_in,
                 }

        # Get pred based on z index
        # ----------------
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        #print('V batch {:>5}, z {}'.format(j, z))
        x_pred, v_error = sess.run([X_pred, error], feed_dict=fdict)

        # Assign results
        test_predictions[z, p:q] = x_pred
        test_loss[j, z] = v_error
        v_error_loc = sess.run(val_error, feed_dict={X_pred_val: x_pred, X_truth: x_truth})
        test_loss_loc[j, z] = v_error_loc
    #print('{:>3d}: {:.6f}'.format(j, v_error))

# END Validation
# ========================================
# median error
#test_median = np.median(test_loss[:,-1])
#inputs_median = np.median(inputs_loss)
loss_median     = np.median(test_loss, axis=0)
loss_loc_median = np.median(test_loss_loc, axis=0)
#inputs_median = np.median(inputs_loss)
#print('{:<18} median scaled: {:.9f}'.format(model_name, np.median(test_loss[:,-1])))
#print('{:<18} median    loc: {:.9f}'.format(model_name, np.median(test_loss_loc[:,-1])))
rs_steps_tup = [(redshift_steps[i], redshift_steps[i+1]) for i in range(num_rs_layers)]
print('\nEvaluation Median Error Statistics, {:<18}:\n{}'.format(model_name, '='*78))
print('# SCALED LOSS:')
for i, tup in enumerate(rs_steps_tup):
    zx, zy = tup
    print('  {:>2} --> {:>2}: {:.9f}'.format(zx, zy, loss_median[i]))
print('# LOCATION LOSS:')
for i, tup in enumerate(rs_steps_tup):
    zx, zy = tup
    print('  {:>2} --> {:>2}: {:.9f}'.format(zx, zy, loss_loc_median[i]))
print('\nEND EVALUATION, SAVING CUBES AND LOSS\n{}'.format('='*78))

# save loss and predictions
utils.save_loss(loss_path + model_name, test_loss, validation=True)
utils.save_loss(loss_path + model_name + '_locMSE', test_loss_loc, validation=True)
utils.save_test_cube(test_predictions, cube_path, (zX, zY), prediction=True)

#'''
print('Scalars, final values: ') #'coeff_{}_{}'
#for i in range(num_rs_layers):
scalar_tag = 'coeff_{}'.format(0)
scalar_tag2 = 'coeff_{}'.format(1)
scalar_value = get_var(scalar_tag)[0]
scalar_value2 = get_var(scalar_tag2)[0]
#rsa, rsb = redshifts[i], redshifts[i+1]
print('LOC = {:.6f} VEL = {:.6f}'.format(scalar_value, scalar_value2))
#'''
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

''' # Previous, with scalars different for each RS
Training Single-step:
==============================================================================
CHECKPOINT  100:
  Error 0 --> 1 = 1.15358412
  Error 1 --> 2 = 1.76240587
elapsed time: 255.04985117912292

Evaluation:
==============================================================================

Evaluation Median Error Statistics, test_ZG_7-19      :
==============================================================================
# SCALED LOSS:
   7 --> 13: 1.146232486
  13 --> 19: 1.744175792
# LOCATION LOSS:
   7 --> 13: 0.000198614
  13 --> 19: 0.000654184

END EVALUATION, SAVING CUBES AND LOSS
==============================================================================
saved ./Model/test_ZG_7-19/Cubes/X32_7-19_prediction
Scalars, final values:
  1.1212 --> 0.3758 : LOC = -0.035935 VEL = 0.025082
  0.3758 --> 0.0000 : LOC = -0.010543 VEL = 0.010450

'''
