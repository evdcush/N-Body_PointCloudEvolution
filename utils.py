import os, glob, struct, code, sys, shutil, time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

#=============================================================================
# Globals
#=============================================================================
# dataset
DATA_PATH     = '/home/evan/Data/nbody_simulations/N_{0}/DM*/{1}_dm.z=0{2}000'
DATA_PATH_NPY = '/home/evan/Data/nbody_simulations/nbody_{}.npy'
DATA_PATH_THINKPAD = '/home/evan/Data/nbody_simulations/npy_data/X_{:.4f}_.npy'

REDSHIFTS = [6.0, 4.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
RS_TAGS = {6.0:'60', 4.0:'40', 2.0:'20', 1.5:'15', 1.2:'12', 1.0:'10',
           0.8:'08', 0.6:'06', 0.4:'04', 0.2:'02', 0.0:'00'}

# dataset uniform timestep
DATA_PATH_ZUNI     = '/home/evan/Data/nbody_simulations/N_uniform/run*/xv_dm.z=0{}'
DATA_PATH_ZUNI_NPY = '/home/evan/Data/nbody_simulations/N_uniform/npy_data/X_{:.4f}_.npy'
REDSHIFTS_ZUNI = [9.0000, 4.7897, 3.2985, 2.4950, 1.9792, 1.6141, 1.3385,
                 1.1212, 0.9438, 0.7955, 0.6688, 0.5588, 0.4620, 0.3758,
                 0.2983, 0.2280, 0.1639, 0.1049, 0.0505, 0.0000]
SAVE_FILE_NAMES = ['utils.py', 'nn.py', 'z_train.py', 'z_rad_train.py', 'train_ShiftInv.py',]

# rng seeds
PARAMS_SEED  = 77743196 # for graph, set models do better with 98765
DATASET_SEED = 12345 # for train/validation data splits

# models
GRAPH_CHANNELS = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 3]
SET_CHANNELS   = [6, 32, 128, 256, 128, 32, 256, 16, 3]
LEARNING_RATE = 0.01
NBODY_MODELS = {0:{'channels':   SET_CHANNELS, 'tag': 'S'},
                1:{'channels': GRAPH_CHANNELS, 'tag': 'G'},}

LEARNING_RATE = 0.01
NUM_VAL_SAMPLES = 200

# variable tags
WEIGHT_TAG = 'W_{}'
MULTI_WEIGHT_TAG = 'W{}_{}'
BIAS_TAG   = 'B_{}'
VEL_COEFF_TAG = 'V'

# variable scope
VAR_SCOPE = 'params_{}-{}' # formerly just 'params'


#GRAPH_TAG  = 'Wg_{}'
#VAR_SCOPE_MULTI = 'params_{}'
#VAR_SCOPE_SINGLE_MULTI = 'params_{}-{}'
#AGG_PSCOPE = 'params_agg'

SHIFT_INV_W_IDX = [1,2,3,4]

# ROTATION
SEGNAMES_3D = ['no-pooling', 'col-depth', 'row-depth', 'row-col', 'depth', 'col', 'row', 'all']

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# TF-related utils
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

#=============================================================================
# tf.Variable inits, for model parameters
#=============================================================================
def init_weight(k_in, k_out, name, ShiftInv_rescale=None, seed=None, restore=False):
    """ initialize weight Variable
    Args:
        k_in, k_out (int): kernel sizes
        name (str): variable name"
        seed (int): random seed
        restore: if restore, then do not user initializer
    """
    #std = scale * np.sqrt(2. / k_in)
    #henorm = tf.random_normal((k_in, k_out), stddev=std, seed=seed)
    var_args = (name,)
    if restore:
        init = None
        var_args = var_args + ((k_in, k_out),)
    else:
        if ShiftInv_rescale is not None:
            #std = tf.sqrt(2. / (k_in+k_out))
            std = 0.001
            #init = tf.random_normal((k_in,k_out), stddev=std, seed=seed) / ShiftInv_rescale
            init = tf.truncated_normal_initializer(stddev=std)
            var_args = var_args + ((k_in, k_out),)
            #init = np.random.rand(k_in, k_out).astype(np.float32) - 0.5
        else:
            init = tf.glorot_normal_initializer(seed=seed)
            var_args = var_args + ((k_in, k_out),)
    tf.get_variable(*var_args, dtype=tf.float32, initializer=init)

def init_bias(k_in, k_out, name, restore=False):
    """ biases initialized to be near zero
    Args:
        k_in, k_out (int): kernel sizes  (only k_out needed for bias)
        name (str): variable name
        restore: if restore, then do not user initializer
    """
    if not restore:
        init = tf.ones((k_out,), dtype=tf.float32) * 1e-8
        #init = tf.zeros((k_out,), dtype=tf.float32)
        tf.get_variable(name, dtype=tf.float32, initializer=init)
    else:
        tf.get_variable(name, (k_out,), dtype=tf.float32, initializer=None)

def init_vel_coeff(restore=False, vinit=None):
    """ scalar weight used in skip connection with velocity, approximates
    timestep
    Note: get_variable does not accept shape when using concrete values

    Args:
        restore: if restore, learned values will be loaded so do not initialize
        vinit: initial weight value
    """
    args = (VEL_COEFF_TAG,)
    if restore:
        init = None
        args = args + ((1,))
    else:
        if vinit is not None:
            init = tf.constant([vinit])
        else:
            init = tf.random_uniform_initializer(0,1)
            args = args + ((1,))
    tf.get_variable(*args, dtype=tf.float32, initializer=init)

MCOEFFTAG = 'coeff_{}'
MCOEFFTAG2 = 'coeff_{}_{}'

def init_coeff_multi(num_rs, restore=False, vinit=0.002):
    for i in range(num_rs):
        tag = MCOEFFTAG.format(i)
        tf.get_variable(tag, dtype=tf.float32, initializer=tf.constant([vinit]))

def get_scoped_coeff_multi(idx):
    return tf.get_variable(MCOEFFTAG.format(idx))

def init_coeff_multi2(num_rs, restore=False, vinit=0.002):
    num_coeff = 2
    for i in range(num_rs):
        for j in range(num_coeff):
            tag = MCOEFFTAG2.format(i, j)
            args = (tag,)
            if restore:
                init = None
                args = args + ((1,))
            else:
                init = tf.constant([vinit])
            tf.get_variable(*args, dtype=tf.float32, initializer=init)

def get_scoped_coeff_multi2(idx):
    t0 = tf.get_variable(MCOEFFTAG2.format(idx,0))
    t1 = tf.get_variable(MCOEFFTAG2.format(idx,1))
    return t0, t1

# Model init wrappers ========================================================

# Single-step
def init_params(channels, var_scope=VAR_SCOPE, vcoeff=False, seed=None, restore=False):
    """ Initialize parameters for model
    Args:
        channels (list int): list of channel sizes
        var_scope (str): variable scope for variables (basically what prefixes var names)
    """
    # get (k_in, k_out) tuples from channels
    kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
    with tf.variable_scope(var_scope):
        # initialize variables for each layer
        for idx, ktup in enumerate(kdims):
            init_weight(*ktup, WEIGHT_TAG.format(idx), seed=seed, restore=restore)
            init_bias(  *ktup,   BIAS_TAG.format(idx), restore=restore)
        if vcoeff: # scalar weight for simulating timestep, only one
            init_vel_coeff(restore)

# shift-inv params
def init_ShiftInv_params(channels, var_scope, rescale=None, vcoeff=False, restore=False, seed=None):
    """ Init parameters for 'new' perm-equivariant, shift-invariant model
    For every layer in this model, there are 4 weights and 1 bias
        W1: (k, q) no-pooling
        W2: (k, q) pooling rows
        W3: (k, q) pooling cols
        W4: (k, q) pooling all
        B: (q,) bias
    """
    vinit = 0.002 # best vcoeff constant for 15-19 redshifts
    # get (k_in, k_out) tuples from channels
    kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]

    with tf.variable_scope(var_scope):
        # initialize variables for each layer
        for layer_idx, ktup in enumerate(kdims):
            init_bias(*ktup, BIAS_TAG.format(layer_idx), restore=restore) # B
            for w_idx in SHIFT_INV_W_IDX: # just [1,2,3,4]
                init_weight(*ktup, MULTI_WEIGHT_TAG.format(layer_idx, w_idx),
                            ShiftInv_rescale=rescale, restore=restore, seed=seed)
        if vcoeff:
            init_vel_coeff(restore, vinit)

# Multi-step params for aggregate model
def init_params_multi(channels, num_rs, var_scope=VAR_SCOPE, seed=None, restore=False):
    """ Initialize network parameters for multi-step model, for predicting
    across multiple redshifts. (eg 6.0 -> 4.0 -> ... -> <target rs>)
    Multi-step model is network where each layer is a single-step model
    """
    for j in range(num_rs):
        tf.set_random_seed(seed) # ensure each sub-model has same weight init
        cur_scope = var_scope.format(j) #
        init_params(channels, var_scope=cur_scope, restore=restore)

#=============================================================================
# var gets
#=============================================================================
def get_layer_vars(layer_idx, var_scope=VAR_SCOPE):
    with tf.variable_scope(var_scope, reuse=True):
        W = tf.get_variable(WEIGHT_TAG.format(layer_idx))
        B = tf.get_variable(  BIAS_TAG.format(layer_idx))
    return W, B

def get_ShiftInv_layer_vars(layer_idx, var_scope=VAR_SCOPE):
    layer_vars = []
    with tf.variable_scope(var_scope, reuse=True):
        for w_idx in SHIFT_INV_W_IDX:
            layer_vars.append(tf.get_variable(MULTI_WEIGHT.format(layer_idx, w_idx)))
        layer_vars.append(tf.get_variable(BIAS_TAG.format(layer_idx)))
    return layer_vars # [W1, W2, W3, W4, B]

def get_vcoeff(var_scope):
    with tf.variable_scope(var_scope, reuse=True):
        vel_coeff = tf.get_variable(VEL_COEFF_TAG)
    return vel_coeff

#=============================================================================
# var gets, assumes caller in proper variable_scope
#=============================================================================
def get_scoped_layer_vars(layer_idx):
    W = tf.get_variable(WEIGHT_TAG.format(layer_idx))
    B = tf.get_variable(  BIAS_TAG.format(layer_idx))
    return W, B

def get_scoped_ShiftInv_layer_vars(layer_idx):
    #layer_vars = []
    #for w_idx in SHIFT_INV_W_IDX:
    #    layer_vars.append(tf.get_variable(MULTI_WEIGHT_TAG.format(layer_idx, w_idx)))
    #layer_vars.append(tf.get_variable(BIAS_TAG.format(layer_idx)))
    #return layer_vars # [W1, W2, W3, W4, B]

    # TRYING EXPLICIT GETS iNSTEAD OF LIST, just in case
    # riskier, careful of var name to W_idx
    W1 = tf.get_variable(MULTI_WEIGHT_TAG.format(layer_idx, 1))
    W2 = tf.get_variable(MULTI_WEIGHT_TAG.format(layer_idx, 2))
    W3 = tf.get_variable(MULTI_WEIGHT_TAG.format(layer_idx, 3))
    W4 = tf.get_variable(MULTI_WEIGHT_TAG.format(layer_idx, 4))

    B  = tf.get_variable(BIAS_TAG.format(layer_idx))
    return W1, W2, W3, W4, B

def get_scoped_RotInv_weight(layer_idx, w_idx):
    W = tf.get_variable(MULTI_WEIGHT_TAG.format(layer_idx, w_idx))
    return W

def get_scoped_bias(layer_idx):
    B = tf.get_variable(BIAS_TAG.format(layer_idx))
    return B

def get_scoped_vcoeff():
    vcoeff = tf.get_variable(VEL_COEFF_TAG)
    return vcoeff


#=============================================================================
# graph save and restore
#=============================================================================
def load_graph(sess, save_dir):
    saver = tf.train.Saver()
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    path = tf.train.get_checkpoint_state(save_dir)
    saver.restore(sess, path.model_checkpoint_path)


#=============================================================================
'''
There were a few things wrong with this function, and it's usage at caller level:
 - firstly, it was only loading pretrained weights from the first RS tuple (3-7)
 - but it was also assigning these params to the final RS tuple (15-19)
    So both only one set of trained params was being loaded, and it was being
    loaded to the wrong redshift model
 - take a look at the git diff before the commit tagged with 'UTILS.LMG'
   shit was fucked

# Have confirmed now that variables do change value after saver.restore
'''
#=============================================================================
def load_multi_graph(sess, vscopes, num_layers, save_paths, vel_coeff=False):
    #for vidx, vscope in enumerate(vscopes):
    for spath, vscope in zip(save_paths, vscopes):
        sdict = {}
        for layer_idx in range(num_layers):
            wtag = vscope + '/' + WEIGHT_TAG.format(layer_idx)
            btag = vscope + '/' +   BIAS_TAG.format(layer_idx)
            W, B = get_layer_vars(layer_idx, vscope)
            sdict[wtag] = W
            sdict[btag] = B

            if vel_coeff:
                vtag = vscope + '/' + VEL_COEFF_TAG
                sdict[vtag] = get_vel_coeff(vscope)
        print('loading pretrained params for scope {} @ {}'.format(vscope, spath))
        saver = tf.train.Saver(sdict)
        path = tf.train.get_checkpoint_state(spath)
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        saver.restore(sess, path.model_checkpoint_path)
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# END TF-related utils
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

#=============================================================================
# Loading utils
#=============================================================================
def read_sim(file_list, n_P):
    """ reads simulation data from disk and returns

    Args:
        file_list: (list<str>) paths to files
        n_P: (int) number of particles base (n_P**3 particles)
    """
    num_particles = n_P**3
    dataset = []
    for file_name in file_list:
        this_set = []
        with open(file_name, "rb") as f:
            for i in range(num_particles*6):
                s = struct.unpack('=f',f.read(4))
                this_set.append(s[0])
        dataset.append(this_set)
    dataset = np.array(dataset).reshape([len(file_list),num_particles,6])
    return dataset

def load_datum(n_P, redshift, normalize_data=False):
    """ loads two redshift datasets from proper data directory

    Args:
        redshift: (float) redshift
        n_P: (int) base of number of particles (n_P**3 particles)
    """
    N_P = 10000 if n_P == 32 else 1000
    glob_paths = glob.glob(DATA_PATH.format(N_P, 'xv', redshift))
    X = read_sim(glob_paths, n_P)
    if normalize_data:
        X = normalize(X)
    return X

def load_data(n_P, *args, **kwargs):
    """ loads datasets from proper data directory
    # note: this function is redundant

    Args:
        n_P: (int) base of number of particles (n_P**3 particles)
    """
    data = []
    for redshift in args:
        x = load_datum(n_P, redshift, **kwargs)
        data.append(x)
    return data

def load_npy_data(n_P, redshifts=None, normalize=False):
    """ Loads data serialized as numpy array of np.float32
    Args:
        n_P: base number of particles (16 or 32)
        redshifts (tuple): tuple of redshifts
    """
    assert n_P in [16, 32]
    X = np.load(DATA_PATH_NPY.format(n_P)) # (11, N, n_P, 6)
    if redshifts is not None:
        zX, zY = redshifts
        rs_start  = REDSHIFTS.index(zX)
        rs_target = REDSHIFTS.index(zY)
        X = X[[rs_start, rs_target]] # (2, N, n_P, 6)
    if normalize:
        X = normalize_fullrs(X)
    return X

def load_rs_old_npy_data(redshifts, norm_coo=False, norm_vel=False):
    """ Loads new uniformly timestep data serialized as np array of np.float32
    Args:
        redshifts (list int): list of indices into redshifts in order
        norm_coo: normalize coordinate values to [0,1]
        norm_vel: normalize vel values (only norm'd if coord norm'd)
    """
    n_P = 32
    X_all_rs = load_npy_data(n_P, normalize=False)[redshifts]
    num_rs, N, M, D = X_all_rs.shape
    X = np.zeros((num_rs, N, M, D+1)).astype(np.float32)
    X[...,:-1] = X_all_rs
    X_all_rs = None # reduce memory overhead

    for idx, z_idx in enumerate(redshifts):
        X[idx,:,:,-1] = REDSHIFTS[z_idx]
    if norm_coo:
        X[...,:3] = X[...,:3] / n_P
    return X

def thinkpadx201_load_npy(redshifts, norm_coo=False, norm_vel=False):
    """ Loads new uniformly timestep data serialized as np array of np.float32
    Args:
        redshifts (list int): list of indices into redshifts in order
        norm_coo: normalize coordinate values to [0,1]
        norm_vel: normalize vel values (only norm'd if coord norm'd)
    """
    num_rs = len(redshifts)
    N = 1000
    M = 16**3
    D = 7
    X = np.zeros((num_rs, N, M, D)).astype(np.float32)
    for idx, z_idx in enumerate(redshifts):
        z_rs   = REDSHIFTS[z_idx]
        z_path = DATA_PATH_THINKPAD.format(z_rs)
        print('LD: {}'.format(z_path[-13:]))
        X[idx, :, :, :-1] = np.load(z_path)
        X[idx, :, :,-1] = z_rs
    if norm_coo:
        X[...,:3] = X[...,:3] / 32.0
    return X


def load_rs_npy_data(redshifts, norm_coo=False, norm_vel=False, old_dataset=False):
    """ Loads new uniformly timestep data serialized as np array of np.float32
    Args:
        redshifts (list int): list of indices into redshifts in order
        norm_coo: normalize coordinate values to [0,1]
        norm_vel: normalize vel values (only norm'd if coord norm'd)
    """
    if old_dataset:
        return load_rs_old_npy_data(redshifts, norm_coo)
    else:
        return load_zuni_npy_data(redshifts, norm_coo)


#=============================================================================
# NEW DATA UTILS, uniformly displaced
#=============================================================================
def load_zuni_datum(redshift):
    """ loads a single redshift datum from uniformly timestepped redshift data

    Args:
        redshift: (float) redshift
    """
    n_P = 32
    assert redshift in REDSHIFTS_ZUNI
    redshift_str = '{:.4f}'.format(redshift)
    glob_paths = glob.glob(DATA_PATH_ZUNI.format(redshift_str))
    X = read_sim(glob_paths, n_P).astype(np.float32)
    return X

def load_zuni_npy_data(redshifts=None, norm_coo=False, norm_vel=False):
    """ Loads new uniformly timestep data serialized as np array of np.float32
    Args:
        redshifts (list int): list of indices into redshifts in order
        norm_coo: normalize coordinate values to [0,1]
        norm_vel: normalize vel values (only norm'd if coord norm'd)
    """
    if redshifts is None:
        redshifts = list(range(len(REDSHIFTS_ZUNI))) # copy
    num_rs = len(redshifts)
    S = 1000
    N = 32**3
    D = 7
    X = np.zeros((num_rs, S, N, D)).astype(np.float32)
    for idx, z_idx in enumerate(redshifts):
        z_rs   = REDSHIFTS_ZUNI[z_idx]
        z_path = DATA_PATH_ZUNI_NPY.format(z_rs)
        print('LD: {}'.format(z_path[-13:]))
        X[idx,:,:,:-1] = np.load(z_path)
        X[idx,:,:,-1] = z_rs
    if norm_coo:
        X[...,:3] = X[...,:3] / 32.0
        assert np.max(X[...,:3]) <= 1.0
        assert np.min(X[...,:3]) >= 0.0
    return X


#=============================================================================
# Data utils
#=============================================================================

def normalize(X_in, scale_range=(0,1)):
    """ Normalize data features
    coordinates are rescaled to be in range [0,1]
    velocities are normalized to zero mean and unit variance

    Args:
        X_in (ndarray): data to be normalized, of shape (N, D, 6)
        scale_range   : range to which coordinate data is rescaled
    """
    x_r = np.reshape(X_in, [-1,6])
    coo, vel = np.split(x_r, [3], axis=-1)

    coo_min = np.min(coo, axis=0)
    coo_max = np.max(coo, axis=0)
    a,b = scale_range
    x_r[:,:3] = (b-a) * (x_r[:,:3] - coo_min) / (coo_max - coo_min) + a

    vel_mean = np.mean(vel, axis=0)
    vel_std  = np.std( vel, axis=0)
    x_r[:,3:] = (x_r[:,3:] - vel_mean) / vel_std

    X_out = np.reshape(x_r,X_in.shape).astype(np.float32) # just convert to float32 here
    return X_out

def normalize_rescale_vel(X_in, scale_range=(0,1)):
    """ Normalize data features
    coordinates are rescaled to be in range [0,1]
    velocities are normalized to zero mean and unit variance

    Args:
        X_in (ndarray): data to be normalized, of shape (N, D, 6)
        scale_range   : range to which coordinate data is rescaled
    """
    x_r = np.reshape(X_in, [-1,6])
    coo, vel = np.split(x_r, [3], axis=-1)

    coo_min = np.min(coo, axis=0)
    coo_max = np.max(coo, axis=0)
    #a,b = scale_range
    a,b = (0, 1)
    x_r[:,:3] = (b-a) * (x_r[:,:3] - coo_min) / (coo_max - coo_min) + a

    # PREVIOUS velocity normalization
    #vel_mean = np.mean(vel, axis=0)
    #vel_std  = np.std( vel, axis=0)
    #x_r[:,3:] = (x_r[:,3:] - vel_mean) / vel_std

    # RESCALE velocity to be within scale_range
    a,b = scale_range
    #vel_max = np.max(np.max(vel, axis=0), axis=0)
    #vel_min = np.min(np.min(vel, axis=0), axis=0)
    vel_max = np.max(vel)
    vel_min = np.min(vel)
    x_r[:,3:] = (x_r[:,3:] - vel_min) / (vel_max - vel_min)

    X_out = np.reshape(x_r,X_in.shape).astype(np.float32) # just convert to float32 here
    return X_out

def normalize_fullrs(X, scale_range=(0,1)):
    """ Normalize data features, for full data array of redshifts
    coordinates are rescaled to be in range [0,1]
    velocities are normalized to zero mean and unit variance

    Args:
        X_in (ndarray): data to be normalized, of shape (rs, N, D, 6)
        scale_range   : range to which coordinate data is rescaled
    """
    for rs_idx in range(X.shape[0]):
        X[rs_idx] = normalize_rescale_vel(X[rs_idx])
    return X


def split_data_validation(X, Y, num_val_samples=NUM_VAL_SAMPLES, seed=DATASET_SEED):
    """ split dataset into training and validation sets

    Args:
        X, Y (ndarray): data arrays of shape (num_samples, num_particles, 6)
        num_val_samples (int): size of validation set
    Returns: tuple([X_train, X_val], [Y_train, Y_val])
    """
    num_samples = X.shape[0]
    np.random.seed(seed)
    idx_list = np.random.permutation(num_samples)
    X = np.split(X[idx_list], [-num_val_samples])
    Y = np.split(Y[idx_list], [-num_val_samples])
    return X, Y

def split_data_validation_combined(X, num_val_samples=200, seed=DATASET_SEED):
    """ split dataset into training and validation sets

    Args:
        X (ndarray): data arrays of shape (num_rs, num_samples, num_particles, 6)
        num_val_samples (int): size of validation set
    """
    np.random.seed(seed)
    idx_list = np.random.permutation(X.shape[1])
    X = X[:,idx_list]
    X_train = X[:, :-num_val_samples]
    X_val   = X[:, -num_val_samples:]
    return X_train, X_val

def random_augmentation_shift(batch):
    """ Randomly augment data by shifting indices
    and symmetrically relocating particles
    Args:
        batch (ndarray): (num_rs, batch_size, D, 6)
    Returns:
        batch (ndarray): randomly shifted data array
    """
    #batch_size = batch.shape[1]
    rands = np.random.rand(6)
    if batch.ndim > 3: # there is rs dim
        batch_size = batch.shape[1]
        shift = np.random.rand(1,batch_size,1,3)
    else:
        batch_size = batch.shape[0]
        shift = np.random.rand(batch_size,1,3)
    # shape (11, bs, n_P, 6)
    if rands[0] < .5:
        batch = batch[...,[1,0,2,4,3,5]]
    if rands[1] < .5:
        batch = batch[...,[0,2,1,3,5,4]]
    if rands[2] < .5:
        batch = batch[...,[2,1,0,5,4,3]]
    if rands[3] < .5:
        batch[...,0] = 1 - batch[...,0]
        batch[...,3] = -batch[...,3]
    if rands[4] < .5:
        batch[...,1] = 1 - batch[...,1]
        batch[...,4] = -batch[...,4]
    if rands[5] < .5:
        batch[...,2] = 1 - batch[...,2]
        batch[...,5] = -batch[...,5]
    batch_coo = batch[...,:3]
    batch_coo += shift
    gt1 = batch_coo > 1
    batch_coo[gt1] = batch_coo[gt1] - 1
    batch[...,:3] = batch_coo
    return batch


def next_minibatch(X_in, batch_size, data_aug=False):
    """ randomly select samples for training batch

    Args:
        X_in (ndarray): (num_rs, S, N, 6) data input
        batch_size (int): minibatch size
        data_aug: if data_aug, randomly shift input data
    Returns:
        batches (ndarray): randomly selected and shifted data
    """
    index_list = np.random.choice(X_in.shape[1], batch_size)
    batches = np.copy(X_in[:,index_list])
    if data_aug:
        batches = random_augmentation_shift(batches)

    #assert np.all(batches[0,:,:,-1] > batches[1,:,:,-1])
    return batches


def next_zuni_minibatch(X_in, batch_size, data_aug=True):
    """ randomly select samples for training batch

    Args:
        X_in (ndarray): (num_rs, N, D, 6) data input
        batch_size (int): minibatch size
        data_aug: if data_aug, randomly shift input data
    Returns:
        batches (ndarray): randomly selected and shifted data
    """
    index_list = np.random.choice(X_in.shape[1], batch_size)
    batches = X_in[:,index_list]
    if data_aug:
        batches[...,:-1] = random_augmentation_shift(batches[...,:-1])
    return batches
#=============================================================================
# Saving utils
#=============================================================================
def make_dirs(dirs):
    """ Make directories based on paths in dirs
    Args:
        dirs (list): list of paths of dirs to create
    """
    for path in dirs:
        if not os.path.exists(path): os.makedirs(path)

def make_save_dirs(model_dir, model_name):
    """ Make save directories for saving:
        - model hyper parameters
        - loss data
        - cube data
    Args:
        model_dir (str): the root path for saving model
        model_name (str): name for model
    Returns: (model_path, loss_path, cube_path)
    """
    model_path = '{}{}/'.format(model_dir, model_name)
    tf_params_save_path = model_path + 'Session/'
    loss_path  = model_path + 'Loss/'
    cube_path  = model_path + 'Cubes/'
    make_dirs([tf_params_save_path, loss_path, cube_path]) # model_dir lower dir, so automatically created
    #save_pyfiles(model_path)
    return tf_params_save_path, loss_path, cube_path


def save_pyfiles(model_dir):
    """ Save project files to save_path
    For backing up files used for a model
    Args:
        save_path (str): path to save files
    """
    save_path = model_dir + '.original_files/'
    make_dirs([save_path])
    for fname in SAVE_FILE_NAMES:
        src = './{}'.format(fname)
        dst = '{}{}'.format(save_path, fname)
        shutil.copyfile(src, dst)
        print('saved {} to {}'.format(src, dst))


def get_model_name(dparams, mtype, vel_coeff, save_prefix):
    """ Consistent model naming format
    Model name examples:
        'GL_32_12-04': GraphModel|WithVelCoeff|32**3 Dataset|redshift 1.2->0.4
        'S_16_04-00': SetModel|16**3 Dataset|redshift 0.4->0.0
    """
    n_P, rs = dparams
    zX = RS_TAGS[rs[0]]
    zY = RS_TAGS[rs[1]]

    model_tag = NBODY_MODELS[mtype]['tag']
    vel_tag = 'L' if vel_coeff is not None else ''

    model_name = '{}{}_{}_{}-{}'.format(model_tag, vel_tag, n_P, zX, zY)
    if save_prefix != '':
        model_name = '{}_{}'.format(save_prefix, model_name)
    return model_name

def get_zuni_model_name(mtype, zX, zY, save_prefix):
    """ Consistent model naming format
    Model name examples:
        'GL_32_12-04': GraphModel|WithVelCoeff|32**3 Dataset|redshift 1.2->0.4
        'S_16_04-00': SetModel|16**3 Dataset|redshift 0.4->0.0
    """
    #n_P, rs = dparams
    #zX = RS_TAGS[rs[0]]
    #zY = RS_TAGS[rs[1]]

    model_tag = NBODY_MODELS[mtype]['tag']
    #vel_tag = 'L' if vel_coeff is not None else ''

    #model_name = '{}{}_{}_{}-{}'.format(model_tag, vel_tag, n_P, zX, zY)
    #model_name = 'ZG_90-00'
    model_name = 'Z{}_{}-{}'.format(model_tag, zX, zY)
    if save_prefix != '':
        model_name = '{}_{}'.format(save_prefix, model_name)
    return model_name


def save_test_cube(x, cube_path, rs, prediction=False):
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    '''
    if prediction:
        rs_tag = '{}-{}'.format(*rs) # (zX, zY)
        ptag   = 'prediction'
        save_cube(x, cube_path, rs_tag, ptag)
    else:
        for i in range(x.shape[0]): # 2
            rs_tag = '{}'.format(rs[i])
            ptag   = 'data'
            save_cube(x[i], cube_path, rs_tag, ptag)
    '''
    rs_tag = '{}-{}'.format(*rs) # (zX, zY)
    ptag = 'prediction' if prediction else 'true'
    save_cube(x, cube_path, rs_tag, ptag)


def save_cube(x, cube_path, rs_tag, ptag):
    """ Save validation data
    """
    num_particles = 16 if x.shape[-2] == 4096 else 32
    # eg X32_0.6-0.0_val_prediction.npy'
    val_fname = 'X{}_{}_{}'.format(num_particles, rs_tag, ptag)
    save_path = '{}{}'.format(cube_path, val_fname)
    np.save(save_path, x)
    print('saved {}'.format(save_path))

def save_loss(save_path, data, validation=False):
    save_name = '_loss_validation' if validation else '_loss_train'
    np.save(save_path + save_name, data)

def plot_3D_pointcloud(xt, xh, j, pt_size=(.9,.9), colors=('b','r'), fsize=(12,12), xin=None):
    xt_x, xt_y, xt_z = np.split(xt[...,:3], 3, axis=-1)
    xh_x, xh_y, xh_z = np.split(xh[...,:3], 3, axis=-1)

    fig = plt.figure(figsize=fsize)
    ax = fig.gca(projection='3d')
    ax.scatter(xt_x[j], xt_y[j], xt_z[j], s=pt_size[0], c=colors[0])
    ax.scatter(xh_x[j], xh_y[j], xh_z[j], s=pt_size[1], c=colors[1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig
