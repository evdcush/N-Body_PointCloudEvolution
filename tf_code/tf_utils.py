import os, glob, struct, code, sys, shutil, time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import chainer
import chainer.serializers as serializers

import tensorflow as tf

#=============================================================================
# Globals
#=============================================================================
# dataset
DATA_PATH     = '/home/evan/Data/nbody_simulations/N_{0}/DM*/{1}_dm.z=0{2}000'
DATA_PATH_NPY = '/home/evan/Data/nbody_simulations/nbody_{}.npy'
REDSHIFTS = [6.0, 4.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
RS_TAGS = {6.0:'60', 4.0:'40', 2.0:'20', 1.5:'15', 1.2:'12', 1.0:'10', 0.8:'08', 0.6:'06', 0.4:'04', 0.2:'02', 0.0:'00'}

# rng seeds
PARAMS_SEED  = 77743196 # best consistent performance for graph models, set models do better with 98765
DATASET_SEED = 12345 # for train/validation data splits

# models
GRAPH_CHANNELS = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 3]
SET_CHANNELS   = [6, 32, 128, 256, 128, 32, 256, 16, 3]
LEARNING_RATE = 0.01
NBODY_MODELS = {0:{'channels':   SET_CHANNELS, 'tag': 'S'},
                1:{'channels': GRAPH_CHANNELS, 'tag': 'G'},}

LEARNING_RATE = 0.01

WEIGHT_TAG = 'W_{}'
GRAPH_TAG  = 'Wg_{}'
BIAS_TAG   = 'B_{}'
VAR_SCOPE  = 'params'
VAR_SCOPE_MULTI = 'params_{}'

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# TF-related utils
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

#=============================================================================
# var inits
#=============================================================================
def init_weight(k_in, k_out, name,  seed=None):
    """ initialize weight Variable
    Args:
        k_in, k_out (int): weight sizes
        name (str): variable name
    """
    #std = scale * np.sqrt(2. / k_in)
    #henorm = tf.random_normal((k_in, k_out), stddev=std, seed=seed)
    norm = tf.glorot_normal_initializer(seed=seed)
    tf.get_variable(name, shape=(k_in, k_out), dtype=tf.float32, initializer=norm)

def init_bias(k_in, k_out, name,):
    """ biases initialized to be near zero
    """
    bval = tf.ones((k_out,), dtype=tf.float32) * 1e-8
    tf.get_variable(name, dtype=tf.float32, initializer=bval)

#=============================================================================
# Wrappers
def init_params(channels, graph_model=False, seed=None, var_scope=VAR_SCOPE):
    """ Initialize network parameters
    graph model has extra weight, no bias
    set model has bias
    """
    kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
    with tf.variable_scope(var_scope):
        for idx, ktup in enumerate(kdims):
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            init_weight(*ktup, WEIGHT_TAG.format(idx), seed=seed)
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            if graph_model: # graph
                init_weight(*ktup,  GRAPH_TAG.format(idx), seed=seed)
            else: # set
                init_bias(*ktup, BIAS_TAG.format(idx))

# Multi
def init_params_multi(channels, num_rs, graph_model=False, var_scope=VAR_SCOPE_MULTI, seed=None):
    """ Initialize network parameters
    graph model has extra weight, no bias
    set model has bias
    """
    #kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
    for j in range(num_rs):
        tf.set_random_seed(seed)
        cur_scope = var_scope.format(j)
        init_params(channels, graph_model=graph_model, var_scope=cur_scope, seed=None)
        #print('init_params: {}'.format(cur_scope))
        '''
        with tf.variable_scope(cur_scope):
            init_params(channels, graph_model=graph_model, var_scope=cur_scope)
            for idx, ktup in enumerate(kdims):
                #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
                init_weight(*ktup, WEIGHT_TAG.format(idx))
                if graph_model: # graph
                    init_weight(*ktup,  GRAPH_TAG.format(idx))
                else: # set
                    init_bias(*ktup, BIAS_TAG.format(idx))
        '''
#=============================================================================
# get layer params
def get_layer_vars(layer_idx, var_scope=VAR_SCOPE):
    """ gets variables for layer
    """
    with tf.variable_scope(var_scope, reuse=True):
        W = tf.get_variable(WEIGHT_TAG.format(layer_idx))
        B = tf.get_variable(  BIAS_TAG.format(layer_idx))
    return W, B

def get_layer_vars_graph(layer_idx, var_scope=VAR_SCOPE):
    """ gets variables for graph layer
    """
    with tf.variable_scope(var_scope, reuse=True):
        W  = tf.get_variable(WEIGHT_TAG.format(layer_idx))
        Wg = tf.get_variable(GRAPH_TAG.format(layer_idx))
    return W, Wg

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
    a,b = scale_range
    x_r[:,:3] = (b-a) * (x_r[:,:3] - coo_min) / (coo_max - coo_min) + a

    #vel_mean = np.mean(vel, axis=0)
    #vel_std  = np.std( vel, axis=0)
    #x_r[:,3:] = (x_r[:,3:] - vel_mean) / vel_std
    a,b = (-.5, .5)
    vel_max = np.max(np.max(vel, axis=0), axis=0)
    vel_min = np.min(np.min(vel, axis=0), axis=0)
    x_r[:,3:] = (b-a) * (x_r[:,3:] - vel_min) / (vel_max - vel_min) + a

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
        X[rs_idx] = normalize(X[rs_idx])
    return X


def split_data_validation(X, Y, num_val_samples=200, seed=DATASET_SEED):
    """ split dataset into training and validation sets

    Args:
        X, Y (ndarray): data arrays of shape (num_samples, num_particles, 6)
        num_val_samples (int): size of validation set
    Returns: tuple([X_train, X_val], [Y_train, Y_val])
    """
    num_samples = X.shape[0]
    np.random.seed(seed)
    idx_list    = np.random.permutation(num_samples)
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
    batch_size = batch.shape[1]
    rands = np.random.rand(6)
    shift = np.random.rand(1,batch_size,1,3)
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


def next_minibatch(X_in, batch_size, data_aug=True):
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
        return random_augmentation_shift(batches)

    else:
        return batches

def load_velocity_coefficients(num_particles):
    vel_coeffs = np.load('../Data/velocity_coefficients_{}.npy'.format(num_particles)).item()
    return vel_coeffs


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
    save_pyfiles(model_path)
    return tf_params_save_path, loss_path, cube_path


def save_pyfiles(model_dir):
    """ Save project files to save_path
    For backing up files used for a model
    Args:
        save_path (str): path to save files
    """
    save_path = model_dir + '.original_files/'
    make_dirs([save_path])
    file_names = ['tf_train.py', 'tf_utils.py', 'tf_nn.py']
    for fname in file_names:
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

def save_model(model, optimizer, save_name):
    """ Save model and optimizer parameters
    Model has fwd computation graph and weights saved
    Optimizer has current weights from momentum vectors saved
    Args:
        model (chainer.Chain): a model with weights to save
        optimizer (chainer.optimizers): optimizer
    """
    serializers.save_npz('{}{}'.format(save_name, '.model'),         model)
    serializers.save_npz('{}{}'.format(save_name, '.optimizer'), optimizer)
    print('Saved model and optimizer at {}'.format(save_name))

def save_test_cube(x, cube_path, rs, prediction=False):
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    if prediction:
        rs_tag = '{}-{}'.format(*rs) # (zX, zY)
        ptag   = 'prediction'
        save_cube(x, cube_path, rs_tag, ptag)
    else:
        for i in range(x.shape[0]): # 2
            rs_tag = '{}'.format(rs[i])
            ptag   = 'data'
            save_cube(x[i], cube_path, rs_tag, ptag)

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

