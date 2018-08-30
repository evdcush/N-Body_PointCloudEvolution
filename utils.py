import os, glob, struct, shutil, code, sys, time, argparse
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''' TODO:
# TOP PRIORITY
 - Eventually create a trainer class, or abstract as much training code from
   train to here. Ideally only needing a single train.py.
     - NEED higher-level abstraction and modularity, currently too much
       thrashing and potential for code error
     - Training consistency, for all models, all network types
     - make train.py invariant to:
       - model-type: multi-step or single
       - layer-type: vanilla, shift-inv, or rot-inv
 - In order to abstract train functionality or make a trainer class,
   need to evaluate:
     - Can you instantiate placeholders OUTSIDE of the caller script?
       - must they be placed in train.py for scope reasons? or feeding?

# EVENTUALLY
 - Make a separate evaluation script ('evaluation.py' or whatev), that can
   evaluate a model separate from training script.
     - Then, to use in train.py, just import the respective functions
 - Visualizations:
   - automatic generation of plots for error or whatev,
   - maybe a meta script that maintains all models trained in a spreadsheet or
     npy array, that tracks performance based on num_iters, model type, redshifts
     hyperparameters, etc.
 - Have density-based graph model as another option, like layer-type, for models
   instead of just KNN
'''
#=============================================================================
# Utility classes
#=============================================================================
class AttrDict(dict): # just a dict mutated/accessed by attribute instead index
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__



#=============================================================================
# Globals
#=============================================================================
#------------------------------------------------------------------------------
# Data and pathing
#------------------------------------------------------------------------------
# Redshifts available in dataset
# ========================================
REDSHIFTS = [9.0000, 4.7897, 3.2985, 2.4950, 1.9792, 1.6141, 1.3385,
             1.1212, 0.9438, 0.7955, 0.6688, 0.5588, 0.4620, 0.3758,
             0.2983, 0.2280, 0.1639, 0.1049, 0.0505, 0.0000]

# Data load paths (must be changed for your machine!)
# ========================================
DATA_ROOT_PATH = '/home/evan/Data/nbody_simulations/N_uniform/{}'
DATA_PATH_BINARIES = DATA_ROOT_PATH.format('binaries/run*/xv_dm.z=0{:.4f}') # not in use
DATA_PATH_NPY      = DATA_ROOT_PATH.format('npy_data/X_{:.4f}_.npy')


# Data write paths
# ========================================
# All model save paths
BASE_SAVE_PATH = './Models/{}/'
MODEL_SAVE_PATH   = BASE_SAVE_PATH  + 'Session/'
FILE_SAVE_PATH    = MODEL_SAVE_PATH + 'original_files/'
RESULTS_SAVE_PATH = BASE_SAVE_PATH  + 'Results/'
FILE_SAVE_NAMES = ['utils.py', 'nn.py', 'train_ShiftInv.py', 'train_multi_ShiftInv.py']

# Model data names
CUBE_BASE_NAME = 'X_{}-{}_'
CUBE_NAME_TRUTH = CUBE_BASE_NAME + 'truth'
CUBE_NAME_PRED  = CUBE_BASE_NAME + 'prediction'


#------------------------------------------------------------------------------
# Model naming
#------------------------------------------------------------------------------
# Model and layer names
# ========================================
# Model names
MULTI_STEP  = 'multi-step'
SINGLE_STEP = 'single-step'
MODEL_TYPES = [MULTI_STEP, SINGLE_STEP]
MODEL_BASENAME = '{}_{}_{}' # {model-type}_{layer-type}_{rs1-...-rsN}_{extra-naming}

# Layer names
VANILLA   = 'vanilla'
SHIFT_INV = 'shift-inv'
ROT_INV   = 'rot-inv'
LAYER_TAGS = {VANILLA:'V', SHIFT_INV:'SI', ROT_INV:'RI'}
LAYER_TYPES = [VANILLA, SHIFT_INV, ROT_INV]

# Variable names
# ========================================
# Scope naming
VARIABLE_SCOPE = 'params_{}-{}' # eg. 'params_0-7' for rs 9.0000 --> 1.1212

# Model variable names
WEIGHT_TAG = 'W{}_{}'
BIAS_TAG   = 'B_{}'
SCALAR_TAG = 'T_{}'


#------------------------------------------------------------------------------
# Model variables
#------------------------------------------------------------------------------
# Model params
# ========================================
# RNG seeds
PARAMS_SEED  = 77743196 # Randomly generated seed selected by cross-validation
DATASET_SEED = 12345    # for train/validation data splits

# Network channels
CHANNELS = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 3]
CHANNELS_SHALLOW = [9, 32, 16, 8, 6]

# Layer variables
# ========================================
# Shift-invariant
SHIFT_INV_W_IDX = [1,2,3,4]

# Rotation-invariant
ROTATION_INV_SEGMENTS = ['no-pooling', 'col-depth', 'row-depth',
                         'row-col', 'depth', 'col', 'row', 'all']

# Training variables
# ========================================
LEARNING_RATE = 0.01
NUM_VAL_SAMPLES = 200



#=============================================================================
# TensorFlow Variable inits and gets
#=============================================================================
#------------------------------------------------------------------------------
# Parameter initialization
#------------------------------------------------------------------------------
# Variable inits
# ========================================
def initialize_var(args_init, initializer):
    tf.get_variable(*args_init, dtype=tf.float32, initializer=initializer)


def initialize_weight(name, kdims, restore=False):
    """ initialize weight Variable
    Args:
        name (str): variable name
        kdims tuple(int): kernel sizes (k_in, k_out)
        restore (bool): if restore, then do not user initializer
    """
    args_init = (name, kdims)
    initializer = None if restore else tf.glorot_normal_initializer(None)
    initialize_var(args_init, initializer)


def initialize_bias(name, kdims, restore=False):
    """ biases initialized to be near zero
    Args:
        name (str): variable name
        kdims tuple(int): kernel sizes (k_in, k_out), only k_out used for bias
        restore (bool): if restore, then do not user initializer
    """
    k_out = kdims[-1]
    args_init = (name,)
    if restore:
        initializer = None
        args_init += (k_out,)
    else:
        initializer = tf.ones((k_out,), dtype=tf.float32) * 1e-8
    initialize_var(args_init, initializer)


def initialize_scalars(init_val=0.002, restore=False):
    """ 1D scalars used to scale network outputs """
    initializer = tf.constant([init_val])
    for i in range(2):
        initialize_var((SCALAR_TAG.format(i),), initializer)


# Model parameter init wrappers
# ========================================
def initialize_vanilla_params(kdims, restore=False, **kwargs):
    """ Vanilla layers have 1 bias, 1 weight, scalars"""
    for layer_idx, kdim in enumerate(kdims):
        bname = BIAS_TAG.format(layer_idx)
        Wname = WEIGHT_TAG.format(layer_idx, 0)
        initialize_bias(  bname, kdim, restore=restore)
        initialize_weight(Wname, kdim, restore=restore)
    initialize_scalars(restore=restore)


def initialize_ShiftInv_params(kdims, restore=False, **kwargs):
    """ ShiftInv layers have 1 bias, 4 weights, scalars """
    for layer_idx, kdim in enumerate(kdims):
        initialize_bias(BIAS_TAG.format(layer_idx), kdim, restore=restore)
        for w_idx in SHIFT_INV_W_IDX:
            Wname = WEIGHT_TAG.format(layer_idx, w_idx)
            initialize_weight(Wname, kdim, restore=restore)
    initialize_scalars(restore=restore)


def initialize_RotInv_params(kdims, restore=False, **kwargs):
    # TODO
    assert False

LAYER_INIT_FUNCS = {VANILLA:   initialize_vanilla_params,
                    SHIFT_INV: initialize_ShiftInv_params,
                    ROT_INV:   initialize_RotInv_params}

def initialize_model_params(layer_type, channels, scope,
                            seed=PARAMS_SEED, restore=False, **kwargs):
    """ Initialize model parameters, dispatch based on layer_type
    Args:
        layer_type (str): layer-type ['vanilla', 'shift-inv', 'rot-inv']
        channels list(int): network channels
        scope (str): scope for tf.variable_scope
        seed (int): RNG seed for param inits
        restore (bool): whether new params are initialized, or just placeholders
    """
    # Check layer_type integrity
    assert layer_type in LAYER_TYPES

    # Convert channels to (k_in, k_out) tuples
    kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]

    # Seed and initialize
    tf.set_random_seed(seed)
    #with tf.variable_scope(scope, reuse=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        layer_init_func = LAYER_INIT_FUNCS[layer_type]
        layer_init_func(kdims, restore=restore, **kwargs)
    print('Initialized {} layer parameters'.format(layer_type))



#------------------------------------------------------------------------------
# Parameter getters
#------------------------------------------------------------------------------
# Variable gets
# ========================================
def get_var(name):
    """ Assumes within variable scope """
    return tf.get_variable(name)


def get_weight(layer_idx, w_idx=0):
    name = WEIGHT_TAG.format(layer_idx, w_idx)
    return get_var(name)


def get_bias(layer_idx):
    name = BIAS_TAG.format(layer_idx)
    return get_var(name)


def get_scalars():
    scalars = [get_var(SCALAR_TAG.format(i)) for i in range(2)]
    return scalars


# Layer var get wrappers
# ========================================
"""
Assumed to be within the tf.variable_scope of the respective network funcs
  themselves (called directly), so no dispatching layer wrapper get func
"""
def get_vanilla_layer_vars(layer_idx, **kwargs):
    weight = get_weight(layer_idx)
    bias   = get_bias(layer_idx)
    return weight, bias


def get_ShiftInv_layer_vars(layer_idx, **kwargs):
    weights = []
    for w_idx in SHIFT_INV_W_IDX:
        weights.append(get_weight(layer_idx, w_idx=w_idx))
    bias = get_bias(layer_idx)
    return weights, bias


def get_RotInv_layer_vars(layer_idx, **kwargs):
    # TODO
    assert False


#=============================================================================
# Model save, load utilities
#=============================================================================
""" Note: all save, load utilities are provided by the functions below, but are
    interfaced to the trainer by TrainSaver
"""
# Model name getter
# ========================================
def get_model_name(sess_args):
    """ Consistent model naming format
    Args:
        sess_args (dict): all model and session variables
    """
    # ==== Extract relevant args
    model_name = sess_args.model_name
    suffix = sess_args.name_suffix
    rs_idx = sess_args.rs_idx
    ltype  = sess_args.layer_type
    mtype  = sess_args.model_type

    # ==== Confirm valid types
    assert mtype in MODEL_TYPES and ltype in LAYER_TYPES
    if model_name != MODEL_BASENAME: # if optional model_name used
        return model_name

    # ==== Model name fields
    layer_tag = LAYER_TAGS[ltype] # in ['V', 'SI', 'RI']
    rs_tag = "".join([str(z)+'-' if z != rs_idx[-1] else str(z) for z in rs_idx])
    suff = '_{}'.format(suffix) if len(suffix) > 0 else ''

    # ==== Format model name
    model_name = model_name.format(layer_tag, mtype, rs_tag) + suff
    return model_name



#------------------------------------------------------------------------------
# Standalone save/restore util functions
#------------------------------------------------------------------------------
# Model save utils
# ========================================
def make_dirs(dirs):
    """ Make all directories along paths in dirs """
    for path in dirs:
        if not os.path.exists(path): os.makedirs(path)

def save_files(save_path, files_to_save=FILE_SAVE_NAMES):
    """ Copy project files to directory """
    for f in files_to_save:
        src = './{}'.format(f)
        dst = save_path + f
        shutil.copy(src, dst)

def save_cube(cube, redshifts, save_path, ground_truth=False):
    """ Save data cube """
    rsX, rsY = redshifts
    name = CUBE_NAME_TRUTH if ground_truth else CUBE_NAME_PRED
    name = name.format(rsX, rsY)
    np.save(save_path + name, cube)
    print('Saved cube: {}'.format(name))

def save_error(error, save_path, training=False):
    """ Save model error """
    suffix = 'training' if training else 'validation'
    name = 'error_{}'.format(suffix)
    np.save(save_path + name, error)
    print('Saved {}'.format(name))

def save_params(saver, sess, cur_iter, path, write_meta_graph=True):
    """ Save trained model parameters """
    step = cur_iter + 1
    saver.save(sess, path, global_step=step, write_meta_graph=write_meta_graph)

# Model restore utils
# ========================================
def restore_parameters(saver, sess, save_dir):
    """ restore trained model parameters """
    path = tf.train.get_checkpoint_state(save_dir)
    saver.restore(sess, path.model_checkpoint_path)
    print('Restored trained model parameters from {}'.format(save_dir))



#=============================================================================
# Simulation dataset read/load utilities
#=============================================================================
""" Note: due to long read times and disk space constraints, data is not
    read from binaries.
    Instead, the numpy npy formatted simulation data is what is used regularly
"""
#------------------------------------------------------------------------------
# Dataset (binary) dataset read functions
#------------------------------------------------------------------------------
# Read simulation cubes stored in binary structs
# ========================================
def read_simulation_binaries(file_list, n_P=32):
    """ reads simulation data from binaries and and converts to numpy ndarray
    Args:
        file_list list(str): paths to files
        n_P (int): number of particles base (n_P**3 particles)
            NB: only 32**3 simulation data is used
    Returns: numpy array of data
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


# Interface for binary read func
# ========================================
def load_simulation_cube_binary(redshift, data_path=DATA_PATH_BINARIES):
    """ loads two redshift datasets from proper data directory
    Args:
        redshift (float): redshift value
    """
    glob_paths = glob.glob(data_path.format(redshift))
    X = read_sim(glob_paths).astype(np.float32)
    return X


#------------------------------------------------------------------------------
# Dataset (npy) read functions
#------------------------------------------------------------------------------
# Load simulation cube from npy
# ========================================
def load_simulation_cube_npy(redshift, cat_dim=True):
    """ Loads uniformly timestepped simulation cube stored in npy format
    Note redshift here is true redshift float value
    Args:
        redshift (float): redshift value
        cat_dim (bool): expand a new dim at axis 0 for concatenation
    """
    # Cube dims
    num_cubes = 1000; num_particles = 32**3; num_features = 6;

    # Load cube
    cube_path = DATA_PATH_NPY.format(redshift)
    print('Loading Redshift {:.4f} Cube from: {}'.format(redshift, cube_path[-13:]))
    X = np.load(cube_path).astype(np.float32)
    if cat_dim:
        X = np.expand_dims(X, 0)
    return X


# Load cubes for each redshift
# ========================================
def load_simulation_data(redshift_indices):
    """ Loads uniformly timestep data serialized as np array of np.float32
    Redshift indices are used instead of true float values for ease
    Args:
        redshift_indices list(int): ordered list of indices into REDSHIFTS
    """
    num_rs = len(redshift_indices) # number of cubes to load

    # Load cubes
    rs_idx = redshift_indices[0]
    redshift = REDSHIFTS[rs_idx]
    X = load_simulation_cube_npy(redshift)
    if num_rs == 1:
        return X
    for rs_idx in redshift_indices[1:]:
        redshift = REDSHIFTS[rs_idx]
        X = np.concatenate([X, load_simulation_cube_npy(redshift)], axis=0)
    return X



#=============================================================================
# Data processing and manipulation utils
#=============================================================================
#------------------------------------------------------------------------------
# Normalization
#------------------------------------------------------------------------------
def normalize(X):
    """ Normalize data features
    coordinates are rescaled to be in range [0,1]
    velocities should not be normalized

    Args:
        X (ndarray): data to be normalized, of shape (R, N, D, 6)
    """
    # Rescale coordinate values range from (0,32) -> (0,1)
    X[...,:3] = X[...,:3] / 32.0
    return X


#------------------------------------------------------------------------------
# Data batching and mutation
#------------------------------------------------------------------------------
# Data train/test split
# ========================================
def split_data_validation(X, num_val=NUM_VAL_SAMPLES, seed=DATASET_SEED):
    """ split dataset into training and validation sets

    Args:
        X (ndarray): data of shape (num_redshifts, num_cubes, num_particles, 6)
        num_val_samples (int): size of validation set
    """
    num_cubes = X.shape[1]

    # Seed split for consistency between models (for comparison)
    np.random.seed(seed)
    idx_list = np.random.permutation(num_cubes)
    X_train, X_test = np.split(X[:,idx_list], [-num_val], axis=1)

    return X_train, X_test


# Symmetric, random data shift augmentation
# ========================================
def random_augmentation_shift(batch):
    # Note, ONLY USED FOR VANILLA MODEL!
    """ Randomly augment data by shifting indices
    and symmetrically relocating particles
    Args:
        batch (ndarray): (num_rs, batch_size, N, 6)
    Returns:
        batch (ndarray): randomly shifted data array
    """
    # Random vals
    batch_size = batch.shape[1]
    rands = np.random.rand(6)
    shift = np.random.rand(1,batch_size,1,3)

    # ==== Swap axes
    if rands[0] < .5:
        batch = batch[...,[1,0,2,4,3,5]]
    if rands[1] < .5:
        batch = batch[...,[0,2,1,3,5,4]]
    if rands[2] < .5:
        batch = batch[...,[2,1,0,5,4,3]]
    # ==== Relocate particles
    if rands[3] < .5:
        batch[...,0] = 1 - batch[...,0]
        batch[...,3] = -batch[...,3]
    if rands[4] < .5:
        batch[...,1] = 1 - batch[...,1]
        batch[...,4] = -batch[...,4]
    if rands[5] < .5:
        batch[...,2] = 1 - batch[...,2]
        batch[...,5] = -batch[...,5]
    # ==== Shift particle locations
    batch_coo = batch[...,:3]
    batch_coo += shift

    # ==== Preserve periodic boundary constraints
    gt1 = batch_coo > 1
    batch_coo[gt1] = batch_coo[gt1] - 1
    batch[...,:3] = batch_coo
    return batch


# Data batching
# ========================================
def next_minibatch(X_in, batch_size, data_aug=False):
    """ randomly select sample cubes for training batch
    Args:
        X_in (ndarray): (num_rs, num_cubes, N, 6) data input
        batch_size (int): minibatch size
        data_aug: if data_aug, randomly shift input data (for vanilla)
    Returns:
        batches (ndarray): randomly selected data
    """
    # ==== Get batch indices
    index_list = np.random.choice(X_in.shape[1], batch_size)

    # ==== Get batches, copy for data integrity
    batches = np.copy(X_in[:,index_list])
    if data_aug: # VANILLA ONLY!
        batches = random_augmentation_shift(batches)
    return batches



#=============================================================================
# Model state and information utils
#=============================================================================
#------------------------------------------------------------------------------
# Error prints
#------------------------------------------------------------------------------
def print_checkpoint(step, err, sc_err=None):
    """ Print current training error """
    text = 'Checkpoint {:>5}--> LOC: {:.8f}'.format(step+1, err)
    if sc_err is not None:
        text = text + ', SCA:: {:.6f}'.format(sc_err)
    print(text)


def print_median_validation_loss(rs, err, sc_err=None):
    zx, zy = rs
    err_median = np.median(err)
    print('\nEvaluation Median Error:\n{}'.format('='*78))
    print('# LOCATION LOSS:')
    print('  {:>2} --> {:>2}: {:.9f}'.format(zx, zy, err_median))
    if sc_err is not None:
        sc_err_median = np.median(sc_err)
        print('# SCALED LOSS:')
        print('  {:>2} --> {:>2}: {:.9f}'.format(zx, zy, sc_err_median))



#=============================================================================
# Model utility classes
#=============================================================================
#------------------------------------------------------------------------------
# Save/Restore utils interface class
#------------------------------------------------------------------------------
class TrainSaver:
    """ TrainSaver wraps tf.train.Saver() for session,
        and interfaces all essential save/restore utilities
    """
    def __init__(self, sess_args, always_write_meta=False):
        # Training vars
        self.saver = tf.train.Saver() # must be done AFTER sess.run(tf.global_variables_initializer())
        self.model_name = get_model_name(sess_args)
        self.num_iters = sess_args.num_iters
        self.rs_idx = sess_args.rs_idx
        self.restore = sess_args.restore
        self.always_write_meta = always_write_meta

        # Paths
        self.model_path  = MODEL_SAVE_PATH.format(mname)
        self.result_path = RESULTS_SAVE_PATH.format(mname)
        self.file_path   = FILE_SAVE_PATH.format(mname)
        self.make_model_dirs()

    def initialize_saver(self):
        self.saver = tf.train.Saver()

    def make_model_dirs(self,):
        paths = [self.model_path, self.result_path, self.file_path]
        make_dirs(paths)

    # ==== Restoring
    def restore_model_parameters(self, sess):
        restore_parameters(self.saver, sess, self.model_path)

    # ==== Saving
    def save_model_files(self,):
        path = self.file_path
        save_files(path)

    def save_model_error(self, error, save_path=None, training=False):
        if save_path is None:
            save_path = self.result_path
        save_error(error, save_path, training)

    def save_model_cube(self, cube, save_path=None, ground_truth=False):
        rs = self.rs_idx
        if save_path is None:
            save_path = self.result_path
        save_cube(cube, rs, save_path, ground_truth)

    def save_model_params(self, session, cur_iter):
        is_final_step = cur_iter == self.num_iters
        wr_meta = True if is_final_step else self.always_write_meta
        save_params(self.saver, session, cur_iter, self.model_path, wr_meta)


#------------------------------------------------------------------------------
# Model function args
#------------------------------------------------------------------------------




#=============================================================================
# New utils made for train
#=============================================================================
class Trainer():
    def __init__(self, model, data, evaluation=True):

        if evaluation:jop


class Model():
    pass


class Parser:
    """ Wrapper for argparse parser
    """
    def __init__(self):
        self.p = argparse.ArgumentParser()
        self.add_parse_args()

    def add_parse_args(self,):
        add = self.p.add_argument

        # ==== Data variables
        add('--seed',       '-s', type=int, default=PARAMS_SEED,)
        add('--rs_idx',     '-z', type=int, default=[18,19], nargs='+',)
        add('--model_name', '-m', type=str, default=MODEL_BASENAME,)
        add('--name_suffix','-n', type=str, default='')

        # ==== Model parameter variables
        add('--layer_type', '-l', type=str, default='shift-inv')
        add('--graph_var',  '-k', type=int, default=14,) # type might be float for rad
        add('--channels',   '-c', type=int, default=CHANNELS_SHALLOW, nargs='+')
        add('--var_scope',  '-v', type=str, default=VARIABLE_SCOPE)
        add('--learn_rate', '-a', type=float, default=LEARNING_RATE)

        # ==== Training variables
        add('--num_test',   '-t', type=int, default=NUM_VAL_SAMPLES)
        add('--num_iters',  '-i', type=int, default=2000)
        add('--batch_size', '-b', type=int, default=4)
        add('--restore',    '-r', type=int, default=0,) # bool
        add('--pbc_graph',  '-p', type=int, default=0)
        add('--checkpoint', '-c', type=int, default=100)
        add('--variable',   '-q', type=int, default=0)  # bool, multi-purpose

    def parse_args(self):
        parsed = self.add_interpreted_args(AttrDict(vars(self.p.parse_args())))
        self.args = parsed
        return parsed

    def add_interpreted_args(self, parsed):
        # ==== redshifts
        redshift_idx = parsed.rs_idx
        redshifts = [REDSHIFTS[z] for z in redshift_idx]
        parsed.redshifts = redshifts

        # ==== Model-type
        mtype = SINGLE_STEP
        cat_rs = False
        if len(redshift_idx) > 2:
            mtype = MULTI_STEP
            cat_rs = True
        parsed.model_type = mtype
        parsed.cat_rs = cat_rs

        # ==== var_scope formatting
        vscope = parsed.var_scope
        parsed.var_scope = vscope.format(redshift_idx[0], redshift_idx[-1])
        return parsed


    def print_args(self):
        print('SESSION CONFIG\n{}'.format('='*79))
        margin = len(max(self.args, key=len)) + 1
        for k,v in self.args.items():
            print('{:>{margin}}: {}'.format(k,v, margin=margin))




