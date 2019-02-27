"""
utils module handles everything related to data/datasets, RW stuff, and
session initialization

Datasets
========
You must adjust the data path variables to point to the location of your
data
I keep the datasets in a directory in my home folder, '.Data/nbody_simulations'

"""
import os
import sys
import glob
import code
import time
import random
import argparse
import datetime
from collections import NamedTuple

import yaml
import numpy as np
import tensorflow as tf

"""
CURRENT GOALS
-------------
Turn utils into a monolithic script that does nearly everything
besides tf.nn and pyplot stuff

Splitting the utils stuff into a module ended up creating more overhead,
and the yaml config script is too static for actual usage

So:
* Consolidate data_loader, initializer, parser, saver into this script
* consolidate config.yml here (ez)
* make all new, better argparse.parser
* clean up train scripts
* clean up nn

"""
# Handy Helpers
# =============

def get_date():
    # OUT: (YYYY, M, D), eg (2019, 2, 27)
    date  = datetime.datetime.now().date()
    year  = str(date.year)
    month = str(date.month)
    day   = str(date.day)
    return (year, month, day)

class AttrDict(dict):
    # dot accessible dict
    # NB: not pickleable
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# yaml RW funcs
# -------------
def R_yml(fname):
    with open(fname) as file:
        return yaml.load(file)

def W_yml(fname, obj):
    with open(fname, 'w') as file:
        yaml.dump(obj, file, default_flow_style=False)




#-----------------------------------------------------------------------------#
#                                   PATHING                                   #
#-----------------------------------------------------------------------------#
"""
These are the variables you need to adjust to point towards your data
"""

# Base paths
# ==========
_home = os.environ['HOME'] # ~, your home directory, eg '/home/evan'
_data = _home + '/.Data'   # '/home/$USER/.Data'
_project = os.path.abspath(os.path.dirname(__file__)) # '/path/to/nbody_project'

# data dirs
data_path = _data + '/nbody_simulations'        # location of simulation datasets
experiments_path = _data + '/Experiments/Nbody' # where model params & test preds saved

# Dataset
# =======
ZA_path = data_path + '/ZA'
ZA_samples = ZA_datasets = sorted(glob.glob(ZA_path + '/*.npy'))

# dataset labels
# --------------
# example za dpath: '/home/evan/.Data/nbody_simulations/ZA/007.npy'
za_labels = ['001', '002', '003', '004', '005',
             '006', '007', '008', '009', '010']

#-----------------------------------------------------------------------------#
#                                   NAMING                                    #
#-----------------------------------------------------------------------------#
# How variables and models are named

# Variable naming
# ===============
VAR_SCOPE  = 'params'  # tf variable scope; sess vars look like: 'params/W2_3'
weight_tag = 'W{}_{}'  # eg 'W2_3' : the 3rd weight for the 2nd net layer
bias_tag   = 'B{}_{}'  # generally only 1 bias per layer, so 'B_2' for 2nd net layer bias
scalar_tag = 'T_{}'    # scalar param, net out (currently no use) (DEFAULT 0.002)
model_tag  = ''        # default model name

# Model naming
# ============
ZA_naming  = AttrDict(model='ZA-FPM_{}', cube='X_{}')
model_gen_tags = ['arae', 'boot', 'cari', 'drac', 'erid', 'forn', 'gemi',
                  'hyda', 'indi', 'lyra', 'mensa', 'norma', 'orion', 'pavo',
                  'reti', 'scut', 'taur', 'ursa', 'virgo']

def name_model(naming, label_idx, model_tag=''):
    """ format naming style with dataset index {0:'001', '1':'002',...} and tag
    eg:
    name_model(ZA_naming, 3, 'testbatch')
        ---> (model='ZA-FPM_3_testbatch', cube='X_3')

    name_model(ZA_naming, 2) # (random tag)
        ---> (model='ZA-FPM_2_erid-ursa-hyda', cube='X_2')
    """
    if model_tag == '':
        mtags = random.choices(model_gen_tags, k=3)
        model_tag = '-'.join(mtags)
    model_tag = f'{label_idx}_{model_tag}'
    naming.model = naming.model.format(model_tag)
    naming.cube  = naming.cube.format(label_idx)
    print(f"NAMED: model={naming.model}\n       cube={naming.cube}")


#-----------------------------------------------------------------------------#
#                                DATA FEATURES                                #
#-----------------------------------------------------------------------------#
# Data vars
# =========
num_samples   = 1000
num_particles = 32**3
dataset_seed  = 12345

# default dataset selection
za_default_idx = 0
za_default = za_labels[za_default_idx] # '001'


#-----------------------------------------------------------------------------#
#                               MODEL SETTINGS                                #
#-----------------------------------------------------------------------------#
# Params
# ======
params_seed  = 77743196 # seed for weight inits; may be modified
channels = [6, 32, 64, 16, 8, 3]
#channels = [9, 32, 16, 8, 3] # shallow for corrected shift-inv (mem)
#channels = [6, 32, 64, 128, 256, 64, 16, 8, 3]  # set can go deeeeeep
num_neighbors = 14

# layer vars
num_layer_W = 4  # num weights per layer in network; 15 for upd. shiftinv, 4 for old
num_layer_B = 1  # 2 for 15op, 1 normally
num_scalar  = 2
scalar_val_init = 0.002





#-----------------------------------------------------------------------------#
#                                  TRAINING                                   #
#-----------------------------------------------------------------------------#

# Default settings
# ================
batch_size = 4
num_iters  = 20000
num_eval_samples  = 200
always_write_meta = False
restore = False # use pretrained model params


#=============================================================================#
#                                   SESSION                                   #
#=============================================================================#
"""
This section has the variable initializers and getters.

Network weights, biases, and scalars are all initialized and retrieved
with these functions.

NOTE:
    Both variable inits and gets ASSUME UNDER VARIABLE SCOPE
    There are scoped wrapper functions for full network param initialization,
    and getter for layer params
"""

# SMELL: getters don't have 'num_*' kwargs, but inits do? remove kwarg?

# Scalars
# =======
def init_scalar(nscalars=num_scalar, sval=scalar_val_init):
    """ scalars initialized by const  value """
    for i in range(nscalars):
        init = tf.constant([sval])
        tag  = scalar_tag.format(i)
        tf.get_variable(tag, dtype=tf.float32, initializer=init)

def get_scalar():
    t1 = tf.get_variable(scalar_tag.format(0))
    t2 = tf.get_variable(scalar_tag.format(1))
    return t1, t2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Biases
# ======
def init_bias(kdims, layer_idx, nbiases=num_layer_B, restore=False):
    """ biases initialized to be near zero """
    for i in range(nbiases):
        bname = bias_tag.format(layer_idx, i)
        args = (bname,)
        k = kdims[-1]
        if restore:
            init = None
            args += (k,)
        else:
            init = tf.ones((k_out,), dtype=tf.float32) * 1e-8
        tf.get_variable(*args, dtype=tf.float32, initializer=init)

def get_bias(layer_idx):
    b0 = tf.get_variable(bias_tag.format(layer_idx, 0))
    # FOR NOW, ASSUME WORKING WITH ONLY 1 BIAS
    #if num_layer_B > 1:
    #    biases = [b0]
    #    for i in range(1, num_layer_B):
    #        biases.append(tf.get_variable(bias_tag.format(layer_idx, i)))
    #    return biases
    return b0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Weights
# =======
def init_weight(kdims, layer_idx, nweights=num_layer_W, restore=False):
    """ weights sampled from glorot normal """
    for i in range(nweights):
        wname = weight_tag.format(layer_idx, i)
        args = (wname, kdims)
        init = None if restore else tf.glorot_normal_initializer(None)
        tf.get_variable(*args, dtype=tf.float32, initializer=init)

def get_weight(layer_idx):
    weights = []
    for i in range(num_layer_W):
        weights.append(tf.get_variable(weight_tag.format(layer_idx, i)))
    return weights

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Scoped wrappers
# ===============
def initialize_params(channels, vscope=VAR_SCOPE, restore=False, seed=params_seed):
    kdims = list(zip(channels[:-1], channels[1:]))
    tf.set_random_seed(seed)
    with tf.variable_scope(vscope, reuse=tf.AUTO_REUSE):
        for layer_idx, kdim in enumerate(kdims):
            #==== layer vars
            init_weight(kdim, layer_idx, restore=restore)
            init_bias(  kdim, layer_idx, restore=restore)
        #==== network out scalar
        init_scalar()

def get_params(layer_idx, vscope=VAR_SCOPE):
    with tf.variable_scope(vscope, reuse=True):
        W = get_weight(layer_idx)
        B = get_bias(layer_idx)
        return W, B

#------------------------------------------------------------------------------

# Session initialization
# ======================
def initialize_session():
    sess_kwargs = {}
    #==== Check for GPU
    if tf.test.is_gpu_available(cuda_only=True):
        gpu_frac = 0.85
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        sess_kwargs['config'] = tf.ConfigProto(gpu_options=gpu_opts)
    #==== initialize session
    sess = tf.InteractiveSession(**sess_kwargs)
    return sess

def initialize_graph(sess):
    """ initializes all variables AFTER computational graph
        has been specified (via endpoints data input and optimized error)
    """
    sess.run(tf.global_variables_initializer())
    print('\n\nAll variables initialized\n')

#██████████████████████████████████████████████████████████████████████████████
#                                 WHERE I STOPPED                             #
#██████████████████████████████████████████████████████████████████████████████

#=============================================================================#
#                                    SAVER                                    #
#=============================================================================#
""" utils for saving model parameters and experiment results """

####  WORK-IN-PROGRESS  ####
# this class has not yet been integrated into utils here !

class ModelSaver:
    #==== directories
    params_dir  = 'Session'
    results_dir = 'Results'
    def __init__(self, args):
        self.experiments_dir = args.experiments_dir
        self.num_iters = args.num_iters
        self.model_tag = args.model_tag
        self.dataset_type = args.dataset_type
        self.z_idx = args.z_idx[self.dataset_type]
        self.always_write_meta = args.always_write_meta

        # Model params
        # ============
        self.restore = args.restore

        # Format naming and paths
        # =======================
        self.assign_names()
        self.assign_pathing()


    def init_sess_saver(self):
        """ tensorflow.train.Saver must be initialized AFTER the computational graph
            has been initialized via tensorflow.global_variables_initializer
        """
        self.saver = tensorflow.train.Saver()
        if self.restore:
            self.restore_model_parameters()

    def assign_names(self):
        self.start_time = time.time()
        #==== key into naming and format args
        dset = self.dataset_type
        zidx = self.z_idx
        naming = naming_map[dset]

        #==== format names
        mname = naming['model'].format(*zidx)
        self.cube_name = naming['cube'].format(*zidx)
        if self.model_tag != '':
            mname = f'{mname}_{self.model_tag}'
        self.model_name = mname


    def assign_pathing(self):
        """ Pathing to directories for this model """
        # Base path
        # ---------
        epath = f"{self.experiments_dir}/{self.model_name}"
        self.experiments_path = epath

        # Directory pathing
        # -----------------
        self.params_path  = f'{epath}/{self.params_dir}'
        self.results_path = f'{epath}/{self.results_dir}'

        # Create model dirs
        # -----------------
        for p in [self.params_path, self.results_path]:
            if not os.path.exists(p): os.makedirs(p)


    def restore_model_parameters(self, sess):
        chkpt_state = tensorflow.train.get_checkpoint_state(self.params_path)
        self.saver.restore(sess, chkpt_state.model_checkpoint_path)
        print(f'Restored trained model parameters from {self.params_path}')


    def save_model_error(self, error, training=False):
        name = 'training' if training else 'validation'
        path = f'{self.results_path}/error_{name}'
        numpy.save(path, error)
        print(f'Saved error: {path}')


    def save_model_cube(self, cube, ground_truth=False):
        #name = self.cube_name_truth if ground_truth else self.cube_name_pred
        suffix = 'truth' if ground_truth else 'prediction'
        path = f'{self.results_path}/{self.cube_name}_{suffix}'
        numpy.save(path, cube)
        print(f'Saved cube: {path}')


    def save_model_params(self, cur_iter, sess):
        write_meta = self.always_write_meta
        if cur_iter == self.num_iters: # then training complete
            write_meta = True
            tsec = time.time() - self.start_time
            tmin  = tsec / 60.0
            thour = tmin / 60.0
            tS, tM, tH = f'{tsec: .3f}${tmin: .3f}${thour: .3f}'.split('$')
            print(f'Training complete!\n est. elapsed time: {tH}h, or {tM}m')
        step = cur_iter + 1
        self.saver.save(sess, self.params_path + '/chkpt',
                        global_step=step, write_meta_graph=write_meta)


    def print_checkpoint(self, step, err):
        print(f'Checkpoint {step + 1 :>5} :  {err:.6f}')


    def print_evaluation_results(self, err):
        #zx, zy = self.z_idx
        if 'ZA' in self.dataset_type:
            zx, zy = 'ZA', 'FastPM'
        else:
            zx, zy = self.z_idx

        #==== Statistics
        err_avg = numpy.mean(err)
        err_std = numpy.std(err)
        err_median = numpy.median(err)

        #==== Text
        title = f'\n# Evaluation Results:\n# {"="*78}'
        body = [f'# Location error : {zx:>3} ---> {zy:<3}',
                f'  median : {err_median : .5f}',
                f'    mean : {err_avg : .5f} +- {err_std : .4f} stdv',]
        #==== Print results
        print(title)
        for b in body:
            print(b)





#-----------------------------------------------------------------------------#
#                                   DATASET                                   #
#-----------------------------------------------------------------------------#

"""

# ZA Data Features
# ================
For each simulation, the shape of data is 32*32*32*19.

32*32*32 is the number of particles and,
they are on the uniform 32*32*32 grid.


## The meaning of the 19 columns is as follows:

oneSimu[...,  1:4] : ZA displacements (Dx,Dy,Dz)
oneSimu[...,  4:7] : 2LPT displacements
oneSimu[..., 7:10] : FastPM displacements
oneSimu[...,10:13] : ZA velocity
oneSimu[...,13:16] : 2LPT velocity
oneSimu[...,16:19] : FastPM velocity

"""

'''
class Dataset:
    seed = 12345  # for consistent splits
    def __init__(self, args):
        self.dataset_type = args.dataset_type
        self.batch_size = args.batch_size
        self.num_eval_samples = args.num_eval_samples
        self.simulation_data_path = args.simulation_data_path
        self.z_idx = args.z_idx[self.dataset_type]

        #=== Load data
        filenames = [self.fname.format(self.cube_steps[z]) for z in self.z_idx]
        paths = [f'{self.simulation_data_path}/{fname}' for fname in filenames]
        self.load_simulation_data(paths) # assigns self.X

    def split_dataset(self):
        """ split dataset into training and evaluation sets
        Both simulation datasets have their sample indices on
        the 1st axis
        """
        num_val = self.num_eval_samples
        np.random.seed(self.seed)
        ridx = np.random.permutation(self.X.shape[1])
        self.X_train, self.X_test = np.split(self.X[:, ridx], [-num_val], axis=1)
        #self.X = None # reduce memory overhead

    def get_minibatch(self):
        """ randomly select training minibatch from dataset """
        batch_size = self.batch_size
        batch_idx = np.random.choice(self.X_train.shape[1], batch_size)
        x_batch = np.copy(self.X_train[:, batch_idx])
        return x_batch

    def normalize(self):
        raise NotImplementedError

    def load_simulation_data(self, paths):
        for i, p in enumerate(paths):
            if i == 0:
                X = np.expand_dims(np.load(p), 0)
                continue
            X = np.concatenate([X, np.expand_dims(np.load(p), 0)], axis=0)
        self.X = X

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class ZA_Dataset(Dataset):
    fname = '/ZA/ZA_{}.npy'
    cube_steps = ZA_STEPS
    def __init__(self, args):
        super().__init__(args)
        self.get_za_fpm_data()
        #self.normalize()
        self.split_dataset()

    def get_za_fpm_data(self):
        """ NO NORMALIZATION """
        #=== formatting data
        reshape_dims = (1, 1000, 32**3, 3)

        #=== get ZA cubes
        ZA_disp = (self.X[...,1:4]).reshape(*reshape_dims)
        ZA_vel  = self.X[...,10:13].reshape(*reshape_dims)
        X_ZA = np.concatenate([ZA_disp, ZA_vel], axis=-1)

        #=== get FastPM cubes
        FPM_disp = (self.X[...,7:10]).reshape(*reshape_dims)
        FPM_vel  = self.X[...,16:19].reshape(*reshape_dims)
        X_FPM = np.concatenate([FPM_disp, FPM_vel], axis=-1)

        #=== Concat ZA and FastPM together, like typical redshift format
        self.X = np.concatenate([X_ZA, X_FPM], axis=0) # (2, 1000, 32**3, 6)


    def normalize(self):
        """ convert to positions and concat respective vels
        For each simulation, the shape of data is 32*32*32*19.

        32*32*32 is the number of particles and,
        they are on the uniform 32*32*32 grid.

        ## The meaning of the 19 columns is as follows:
        oneSimu[...,  1:4] : ZA displacements (Dx,Dy,Dz)
        oneSimu[...,  4:7] : 2LPT displacements
        oneSimu[..., 7:10] : FastPM displacements
        oneSimu[...,10:13] : ZA velocity
        oneSimu[...,13:16] : 2LPT velocity
        oneSimu[...,16:19] : FastPM velocity
        """
        #=== formatting data
        reshape_dims = (1, 1000, 32**3, 3)
        mrng = range(2,130,4)
        q = np.einsum('ijkl->kjli',np.array(np.meshgrid(mrng, mrng, mrng)))
        # q.shape = (32, 32, 32, 3)

        #=== get ZA cubes
        ZA_pos = (self.X[...,1:4] + q).reshape(*reshape_dims)
        ZA_vel = self.X[...,10:13].reshape(*reshape_dims)
        X_ZA = np.concatenate([ZA_pos, ZA_vel], axis=-1)

        #=== get FastPM cubes
        FPM_pos = (self.X[...,7:10] + q).reshape(*reshape_dims)
        FPM_vel = self.X[...,16:19].reshape(*reshape_dims)
        X_FPM = np.concatenate([FPM_pos, FPM_vel], axis=-1)

        #=== Concat ZA and FastPM together, like typical redshift format
        self.X = np.concatenate([X_ZA, X_FPM], axis=0) # (2, 1000, 32**3, 6)

class ZA_Cached_Dataset(Dataset):
    fname = '/ZA/ZA_{}.npy'
    fname_idx  = 'symm_idx'
    fname_feat = 'features'
    cube_steps = ZA_STEPS
    def __init__(self, args):
        super().__init__(args)
        self.cache_path = self.simulation_data_path + '/cached/X_{}_{}.npy'
        self.get_za_fpm_data()
        self.split_dataset()


    def split_dataset(self):
        # This simply splits indices
        np.random.seed(self.seed) # lol just use permutation, whyu use choice?
        #indices = np.random.choice(1000, 1000, replace=False)
        indices = np.random.permutation(np.arange(1000))
        self.eval_idx  = indices[-self.num_eval_samples:]
        self.train_idx = indices[:-self.num_eval_samples]

    def shuffle_train_idx(self):
        np.random.shuffle(self.train_idx)


    #def get_cached_data(self, indices):
    #    get_idx  = lambda i: np.load(self.cache_path.format(self.fname_idx,  i))
    #    get_feat = lambda i: np.load(self.cache_path.format(self.fname_feat, i))
    #    symm_idx = [get_idx[i] for i in indices]
    #    feats    = [get_feat[i][0] for i in indices]
    #    return feats, symm_idx

    def get_cached_data(self, idx):
        j = idx + 1 # the filenames are 1-indexed
        get_idx  = lambda i: np.load(self.cache_path.format(self.fname_idx,  i))
        get_feat = lambda i: np.load(self.cache_path.format(self.fname_feat, i))
        #symm_idx = [get_idx[i] for i in indices]
        #feats    = [get_feat[i][0] for i in indices]
        feats    = get_feat(j)[0] # (S, 9)
        symm_idx = list(get_idx(j)[0]) # (6,)
        return feats, symm_idx

    def get_za_fpm_data(self):
        """ NO NORMALIZATION """
        #=== formatting data
        reshape_dims = (1, 1000, 32**3, 3)

        #=== get ZA cubes
        ZA_disp = (self.X[...,1:4]).reshape(*reshape_dims)
        ZA_vel  = self.X[...,10:13].reshape(*reshape_dims)
        X_ZA = np.concatenate([ZA_disp, ZA_vel], axis=-1)

        #=== get FastPM cubes
        FPM_disp = (self.X[...,7:10]).reshape(*reshape_dims)
        FPM_vel  = self.X[...,16:19].reshape(*reshape_dims)
        X_FPM = np.concatenate([FPM_disp, FPM_vel], axis=-1)

        #=== Concat ZA and FastPM together, like typical redshift format
        self.X = np.concatenate([X_ZA, X_FPM], axis=0) # (2, 1000, 32**3, 6)


    def get_minibatch(self, idx):
        # Cube data (fpm)
        x_data = self.X[:,idx:idx+1]
        # Cached data
        feats, symm_idx = self.get_cached_data(idx)
        return x_data, feats, symm_idx


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Uni_Dataset(Dataset):
    fname = '/uniform/X_{:.4f}_.npy'
    cube_steps  = UNI_REDSHIFTS
    def __init__(self, args):
        super().__init__(args)
        filenames = [self.name_format(self.cube_steps[z]) for z in self.z_idx]
        paths = [f'{self.simulation_data_path}/{fname}' for fname in filenames]
        self.load_simulation_data(paths) # assigns self.X
        self.split_dataset()

    def normalize(self):
        self.X[...,:3] = self.X[...,:3] / 32.0

#------------------------------------------------------------------------------

def get_dataset(args):
    dset = args.dataset_type
    if dset == 'ZA':
        return ZA_Dataset(args)
    elif dset == 'ZA_15':
        return ZA_Cached_Dataset(args)
    else:
        return Uni_Dataset(args)
'''
