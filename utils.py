"""
utils provides the following:

user settings & defaults
------------------------
- data paths
    * change these to point to your data dirs!
- model naming
- default values for initialization, training, model architecture, etc.

Parser
======
The argument parser (CLI) for training a model.
Common argument adjustments are '--num_iters' '--batch_size' and '--name'

Session
=======
Important functions for initialization and retrieval of tensorflow
variables.

Saver
=====
A class that wraps all the naming and saving for experiments and models.

Dataset
=======
Manages the loading and processing of datasets from disk, and also
provides the batching interface to the dataset during training.

"""
import os
import sys
import glob
import code
import time
import random
import argparse
import datetime
from functools import wraps
from collections import namedtuple

import yaml
import numpy as np
import tensorflow as tf


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

def TODO(f):
    """ Decorator that flags a function and insures wrapee will not run """
    @wraps(f)
    def not_finished():
        raise NotImplementedError('\n    {} IS INCOMPLETE'.format(f.__name__))
    return not_finished


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
data_path = _data + '/nbody_simulations'       # location of simulation datasets
EXPERIMENTS_DIR = _data + '/Experiments/Nbody' # where model params & test preds saved
PARAMS_DIR  = EXPERIMENTS_DIR + '/{}' + '/Session'
RESULTS_DIR = EXPERIMENTS_DIR + '/{}' + '/Results'

# Dataset
# =======
ZA_path = data_path + '/ZA'
ZA_PATHS = ZA_datasets = sorted(glob.glob(ZA_path + '/*.npy')) # 10 total ZA datasets
# example paths:
#  ['/home/evan/.Data/nbody_simulations/ZA/ZA_001.npy',
#   '/home/evan/.Data/nbody_simulations/ZA/ZA_002.npy',
#    ...,
#   '/home/evan/.Data/nbody_simulations/ZA/ZA_010.npy']


# dataset labels
# --------------
#ZA_LABELS = [f'{i:0>2}' for i in range(1, len(ZA_PATHS) + 1)]
#ZA_LABELS = [p.split('_')[-1][:3] for p in ZA_paths]
ZA_LABELS = ['001', '002', '003', '004', '005',
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
CUBE_NAME = 'X_{}'
MODEL_NAME_ZA = 'ZA-FPM_{}'
MODEL_TAGLIST = ['arae', 'boot', 'cari', 'drac', 'erid', 'forn', 'gemi',
                 'hyda', 'indi', 'lyra', 'mensa', 'norma', 'orion', 'pavo',
                 'reti', 'scut', 'taur', 'ursa', 'virgo']


#-----------------------------------------------------------------------------#
#                                DATA FEATURES                                #
#-----------------------------------------------------------------------------#
# Data vars
# =========
num_samples   = 1000
NUM_PARTICLES = 32**3
DATASET_SEED  = 12345

# default dataset selection
ZA_DEFAULT_IDX = 0
za_default = ZA_LABELS[ZA_DEFAULT_IDX] # '001'


#-----------------------------------------------------------------------------#
#                               MODEL SETTINGS                                #
#-----------------------------------------------------------------------------#
# Params
# ======
PARAMS_SEED  = 77743196 # seed for weight inits; may be modified
#CHANNELS = [6, 32, 64, 16, 8, 3]
#channels = [9, 32, 16, 8, 3] # shallow for corrected shift-inv (mem constrained)
#CHANNELS = [6, 32, 64, 128, 256, 64, 16, 8, 3]  # set can go deeeeeep
CHANNELS = [6, 64, 128, 128, 256, 64, 128, 16, 3]  # set can go deeeeeep
NUM_NEIGHBORS = 14

# initializers
# ============
# These are the distributions from which WEIGHTS sampled (biases just near 0)
uniform = tf.random_uniform_initializer
normal  = tf.random_normal_initializer
glorot_uniform = tf.glorot_uniform_initializer # trying this out
glorot_normal  = tf.glorot_normal_initializer  # historic default

# layer vars
# ==========
INIT_DISTRIBUTION = glorot_normal #glorot_uniform
num_layer_W = 4  # num weights per layer in network; 15 for upd. shiftinv, 4 for old
num_layer_B = 1  # 2 for 15op, 1 normally
num_scalar  = 1
scalar_val_init = 0.002


#-----------------------------------------------------------------------------#
#                                  TRAINING                                   #
#-----------------------------------------------------------------------------#

# Default settings
# ================
batch_size = 4
num_iters  = 20000
num_test_samples  = 200
always_write_meta = False
restore = False # use pretrained model params

# training-model interface
# ========================
ModelVars = namedtuple('ModelVars', ['num_layers',     # len(channels) - 1
                                    'get_layer_vars', # will wrap `get_params`
                                    'activation']      # tf.nn activation func
                                    )


#=============================================================================#
#                                                                             #
#   88888888ba                                                                #
#   88      "8b                                                               #
#   88      ,8P                                                               #
#   88aaaaaa8P'  ,adPPYYba,  8b,dPPYba,  ,adPPYba,   ,adPPYba,  8b,dPPYba,    #
#   88""""""'    ""     `Y8  88P'   "Y8  I8[    ""  a8P_____88  88P'   "Y8    #
#   88           ,adPPPPP88  88           `"Y8ba,   8PP"""""""  88            #
#   88           88,    ,88  88          aa    ]8I  "8b,   ,aa  88            #
#   88           `"8bbdP"Y8  88          `"YbbdP"'   `"Ybbd8"'  88            #
#                                                                             #
#=============================================================================#
doc = '''\
CLI for model training. Make sure to adjust paths in utils.py to your setup!

# To use this parser in your training script:
from utils import PARSER
args = PARSER.parse_args()  # your config variables are in this namespace
channels = args.channels
num_iters = args.num_iters
'''

epi = '''\
Some example usage:

# defaults
python train.py

# train for 10000 iterations, batch size 8, using param seed 98765
python train.py -i 10000 -b 8 -s 98765

# train with different channels c, name n, on different dataset d
python train.py -c 6 64 64 128 32 3 -n 'denser_layer_test' -d 4

'''

# Parser init and args
PARSER = argparse.ArgumentParser(description=doc, epilog=epi,
                                 formatter_class=argparse.RawTextHelpFormatter)
adg = PARSER.add_argument
adg('-c', '--channels', type=int, nargs='+', default=CHANNELS, metavar='C',
    help='List of ints that define layer sizes')

adg('-i', '--num_iters', type=int, default=num_iters, metavar='N',
    help='Number of training iterations')

adg('-b', '--batch_size', type=int, default=batch_size, metavar='B',
    help='Number of samples per training batch')

adg('-d', '--data_idx', type=int, default=0,
    choices=set(range(len(ZA_LABELS))), metavar='i',
    help='Index, int in [0, 10), corresponding to a dataset; eg 0: ')

adg('-k', '--kneighbors', type=int, default=NUM_NEIGHBORS, metavar='K',
    help='Number of neighbors in graph model (KNN); if K == -1, then set model')

adg('-n', '--name', type=str, default='', metavar='name',
    help='Name for model; randomly generated if not specified')

adg('-s', '--seed', type=int, default=PARAMS_SEED, metavar='X',
    help='Random seed for parameter initialization')

adg('-l', '--learnrate', type=float, default=0.01, metavar='lr',
    help='Learning rate for optimizer')

adg('-t', '--num_test', type=int, default=num_test_samples, metavar='M',
    help='Number of samples in test set')


# NOT YET SUPPORTED
#adg('-r', '--restore', action='store_true',
#    help='Restore pretrained model parameters')



#===============================================================================#
#                                                                               #
#  ad88888ba                                     88                             #
# d8"     "8b                                    ""                             #
# Y8,                                                                           #
# `Y8aaaaa,     ,adPPYba,  ,adPPYba,  ,adPPYba,  88   ,adPPYba,   8b,dPPYba,    #
#   `"""""8b,  a8P_____88  I8[    ""  I8[    ""  88  a8"     "8a  88P'   `"8a   #
#         `8b  8PP"""""""   `"Y8ba,    `"Y8ba,   88  8b       d8  88       88   #
# Y8a     a8P  "8b,   ,aa  aa    ]8I  aa    ]8I  88  "8a,   ,a8"  88       88   #
#  "Y88888P"    `"Ybbd8"'  `"YbbdP"'  `"YbbdP"'  88   `"YbbdP"'   88       88   #
#                                                                               #
#===============================================================================#
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
    #t2 = tf.get_variable(scalar_tag.format(1))
    return t1#, t2

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
            init = tf.ones((k,), dtype=tf.float32) * 1e-8
        tf.get_variable(*args, dtype=tf.float32, initializer=init)

def get_bias(layer_idx):
    b0 = tf.get_variable(bias_tag.format(layer_idx, 0))
    # FOR NOW, ASSUME WORKING WITH ONLY 1 BIAS (fully expressed graph uses 2)
    #if num_layer_B > 1:
    #    biases = [b0]
    #    for i in range(1, num_layer_B):
    #        biases.append(tf.get_variable(bias_tag.format(layer_idx, i)))
    #    return biases
    return b0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Weights
# =======
def init_weight(kdims, layer_idx, nweights=num_layer_W,
                initializer=INIT_DISTRIBUTION, restore=False):
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
def initialize_params(channels, vscope=VAR_SCOPE, restore=False, seed=PARAMS_SEED):
    kdims = list(zip(channels[:-1], channels[1:]))
    tf.set_random_seed(seed)
    with tf.variable_scope(vscope, reuse=tf.AUTO_REUSE):
        for layer_idx, kdim in enumerate(kdims):
            #==== layer vars
            init_weight(kdim, layer_idx, restore=restore)
            init_bias(  kdim, layer_idx, restore=restore)
        #==== network out scalar
        #init_scalar()

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




#=============================================================================#
#                                                                             #
#         ad88888ba                                                           #
#        d8"     "8b                                                          #
#        Y8,                                                                  #
#        `Y8aaaaa,    ,adPPYYba,  8b       d8   ,adPPYba,  8b,dPPYba,         #
#          `"""""8b,  ""     `Y8  `8b     d8'  a8P_____88  88P'   "Y8         #
#                `8b  ,adPPPPP88   `8b   d8'   8PP"""""""  88                 #
#        Y8a     a8P  88,    ,88    `8b,d8'    "8b,   ,aa  88                 #
#         "Y88888P"   `"8bbdP"Y8      "8"       `"Ybbd8"'  88                 #
#                                                                             #
#=============================================================================#
""" utils for saving model parameters and experiment results """

def mkpath(p):
    if not os.path.exists(p):
        os.makedirs(p)

class Saver:
    """ a class which aggregates all the pathing and naming

    NB: This class does *not* hold any data. It is a wrapper for
        the many pathing and file saving utils.

    Attrs
    -----
    name : name of the model, eg 'ZA-FPM_2_erid_ursa_hyda'
    cube : filename for the result cubes
    results : where model results are saved; '.../{name}/Results'
    params  : where model params are saved; '.../{name}/Session'

    restore : bool
        whether to restore trained params

    saver : tf.train.Saver
        tf saver for saving and loading model data

    """
    def __init__(self, label_idx, basename=MODEL_NAME_ZA, cube_name=CUBE_NAME,
                 model_tag=model_tag, restore=False):
        if model_tag == '': # If no tag specified, generate random one
            mtags = random.choices(MODEL_TAGLIST, k=3)
            model_tag = '-'.join(mtags)

        #=== format names
        model_tag = f'{label_idx}_{model_tag}' # eg '2_foo'
        model_name = basename.format(model_tag)  # 'ZA-FPM_{}'.format('2_foo')
        self.name = model_name
        self.cube = cube_name.format(label_idx) # 'X_{}'.format(2)

        #=== format paths with model name
        self.results = RESULTS_DIR.format(model_name) # '{datadir}/ZA-FPM_2_foo/Results'
        self.params  = PARAMS_DIR.format(model_name)  # '{datadir}/ZA-FPM_2_foo/Session'
        mkpath(self.results)
        mkpath(self.params)
        print(f"MODEL NAMED: {self.name}")


        #=== session attrs
        self.restore = restore

    def init_sess_saver(self,):
        """ tensorflow.train.Saver must be initialized AFTER the computational graph
            has been initialized via tensorflow.global_variables_initializer
        """
        self.saver = tf.train.Saver()
        if self.restore:
            self.restore_model_parameters()

    def restore_model_parameters(self):
        raise NotImplementedError('TODO')

    def save_model(self, cur_iter, sess, write_meta=False):
        self.saver.save(sess, self.params + '/chkpt',
            global_step=cur_iter+1, write_meta_graph=write_meta)

    def save_error(self, error, training=False):
        suffix = 'training' if training else 'test'
        dst = self.results + f'/error_{suffix}'
        np.save(dst, error)
        print(f"\n\tSaved model {suffix} error: \n\t\t{dst}\n")

    def save_cube(self, cube, ground_truth=False):
        suffix = 'truth' if ground_truth else 'prediction'
        dst = f'{self.results}/{self.cube}_{suffix}'
        np.save(dst, cube)
        print(f"\n\tSaved {suffix} cube: \n\t\t{dst}\n")

    @staticmethod
    def print_checkpoint(step, err):
        print(f"Checkpoint {step + 1 :>5} : {err:.6f}")

    @staticmethod
    def print_evaluation_results(err):
        #==== Statistics
        err_avg = np.mean(err)
        err_std = np.std(err)
        err_median = np.median(err)
        #==== Text
        tbody = [f'\n# Test Error\n# {"="*17}',
                 f'  median : {err_median : .5f}',
                 f'    mean : {err_avg : .5f} +- {err_std : .4f} stdv',]
        eval_results = '\n'.join(tbody)
        print(eval_results)


#==================================================================================#
#                                                                                  #
# 88888888ba,                                                                      #
# 88      `"8b                 ,d                                          ,d      #
# 88        `8b                88                                          88      #
# 88         88  ,adPPYYba,  MM88MMM  ,adPPYYba,  ,adPPYba,   ,adPPYba,  MM88MMM   #
# 88         88  ""     `Y8    88     ""     `Y8  I8[    ""  a8P_____88    88      #
# 88         8P  ,adPPPPP88    88     ,adPPPPP88   `"Y8ba,   8PP"""""""    88      #
# 88      .a8P   88,    ,88    88,    88,    ,88  aa    ]8I  "8b,   ,aa    88,     #
# 88888888Y"'    `"8bbdP"Y8    "Y888  `"8bbdP"Y8  `"YbbdP"'   `"Ybbd8"'    "Y888   #
#                                                                                  #
#==================================================================================#
"""
# ZA Data Features
# ================
For each simulation, the shape of data is 32*32*32*19.

32*32*32 is the number of particles and,
they are on the uniform 32*32*32 grid.

## The meaning of the 19 columns is as follows:
X[...,  1:4] : ZA displacements (Dx,Dy,Dz)
X[...,  4:7] : 2LPT displacements
X[..., 7:10] : FastPM displacements
X[...,10:13] : ZA velocity
X[...,13:16] : 2LPT velocity
X[...,16:19] : FastPM velocity
"""

class Dataset:
    """ Manages dataset and loading, processing, batching """
    seed = DATASET_SEED # 12345
    data_paths = ZA_PATHS # ['/path/to/data/ZA_001.npy', '/path/to/data/ZA_002.npy',...]
    num_particles = NUM_PARTICLES # 32**3
    num_samples   = num_samples   # 1000
    def __init__(self, data_idx=ZA_DEFAULT_IDX, num_test=num_test_samples):
        self.data_idx = data_idx
        self.num_test = num_test
        X = self.load_data(data_idx)
        self.X_train, self.X_val, self.X_test = self.split_dataset(X, num_test)

    def get_minibatch(self, batch_size=batch_size):
        """ randomly select training minibatch from dataset """
        #N = self.X_train.shape[1]
        N = self.X_train.shape[0]
        batch_idx = np.random.choice(N, batch_size, replace=False)
        #x = np.copy(self.X_train[:, batch_idx])
        x = np.copy(self.X_train[batch_idx])
        return x


    #@TODO
    #def test_epoch(self):
    #    """ make separate 'batching' func for testing? """
    #    pass

    @classmethod
    def split_dataset(cls, X, num_test):
        """ Splits dataset into train, validation, and test sets

        Params
        ------
        X : ndarray.float32; (2, 1000, 32**3, D)
            ZA and FPM data

        num_test : int
            number of samples in the test set
        """
        np.random.seed(cls.seed)
        #rnd_idx = np.random.permutation(X.shape[1])
        rnd_idx = np.random.permutation(X.shape[0])
        split_idx = [-num_test - 100, -num_test] # could just go from front..
        #return np.split(X[:, rnd_idx], split_idx, axis=1)
        return np.split(X[rnd_idx], split_idx, axis=0)

    @classmethod
    def load_data(cls, data_idx):
        """ load dataset given data index which corresponds to the filename """
        dpath = cls.data_paths[data_idx]
        data = np.load(dpath) # (1000, 32, 32, 32, 19)
        print("\nLoaded data from:\n\t" + dpath + '\n')

        #=== Process data
        reshape_dims = (cls.num_samples, cls.num_particles, 3)
        # data is reshaped like:
        #     (1000, 32, 32, 32, 3) ---> (1, 1000, 32**3, 3)

        # displacements
        za  = data[...,1: 4].reshape(*reshape_dims)
        fpm = data[...,7:10].reshape(*reshape_dims)
        fpm = fpm - za # NOTE: THIS IS DIFF FROM PREVIOUS WAY (true_error)

        # grid pos
        mg = range(2, 130, 4)
        q = np.einsum('ijkl->kjli', np.array(np.meshgrid(mg, mg, mg)))
        q = np.broadcast_to(q.reshape(1, -1, 3), za.shape)#.reshape(*za.shape) # broadcast

        za = np.concatenate([q-64, za], axis=-1) # (1000, N, 6)

        # 'cat cubes
        #X = np.concatenate([za, fpm], axis=-1)
        X = np.concatenate([za, fpm], axis=-1)
        #code.interact(local=dict(globals(), **locals()))
        return X


def get_init_pos(za_disp):
    b, N, k = za_disp.shape
    mg = range(2, 130, 4)
    q = np.einsum('ijkl->kjli', np.array(np.meshgrid(mg, mg, mg)))
    qr = q.reshape(-1, 3)
    init_pos = za_disp + qr
    return init_pos
