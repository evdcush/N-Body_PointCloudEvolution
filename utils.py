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
import argparse
from collections import NamedTuple
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

class AttrDict(dict):
    # dot accessible dict
    # NB: not pickleable
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

#-----------------------------------------------------------------------------#
#                            Settings & Constants                             #
#-----------------------------------------------------------------------------#

# Session vars
# ============
VAR_SCOPE  = 'params'  # tf variable scope; sess vars look like: 'params/W2_3'
params_seed = 77743196 # seed for weight inits; may be modified
weight_tag = 'W{}_{}'  # eg 'W2_3' : the 3rd weight for the 2nd net layer
bias_tag   = 'B{}_{}'  # generally only 1 bias per layer, so 'B_2' for 2nd net layer bias
scalar_tag = 'T_{}'    # scalar param, net out (currently no use) (DEFAULT 0.002)
model_tag  = ''        # default model name
restore = False        # if restore, then load previously trained model

# Data vars
# =========
num_particles = 32**3   # particles in point cloud
data_seed = 12345       # for consistent train/test dataset splits (best not to modify!)

# dataset labels
# --------------
za_labels = ['001', '002', '003', '004', '005',
             '006', '007', '008', '009', '010']
za_default = za_labels[3] # '004'

#redshifts = [9.0000, 4.7897, 3.2985, 2.4950, 1.9792,
#             1.6141, 1.3385, 1.1212, 0.9438, 0.7955,
#             0.6688, 0.5588, 0.4620, 0.3758, 0.2983,
#             0.2280, 0.1639, 0.1049, 0.0505, 0.0000] # redshift dset not in use
#rs_default = [10, 19] # (0.6688, 0.000)


# Training vars
# =============
batch_size = 4
num_iters  = 20000
num_eval_samples = 200
always_write_meta = False

# Model vars
# ==========
channels = [9, 32, 16, 8, 3]    # shallow for corrected shift-inv (mem)
#channels = [6, 32, 64, 128, 256, 64, 16, 8, 3]
num_neighbors = 14
num_layer_W = 15   # num weights per layer in network; 15 for upd. shiftinv, 4 for old
num_layer_B = 2    # 2 for 15op, 1 normally


class SessionManager:
    """ inits and gets vars within a tf session

    Variables can be tricky in tf compared to Chainer/torch--especially
    if you have mostly used tf's built-ins and interface assets.

    When not utilizing a tf network or layer interface, you have to be careful
    with variable initialization. Namely, you *must* initialize variables
    using tf's context manager:
    `with tf.variable_scope(vscope, reuse=tf.AUTO_REUSE)`

    Use tf.AUTO_REUSE--it will be True when there exists a variable within
    `vscope`, and False when not (thus making a new var). Static True may
    cause issues when initializing.

    The `reuse` kwarg is the most important part.
    If you do not specify 'True', or `tf.AUTO_REUSE`, *every time you
    "get" your variable, tf will  initialize a new variable and return it,*
    instead of using the same variables--you'll OOM and the model won't learn.

    Both the init and getters of a variable use tf.get_variable. If the
    variable requested is not defined within scope (ie, you did not reuse!),
    then tf.get_variable initializes a variable and returns it, otherwise
    it will return the requested variable.

    In short:
    tf.get_variable MUST be called under tf.variable_scope with tf.AUTO_REUSE

    """
    def __init__(self, args):
        kdims = list(zip(args.channels[:-1], args.channels[1:]))
        tf.set_random_seed(args.seed)
        with tf.variable_scope(VAR_SCOPE, reuse=tf.AUTO_REUSE):
            for layer_vars in enumerate(kdims):
                self.initialize_bias(  *layer_vars, args.restore)
                self.initialize_weight(*layer_vars, args.restore)
            if args.scalar:
                self.initialize_scalars(args.scalar)

    def initialize_scalars(self):
        """ scalars initialized by const value """
        for i in range(2):
            init = tf.constant([self.scalar_val])
            tag = self.scalar_tag.format(i)
            tf.get_variable(tag, dtype=tf.float32, initializer=init)

    def initialize_bias(self, layer_idx):
        """ biases initialized to be near zero """
        args = (self.bias_tag.format(layer_idx),)
        k_out = self.channels[layer_idx + 1] # only output chans relevant
        if self.restore:
            initializer = None
            args += (k_out,)
        else:
            initializer = tf.ones((k_out,), dtype=tf.float32) * 1e-8
        tf.get_variable(*args, dtype=tf.float32, initializer=initializer)

    def initialize_weight(self, layer_idx):
        """ weights sampled from glorot normal """
        kdims = self.channels[layer_idx : layer_idx+2]
        for w_idx in range(self.num_layer_W):
            name = self.weight_tag.format(layer_idx, w_idx)
            args = (name, kdims)
            init = None if self.restore else tf.glorot_normal_initializer(None)
            tf.get_variable(*args, dtype=tf.float32, initializer=init)

    def initialize_params(self):
        tf.set_random_seed(self.seed)
        with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
            for layer_idx in range(len(self.channels) - 1):
                #==== Layer vars
                self.initialize_bias(layer_idx)
                self.initialize_weight(layer_idx)
            #==== model vars
            self.initialize_scalars()

    # - - - - - - - - - - - -

    def get_scalars(self):
        t1 = tf.get_variable(self.scalar_tag.format(0))
        t2 = tf.get_variable(self.scalar_tag.format(1))
        return t1, t2

    def get_layer_vars(self, layer_idx):
        """ Gets all variables for a layer
        NOTE: ASSUMES VARIABLE SCOPE! Cannot get vars outside of scope.
        """
        get_B = lambda i: tf.get_variable(self.bias_tag.format(layer_idx, i))
        get_W = lambda i: tf.get_variable(self.weight_tag.format(layer_idx, i))
        #=== layer vars
        weights = [get_W(i) for i in range(self.num_layer_W)]
        bias    = [get_B(i) for i in range(self.num_layer_B)]
        if len(bias) == 1:
            bias = bias[0]
        #bias = tf.get_variable(self.bias_tag.format(layer_idx))
        return weights, bias

    def initialize_session(self):
        sess_kwargs = {}
        #==== Check for GPU
        if tf.test.is_gpu_available(cuda_only=True):
            gpu_frac = 0.85
            gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
            sess_kwargs['config'] = tf.ConfigProto(gpu_options=gpu_opts)
        #==== initialize session
        self.sess = tf.InteractiveSession(**sess_kwargs)

    def initialize_graph(self):
        """ initializes all variables after computational graph
            has been specified (via endpoints data input and optimized error)
        """
        self.sess.run(tf.global_variables_initializer())
        print('\n\nAll variables initialized\n')

    def __call__(self):
        """ return sess """
        if not hasattr(self, 'sess'):
            self.initialize_session()
        return self.sess


#-----------------------------------------------------------------------------#
#                             Pathing and Naming                              #
#-----------------------------------------------------------------------------#

# Base paths
# ==========
_home = os.environ['HOME'] # ~, your home directory, eg '/home/evan'
_data = _home + '/.Data'
_project = os.path.abspath(os.path.dirname(__file__)) # '/path/to/nbody_project'

# Data paths
# ==========
data_path = _data + '/nbody_simulations'        # location of simulation datasets
experiments_path = _data + '/Experiments/Nbody' # where model params & test preds saved

# Datasets
# ========
"""
Available datasets:
There are ten ZA/FPM datasets available with, numbered '001'...'010',
eg: 'ZA_001.npy', 'ZA_002.npy', ..., 'ZA_010.npy'

Cache : cached graph data for corrected shift-inv model (15op), using ZA_001.npy
    data is cached as features and symm_idx for each sample (1000 total), 1-indexed
    eg 'X_features_37.npy', 'X_symm_idx_37.npy'

    Since the cache filname numbers are not padded with leading zeros,
    eg '98' instead of '0098', care must be taken when sorting fnames
    (so you don't get [..., '979', '98', '980', '981', ...])

"""
ZA_path = data_path + '/ZA'
ZA_datasets = sorted(glob.glob(ZA_path + '/*.npy'))

# Cached data
# -----------
ZA_cache = data_path + '/cached'
cache_sort_key = lambda s: int(s.split('_')[-1][:-4]) # fname number
get_cache = lambda gmatch: sorted(glob.glob(ZA_cache + gmatch), key=cache_sort_key)

cached_features = get_cache('*features*')
cached_indices  = get_cache('*symm*')


# Naming formats
# ==============
""" fnames for model params """
ZA_naming  = dict(model='SI_ZA-FastPM_{}', cube='X_{}')
#UNI_naming = dict(model='SI_{}-{}', cube='X_{}-{}') # redshift-based dataset not used
naming_map = {'ZA': ZA_naming, 'ZA_15': ZA_naming, }#'UNI': UNI_naming}




---

seed : 77743196
var_scope : params_{}

#channels : [3, 32, 16, 8, 3]
channels : [9, 32, 16, 8, 3]
#channels : [3, 32, 64, 128, 256, 64, 16, 8, 3]
#num_layer_W : 4  # weights per layer; 4 for shiftinv
#num_layer_W : 1  # weights per layer, SET
num_layer_W : 15
num_layer_B : 2 # normally 1
scalar_val : 0.002
restore : false

num_eval_samples: 200
num_iters: 2000
#batch_size: 4
batch_size : 1
neighbors : 14
always_write_meta : false
model_tag : ''

z_idx :
    ZA  : [0]
    ZA_15  : [0]
    UNI : [10, 19]
dataset_type : ZA_15 # Either ZA or UNI




#-----------------------------------------------------------------------------#
#                                    Saver                                    #
#-----------------------------------------------------------------------------#



# WIP: this just copy-pasted from utils/saver.py
#
'''
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


'''

'''
import tensorflow as tf
import code
class Initializer:
    """Initializes variables and provides their getters
    """
    weight_tag = 'W{}_{}'
    #bias_tag   = 'B_{}'
    bias_tag   = 'B{}_{}'
    scalar_tag = 'T_{}'
    def __init__(self, args):
        self.seed = args.seed
        self.restore = args.restore
        self.channels = args.channels
        self.var_scope = args.var_scope.format(args.dataset_type)
        self.scalar_val = args.scalar_val
        self.num_layer_W = args.num_layer_W
        self.num_layer_B = args.num_layer_B

    def initialize_scalars(self):
        """ scalars initialized by const value """
        for i in range(2):
            init = tf.constant([self.scalar_val])
            tag = self.scalar_tag.format(i)
            tf.get_variable(tag, dtype=tf.float32, initializer=init)

    #def initialize_bias(self, layer_idx):
    #    """ biases initialized to be near zero """
    #    args = (self.bias_tag.format(layer_idx),)
    #    k_out = self.channels[layer_idx + 1] # only output chans relevant
    #    if self.restore:
    #        initializer = None
    #        args += (k_out,)
    #    else:
    #        initializer = tf.ones((k_out,), dtype=tf.float32) * 1e-8
    #    tf.get_variable(*args, dtype=tf.float32, initializer=initializer)

    def initialize_bias(self, layer_idx):
        """ biases initialized to be near zero """
        #args = (self.bias_tag.format(layer_idx),)
        k_out = self.channels[layer_idx + 1] # only output chans relevant
        for b_idx in range(self.num_layer_B):
            name = self.bias_tag.format(layer_idx, b_idx)
            args = (name,)
            if self.restore:
                init = None
                args +=  (k_out,)
            else:
                init = tf.ones((k_out,), dtype=tf.float32) * 1e-8
            tf.get_variable(*args, dtype=tf.float32, initializer=init)

    def initialize_weight(self, layer_idx):
        """ weights sampled from glorot normal """
        kdims = self.channels[layer_idx : layer_idx+2]
        for w_idx in range(self.num_layer_W):
            name = self.weight_tag.format(layer_idx, w_idx)
            args = (name, kdims)
            init = None if self.restore else tf.glorot_normal_initializer(None)
            tf.get_variable(*args, dtype=tf.float32, initializer=init)

    def initialize_params(self):
        tf.set_random_seed(self.seed)
        with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
            for layer_idx in range(len(self.channels) - 1):
                #==== Layer vars
                self.initialize_bias(layer_idx)
                self.initialize_weight(layer_idx)
            #==== model vars
            self.initialize_scalars()

    # - - - - - - - - - - - -

    def get_scalars(self):
        t1 = tf.get_variable(self.scalar_tag.format(0))
        t2 = tf.get_variable(self.scalar_tag.format(1))
        return t1, t2

    def get_layer_vars(self, layer_idx):
        """ Gets all variables for a layer
        NOTE: ASSUMES VARIABLE SCOPE! Cannot get vars outside of scope.
        """
        get_B = lambda i: tf.get_variable(self.bias_tag.format(layer_idx, i))
        get_W = lambda i: tf.get_variable(self.weight_tag.format(layer_idx, i))
        #=== layer vars
        weights = [get_W(i) for i in range(self.num_layer_W)]
        bias    = [get_B(i) for i in range(self.num_layer_B)]
        if len(bias) == 1:
            bias = bias[0]
        #bias = tf.get_variable(self.bias_tag.format(layer_idx))
        return weights, bias

    def initialize_session(self):
        sess_kwargs = {}
        #==== Check for GPU
        if tf.test.is_gpu_available(cuda_only=True):
            gpu_frac = 0.85
            gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
            sess_kwargs['config'] = tf.ConfigProto(gpu_options=gpu_opts)
        #==== initialize session
        self.sess = tf.InteractiveSession(**sess_kwargs)

    def initialize_graph(self):
        """ initializes all variables after computational graph
            has been specified (via endpoints data input and optimized error)
        """
        self.sess.run(tf.global_variables_initializer())
        print('\n\nAll variables initialized\n')

    def __call__(self):
        """ return sess """
        if not hasattr(self, 'sess'):
            self.initialize_session()
        return self.sess

'''
