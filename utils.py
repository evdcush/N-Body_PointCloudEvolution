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
