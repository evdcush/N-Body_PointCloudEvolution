'''
This file contains the parameters used for data, models, and training.
While rarely, if ever changed, these are not necessarilly fixed or static
Defined here because for consistent usage as globals across .py
scripts and notebooks.

Designed to be used as 'from params import *'
'''

import models
import nn


# dataset
DATA_PATH     = '/home/evan/Data/nbody_simulations/N_{0}/DM*/{1}_dm.z=0{2}000'
DATA_PATH_NPY = '/home/evan/Data/nbody_simulations/nbody_{}.npy'
REDSHIFTS = [6.0, 4.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
RS_TAGS = {6.0:'60', 4.0:'40', 2.0:'20', 1.5:'15', 1.2:'12', 1.0:'10', 0.8:'08', 0.6:'06', 0.4:'04', 0.2:'02', 0.0:'00'}

# rng seeds
#RNG_SEEDS     = [98765, 12345, 319, 77743196] # original training seeds for network params
PARAMS_SEED  = 77743196 # best consistent performance for graph models, set models do better with 98765
DATASET_SEED = 12345

# models
GRAPH_CHANNELS = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 3]
SET_CHANNELS   = [6, 32, 128, 256, 128, 32, 256, 16, 3]
NBODY_MODELS = {0:{'mclass': models.SetModel,       'channels':   SET_CHANNELS, 'tag': 'S', 'loss':nn.bounded_mean_squared_error},
                1:{'mclass': models.GraphModel,     'channels': GRAPH_CHANNELS, 'tag': 'G', 'loss':nn.get_min_readout_MSE},
                2:{'mclass': models.VelocityScaled, 'channels': GRAPH_CHANNELS, 'tag': 'V', 'loss':nn.bounded_mean_squared_error}}
LEARNING_RATE = 0.01