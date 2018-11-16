import os, code, sys, time
import numpy as np
import tensorflow as tf
import nn
import utils

from shift_inv import get_symmetrized_idx, model_func_ShiftInv_symm, get_input_features_ShiftInv_numpy


# Data features
# ========================================
N = 32**3
M = 14
batch_size = 4
zx, zy = 10, 19

# Load data
# ========================================
CACHED_DATA_DIR = './CachedData'
X_input = utils.normalize(utils.load_simulation_data([zx,]))
print(f'zx ---> zy = {zx} ---> {zy}')


# Cached cubes
# =======================================
cached_input_features = []
cached_symm_idx = []

def cache_data(label, data):
    save_name = f'{CACHED_DATA_DIR}/X_{zx}-{zy}_{label}'
    np.save(save_name, data)
    print(f'Cached "{save_name}"!')

def cache_all():
    cache_data('features', cached_input_features)
    cache_data('symm_idx', cached_symm_idx)


#  PROCESS DATA # (1000, 32768, 6)
# ========================================
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
num_batches = X_input.shape[0] // batch_size # 250
for i in range(num_batches):
    # Make "batches"
    # --------------
    p, q = batch_size*i, batch_size*(i+1)
    x_in = X_input[p:q] # (b, N, 6)

    # NN CSRs
    # -------
    csr_list = nn.get_kneighbor_list(x_in, M, include_self=True)

    # Input features
    # --------------
    x_in_feats = get_input_features_ShiftInv_numpy(np.copy(x_in), csr_list, N, None)
    cached_input_features.append(x_in_feats)

    # Symmetrical indices
    # -------------------
    symm_idx = get_symmetrized_idx(csr_list)
    cached_symm_idx.append(symm_idx)

    print(f'Batch {i:>3}, Processed')

    # Intermediate caching
    # --------------------
    if (i + 1) % 25 == 0:
        cache_all()

# CACHE DATA
# ==========
cache_data('features', cached_input_features)
cache_data('symm_idx', cached_symm_idx)


# run in ipython shell instead, this is too risky
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
