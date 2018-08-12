import os, code, sys, time

import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import coo_matrix
import tensorflow as tf

import utils
from utils import VAR_SCOPE, SEGNAMES_3D
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# TF-related ops
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
'''
NEW STANDARD FOR MODEL/NETWORK FUNC ARGS:
Instead of always passing num_layers, var_scope, activation, dims
 - pass dict of the static values, and data as only args
 - maybe make get funcs for each model type?
'''
class ModelFuncArgs():
    def __init__(self, num_layers, var_scope, dims=None, vcoeff=False,
                 add_skip=False, activation_func=tf.nn.relu):
        self.num_layers = num_layers
        self.var_scope = var_scope
        self.dims = dims # LEN/ORDER IS ARBITRARY, whatever individual model funcs expect
        self.vcoeff = vcoeff
        self.add_skip = add_skip
        self.activation_func = activation_func

    def output_skips(self, H_out):
        """ EXPERIMENTAL, don't know how gradient flow affected by funcs wrapped
        in class methods
        TODO
        """
        assert False

    def get_ShiftInv_specs(self,):
        #specs = [self.num_layers, self.var_scope]
        assert False

    def __call__(self):
        # return the only two things EVERY model will have
        return self.num_layers, self.var_scope
#=============================================================================
# ROTATION INVARIANT NETWORK/LAYER OPS
#=============================================================================
#------------------------------------------------------------------------------
# ROTATION invariant layer ops
#------------------------------------------------------------------------------
def pool_rot_graph(X, idx, broadcast):
    """
    Args:
        X (tensor): has shape (c, k), row-major order
        idx (numpy array): has shape (c),
            must be row idx of non-zero entries to pool over columns
            must be column idx of non-zero entries to pool over rows
        N (int): number of segments (number of particles in this case)
        b (int): batch size
        broadcast (bool): if True, after pooling re-broadcast to original shape

    Returns:
        tensor of shape (c, k) if broadcast, else (b*N, k)
    """
    num_segs = tf.reduce_max(idx) + 1
    X_pooled = tf.unsorted_segment_mean(X, idx, num_segs)
    if broadcast:
        X_pooled = tf.gather_nd(X_pooled, tf.expand_dims(idx, axis=2))
    else:
        X_pooled = tf.reshape(X_pooled, [tf.shape(X)[0], -1, tf.shape(X)[2]])
    return X_pooled


def RotInv_layer(H_in, segID_3D, bN, layer_id, is_last=False):
    """
    Args:
        H_in (tensor): (b, e, k)
            b=batch_size
            e=N*(M-1)*(M-2), num of edges in 3D adjacency (diags removed), N=num_part, M=num_neigh
            k=channel size
        segID_3D: (b, 7, e) segment ids for pooling, there are 7:
            col-depth, row-depth, row-col, depth, col, row, all
        W: 8 (or 6) set of weights for pooling
        B: bias
    Returns:
        H_out (tensor): (c, q), or (b, N, q) if is_last
    """
    # Prepare data and parameters
    # ========================================
    # split inputs
    b, N = bN
    #row_idx, col_idx, cube_idx = tf.split(COO_feats, 3, axis=0)
    row_idx  = COO_feats[0]
    col_idx  = COO_feats[1]
    cube_idx = COO_feats[2]

    # Helper funcs
    # ========================================
    def _left_mult(h, W_name):
        W = utils.get_scoped_RotInv_weight(layer_idx, W_name)
        return tf.einsum("ijk,kq->ijq", h, W)

    # Layer forward pass
    # ========================================
    H_out = _left_mult(H_in, SEGNAMES_3D[0])
    for idx, seg_name in enumerate(SEGNAMES_3D[1:]):
        H_pooled = pool_rot_graph(H_in, segID_3D[:, idx], True)
        H_out = H_out + _left_mult(H_pooled, seg_name)

    # Output
    # ========================================
    bias = tf.get_variable(utils.BIAS_TAG.format(layer_idx))
    H_out = H_out + bias
    if is_last: # pool over depth dim
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        H_out = pool_rot_graph(H_out, segID_3D[:,3], False)
    return H_out



#=============================================================================
# SHIFT INVARIANT NETWORK/LAYER OPS
#=============================================================================
#------------------------------------------------------------------------------
# Shift invariant layer ops
#------------------------------------------------------------------------------
def pool_graph(X, idx, num_segs, broadcast):
    """
    Args:
        X (tensor): has shape (c, k), row-major order
        idx (numpy array): has shape (c),
            must be row idx of non-zero entries to pool over columns
            must be column idx of non-zero entries to pool over rows
        N (int): number of segments (number of particles in this case)
        b (int): batch size
        broadcast (bool): if True, after pooling re-broadcast to original shape

    Returns:
        tensor of shape (c, k) if broadcast, else (b*N, k)
    """
    X_pooled = tf.unsorted_segment_mean(X, idx, num_segs)
    if broadcast:
        X_pooled = tf.gather_nd(X_pooled, tf.expand_dims(idx, axis=1))
    return X_pooled


def ShiftInv_layer(H_in, COO_feats, bN, layer_id, is_last=False):
    """
    Args:
        H_in (tensor): (c, k), stores shift-invariant edge features, row-major
          - c = b*N*M, if KNN then M is fixed, and k = num_edges = num_neighbors = M
        COO_feats (tensor): (3, c), of row, column, cube-wise indices respectively
        bN (tuple(int)): (b, N), where b is batch_size, N is number of particles
        layer_id (int): id of layer in network, for retrieving variables
          - each layer has 4 weights W (k, q), and 1 bias B (q,)
        is_last (bool): if is_last, pool output over columns
    Returns:
        H_out (tensor): (c, q), or (b, N, q) if is_last
    """
    # Prepare data and parameters
    # ========================================
    # split inputs
    b, N = bN
    #row_idx, col_idx, cube_idx = tf.split(COO_feats, 3, axis=0)
    row_idx  = COO_feats[0]
    col_idx  = COO_feats[1]
    cube_idx = COO_feats[2]

    # get layer weights
    W1, W2, W3, W4, B = utils.get_scoped_ShiftInv_layer_vars(layer_id)

    # Helper funcs
    # ========================================
    def _pool(H, idx, broadcast=True):
        # row : col
        # col : row
        # cubes : cubes
        return pool_graph(H, idx, b*N, broadcast)

    def _left_mult(h, W):
        return tf.einsum("ck,kq->cq", h, W)

    # Layer forward pass
    # ========================================
    # H1 - no pooling
    H1 = _left_mult(H_in, W1) # (c, q)

    # H2 - pool rows
    H_pooled_rows = _pool(H_in, col_idx)
    H2 = _left_mult(H_pooled_rows, W2) # (c, q)

    # H3 - pool cols
    H_pooled_cols = _pool(H_in, row_idx)
    H3 = _left_mult(H_pooled_cols, W3) # (c, q)

    # H4 - pool cubes
    H_pooled_all = _pool(H_in, cube_idx)
    H4 =  _left_mult(H_pooled_all, W4) # (c, q)

    # Output
    # ========================================
    H_out = (H1 + H2 + H3 + H4) + B
    if is_last:
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        H_out = tf.reshape(_pool(H_out, row_idx, broadcast=False), (b, N, -1))
    return H_out


#------------------------------------------------------------------------------
# Shift invariant network ops
#------------------------------------------------------------------------------
def include_node_features(X_in_edges, X_in_nodes, COO_feats, redshift=None):
    """ Broadcast node features to edges for input layer
    Args:
        X_in_edges (tensor):   (c, 3), input edge features (relative pos of neighbors)
        X_in_nodes (tensor): (b*N, 3), input node features (velocities)
        COO_feats (tensor): (3, c), rows, cols, cube indices (respectively)
    Returns:
        X_in_graph (tensor): (c, 9), input with node features broadcasted to edges
    """
    # get row, col indices
    row_idx = COO_feats[0]
    col_idx = COO_feats[1]

    # get node row, columns
    node_rows = tf.gather_nd(X_in_nodes, tf.expand_dims(row_idx, axis=1))
    node_cols = tf.gather_nd(X_in_nodes, tf.expand_dims(col_idx, axis=1))

    # full node, edges graph
    X_in_graph = tf.concat([X_in_edges, node_rows, node_cols], axis=1) # (c, 9)

    # broadcast redshifts (for multi)
    if redshift is not None:
        X_in_graph = tf.concat([X_in_graph, redshift], axis=1) # (c, 10)

    return X_in_graph


# ==== Network fn
def ShiftInv_network_func(X_in_edges, X_in_nodes, COO_feats, num_layers, dims, activation, redshift=None):
    # Input layer
    # ========================================
    H_in = include_node_features(X_in_edges, X_in_nodes, COO_feats, redshift=redshift)
    H = activation(ShiftInv_layer(H_in, COO_feats, dims, 0))

    # Hidden layers
    # ========================================
    for layer_idx in range(1, num_layers):
        is_last = layer_idx == num_layers - 1
        H = ShiftInv_layer(H, COO_feats, dims, layer_idx, is_last=is_last)
        if not is_last:
            H = activation(H)
    return H

# ==== Network fn
def ShiftInv_network_func_rs(X_in_edges, X_in_nodes, COO_feats, num_layers, dims, activation, redshift):
    # Just concats the rs at every layer except output

    # Input layer
    # ========================================
    H_in = include_node_features(X_in_edges, X_in_nodes, COO_feats, redshift=redshift)
    H = activation(ShiftInv_layer(H_in, COO_feats, dims, 0))
    H = tf.concat([H, redshift], axis=-1) # RS ccat'd

    # Hidden layers
    # ========================================
    for layer_idx in range(1, num_layers):
        is_last = layer_idx == num_layers - 1
        H = ShiftInv_layer(H, COO_feats, dims, layer_idx, is_last=is_last)
        if not is_last:
            H = activation(H)
            H = tf.concat([H, redshift], axis=-1) # RS ccat'd
    return H

#------------------------------------------------------------------------------
# Shift invariant model funcs
#------------------------------------------------------------------------------
# ==== Model fn
def ShiftInv_model_func(X_in_edges, X_in_nodes, COO_feats, model_specs, redshift=None):
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope
    num_layers = model_specs.num_layers
    use_vcoeff = model_specs.vcoeff
    activation = model_specs.activation_func # default tf.nn.relu
    dims = model_specs.dims

    # Network forward
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        H_out = ShiftInv_network_func(X_in_edges, X_in_nodes, COO_feats, num_layers, dims, activation, redshift)

        # skip connections
        if use_vcoeff:
            theta = utils.get_scoped_vcoeff()
            H_out = theta * H_out
    return H_out
"""
Impl notes for multi:
 - There may be too many pre/post processing steps to effectively do this
   - may need to keep intermediate predictions in "edge/node" form, and only pool
     for the final redshift prediction
"""
# ==== single fn
def ShiftInv_single_model_func(X_in, COO_feats, redshift, model_specs, coeff_idx=None):
    """
    Args:
        X_in (tensor): (b, N, 6)
        COO_feats (tensor): (3, B*N*M), segment ids for rows, cols, all
        redshift (tensor): (b*N*M, 1) redshift broadcasted
    """
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope
    num_layers = model_specs.num_layers
    activation = model_specs.activation_func # default tf.nn.relu
    dims = model_specs.dims

    # Get graph inputs
    # ========================================
    edges, nodes = get_input_features_TF(X_in, COO_feats, dims)

    # Network forward
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        H_out = ShiftInv_network_func(edges, nodes, COO_feats, num_layers, dims[:-1], activation, redshift)
        # skip connections
        if coeff_idx is not None:
            #theta = utils.get_scoped_coeff_multi(coeff_idx)
            t0, t1 = utils.get_scoped_coeff_multi2(coeff_idx) # (1,)
            #h_coo = H_out_coo +  theta * X_in_vel
            h_coo = H_out[...,:3] * t0
            #h_vel = H_out[...,3:] * t1
            timestep = X_in[...,3:]  * t1
            h_out_coo = h_coo + timestep
            H_out = tf.concat([h_out_coo, H_out[...,3:]], axis=-1)
    #return H_out
    return get_readout(X_in + H_out)

# ==== single fn
def ShiftInv_single_model_func(X_in, COO_feats, model_specs, timestep, scalar_tag, redshift=None):
    """
    Args:
        X_in (tensor): (b, N, 6)
        COO_feats (tensor): (3, B*N*M), segment ids for rows, cols, all
        redshift (tensor): (b*N*M, 1) redshift broadcasted
    """
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope
    num_layers = model_specs.num_layers
    #use_vcoeff = model_specs.vcoeff
    activation = model_specs.activation_func # default tf.nn.relu
    dims = model_specs.dims

    # Get graph inputs
    # ========================================
    edges, nodes = get_input_features_TF(X_in, COO_feats, dims)

    # Network forward V1.
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        #t0, t1 = utils.get_scoped_coeff_multi2(coeff_idx)
        loc_scalar = utils.get_scoped_coeff(scalar_tag)
        H_out = ShiftInv_network_func(edges, nodes, COO_feats, num_layers, dims[:-1], activation, redshift)
        num_feats = H_out.get_shape().as_list()[-1]

        if num_feats <= 3: # then only predicting location
            H_out = H_out*loc_scalar + X_in[...,:3] + X_in[...,3:] * timestep

        else: # predicting velocity
            h_coo = H_out[...,:3]*loc_scalar + X_in[...,:3] + X_in[...,3:] * timestep
            h_vel = H_out[...,3:] + X_in[...,3:]
            H_out = tf.concat([h_coo, h_vel], axis=-1)
        return get_readout(H_out)


# ==== single fn
def ShiftInv_model_func_timestep_rs(X_in, COO_feats, model_specs, redshift, timestep, scalar_tag):
    """
    Args:
        X_in (tensor): (b, N, 6)
        COO_feats (tensor): (3, B*N*M), segment ids for rows, cols, all
        redshift (tensor): (b*N*M, 1) redshift broadcasted
    """
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope
    num_layers = model_specs.num_layers
    activation = model_specs.activation_func # default tf.nn.relu
    dims = model_specs.dims

    # Get graph inputs
    # ========================================
    edges, nodes = get_input_features_TF(X_in, COO_feats, dims)

    # Network forward V1.
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        # network output
        #net_out = ShiftInv_network_func(edges, nodes, COO_feats, num_layers, dims[:-1], activation, redshift)
        net_out = ShiftInv_network_func_rs(edges, nodes, COO_feats, num_layers, dims[:-1], activation, redshift)

        # Scaling and skip connections
        loc_scalar = utils.get_scoped_coeff(scalar_tag)
        num_feats = net_out.get_shape().as_list()[-1]
        H_out = net_out[...,:3]*loc_scalar + X_in[...,:3] + X_in[...,3:] * timestep

        # Concat velocity predictions
        if net_out.get_shape().as_list()[-1] > 3:
            H_vel = net_out[...,3:] + X_in[...,3:]
            H_out = tf.concat([H_out, H_vel], axis=-1)

        return get_readout(H_out)

# ==== single fn
def ShiftInv_model_func_timestep_old(X_in, COO_feats, model_specs, scalar_tag, redshift=None):
    """
    Args:
        X_in (tensor): (b, N, 6)
        COO_feats (tensor): (3, B*N*M), segment ids for rows, cols, all
        redshift (tensor): (b*N*M, 1) redshift broadcasted
    """
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope
    num_layers = model_specs.num_layers
    activation = model_specs.activation_func # default tf.nn.relu
    dims = model_specs.dims

    # Get graph inputs
    # ========================================
    edges, nodes = get_input_features_TF(X_in, COO_feats, dims)

    # Network forward V1.
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        # network output
        net_out = ShiftInv_network_func(edges, nodes, COO_feats, num_layers, dims[:-1], activation, redshift)

        # Scaling and skip connections
        #loc_scalar = utils.get_scoped_coeff(scalar_tag)
        #loc_scalar, vel_scalar = utils.get_scoped_scalars(scalar_tag)
        loc_scalar, vel_scalar = utils.get_scoped_coeff_multi2(scalar_tag)
        num_feats = net_out.get_shape().as_list()[-1]

        H_out = net_out[...,:3]*loc_scalar + X_in[...,:3] + X_in[...,3:]*vel_scalar


        '''
        # timestep --> make optimizable var like loc_scalar
        # - scalar over entire net out (loc & vel)
        # - scalar for velocity (but not timestep)
        #   - for input vel on loc, and on vel
        # - init the scalars (loc: try 1.0, vel: 0.01)
        # - share scalars across redshifts
        '''

        # Concat velocity predictions
        if net_out.get_shape().as_list()[-1] > 3:
            H_vel = net_out[...,3:] * vel_scalar + X_in[...,3:] # Maybe another scalar here for velocity
            #H_vel = net_out[...,3:]*vel_scalar + X_in[...,3:] # Maybe another scalar here for velocity
            H_out = tf.concat([H_out, H_vel], axis=-1)

        return get_readout(H_out)

# ==== single fn
def ShiftInv_model_func_timestep(X_in, COO_feats, model_specs, redshift=None):
    """
    Args:
        X_in (tensor): (b, N, 6)
        COO_feats (tensor): (3, B*N*M), segment ids for rows, cols, all
        redshift (tensor): (b*N*M, 1) redshift broadcasted
    """
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope
    num_layers = model_specs.num_layers
    activation = model_specs.activation_func # default tf.nn.relu
    dims = model_specs.dims

    # Get graph inputs
    # ========================================
    edges, nodes = get_input_features_TF(X_in, COO_feats, dims)

    # Network forward V1.
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        # network output
        #net_out = ShiftInv_network_func(edges, nodes, COO_feats, num_layers, dims[:-1], activation, redshift)
        net_out = ShiftInv_network_func_rs(edges, nodes, COO_feats, num_layers, dims[:-1], activation, redshift)

        # Scaling and skip connections
        #loc_scalar = utils.get_scoped_coeff(scalar_tag)
        #loc_scalar, vel_scalar = utils.get_scoped_scalars(scalar_tag)
        loc_scalar, vel_scalar = utils.get_scoped_coeff_single()
        num_feats = net_out.get_shape().as_list()[-1]

        H_out = net_out[...,:3]*loc_scalar + X_in[...,:3] + X_in[...,3:]*vel_scalar


        '''
        # timestep --> make optimizable var like loc_scalar
        # - scalar over entire net out (loc & vel)
        # - scalar for velocity (but not timestep)
        #   - for input vel on loc, and on vel
        # - init the scalars (loc: try 1.0, vel: 0.01)
        # - share scalars across redshifts
        '''

        # Concat velocity predictions
        if net_out.get_shape().as_list()[-1] > 3:
            H_vel = net_out[...,3:] * vel_scalar + X_in[...,3:] # Maybe another scalar here for velocity
            #H_vel = net_out[...,3:]*vel_scalar + X_in[...,3:] # Maybe another scalar here for velocity
            H_out = tf.concat([H_out, H_vel], axis=-1)

        return get_readout(H_out)

# ==== single fn
def ShiftInv_model_func_timestep_ACCEL(X_in, COO_feats, model_specs, scalar_tag, redshift=None):
    """
    Args:
        X_in (tensor): (b, N, 6)
        COO_feats (tensor): (3, B*N*M), segment ids for rows, cols, all
        redshift (tensor): (b*N*M, 1) redshift broadcasted
    """
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope
    num_layers = model_specs.num_layers
    activation = model_specs.activation_func # default tf.nn.relu
    dims = model_specs.dims

    # Get graph inputs
    # ========================================
    edges, nodes = get_input_features_TF(X_in, COO_feats, dims)

    # Network forward V1.
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        # network output
        A = ShiftInv_network_func(edges, nodes, COO_feats, num_layers, dims[:-1], activation, redshift)

        # Scaling and skip connections
        #loc_scalar = utils.get_scoped_coeff(scalar_tag)
        #loc_scalar, vel_scalar = utils.get_scoped_coeff_multi2(scalar_tag)
        T1, T2 = utils.get_scoped_coeff_multi2(scalar_tag)
        num_feats = net_out.get_shape().as_list()[-1]

        #H_out = net_out[...,:3]*loc_scalar + X_in[...,:3] + X_in[...,3:]*vel_scalar
        #H_out = A
        H_vel = X_in[...,3:] + A

        # Concat velocity predictions
        if net_out.get_shape().as_list()[-1] > 3:
            H_vel = net_out[...,3:] * vel_scalar + X_in[...,3:] # Maybe another scalar here for velocity
            #H_vel = net_out[...,3:]*vel_scalar + X_in[...,3:] # Maybe another scalar here for velocity
            H_out = tf.concat([H_out, H_vel], axis=-1)

        return get_readout(H_out)


# ==== single fn
#def vel_network_func(X_in, activation):

def vel_single_model_func(X_in, model_specs, coeff_idx):
    """
    Args:
        X_in (tensor): (b, N, 6)
    """
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope

    # Network forward
    # ========================================
    with tf.variable_scope(var_scope, reuse=True):
        #t1 = utils.get_scoped_coeff_multi(coeff_idx)
        t0, t1 = utils.get_scoped_coeff_multi2(coeff_idx)
        x_in_loc = X_in[...,:3]
        x_in_vel = X_in[...,3:]

        x_out_loc = x_in_loc + x_in_vel * t1
        x_out_vel = x_in_vel * t0
        #x_vel = X_in[...,3:] * t1
        #x_loc = X_in[...,:3]
        X_out = tf.concat([x_out_loc, x_out_vel], axis=-1)
        #X_out = x_loc * t0
    return get_readout(X_out)

# ==== Multi-A Model fn
def ShiftInv_multi_model_func(X_in, COO_feats, redshifts, model_specs, use_coeff=False):
    """
    Args:
        X_in (tensor): (b, N, 6)
        COO_feats (tensor): (num_rs, 3, b*N*M), segment ids for rows, cols, all
    """
    # Get relevant model specs
    # ========================================
    num_rs = len(redshifts) - 1
    b, N, M = model_specs.dims # (b, N, M)

    # Helpers
    # ========================================
    def _ShiftInv_fwd(h_in, rs_idx):
        coo = COO_feats[rs_idx]
        rs = tf.fill([b*N*M, 1], redshifts[rs_idx])
        cidx = rs_idx if use_coeff else None
        return ShiftInv_single_model_func_v1(h_in, coo, model_specs, rs, cidx)

    # Network forward
    # ======================================== # don't think you need to do var scope here
    h_pred = _ShiftInv_fwd(X_in, 0)
    for i in range(1, num_rs):
        h_pred = _ShiftInv_fwd(h_pred, i)
    return h_pred


#=============================================================================
# LAYER OPS
#=============================================================================
def left_mult(h, W):
    """ batch matmul for set-based data
    """
    return tf.einsum('ijl,lq->ijq', h, W)

#==== set ops
def set_layer(h, layer_idx, var_scope, *args):
    """ Set layer
    *args just for convenience, set_layer has no additional
    Args:
        h: data tensor, (mb_size, N, k_in)
        layer_idx (int): layer index for params
        var_scope (str): variable scope for get variables from graph
    RETURNS: (mb_size, N, k_out)
    """
    W, B = utils.get_layer_vars(layer_idx, var_scope=var_scope)
    mu = tf.reduce_mean(h, axis=1, keepdims=True)
    h_out = left_mult(h - mu, W) + B
    return h_out

#==== graph ops

# Kgraph ops

def kgraph_conv(h, adj, K):
    """ Graph convolution op for KNN-based adjacency lists
    build graph tensor with gather_nd, and take
    mean on KNN dim: mean((mb_size, N, K, k_in), axis=2)
    Args:
        h: data tensor, (mb_size, N, k_in)
        adj: adjacency index list, for gather_nd (*, 2)
        K (int): num nearest neighbors
    """
    dims = tf.shape(h)
    mb = dims[0]; n  = dims[1]; d  = dims[2];
    rdim = [mb,n,K,d]
    nn_graph = tf.reduce_mean(tf.reshape(tf.gather_nd(h, adj), rdim), axis=2)
    return nn_graph

def kgraph_layer(h, layer_idx, var_scope, alist, K):
    """ Graph layer for KNN

    Args:
        h: data tensor, (mb_size, N, k_in)
        layer_idx (int): layer index for params
        var_scope (str): variable scope for get variables from graph
        alist: adjacency index list tensor (*, 2), of tf.int32
        K (int): number of nearest neighbors in KNN
    RETURNS: (mb_size, N, k_out)
    """
    W, B = utils.get_layer_vars(layer_idx, var_scope=var_scope)
    graph_mu = kgraph_conv(h, alist, K)

    h_out = left_mult(h - graph_mu, W) + B
    return h_out

# Radius graph ops

def rad_graph_conv(h, spT):
    """ graph conv for radius neighbors graph
    NB: the sparse tensor for the radius graph has ALREADY been processed
    such that the mean pooling is performed in the matmul

    Args:
        h: tensor of shape (b, N, D)
        spT: sparse_tensor of shape (b*N, b*N)
    """
    dims = tf.shape(h)
    h_flat = tf.reshape(h, (-1, dims[-1]))
    rad_conv = tf.reshape(tf.sparse_tensor_dense_matmul(spT, h_flat), dims)
    return rad_conv

def rad_graph_layer(h, layer_idx, var_scope, spT):
    #W, Wg = utils.get_graph_layer_vars(layer_idx, var_scope=var_scope)
    W, B = utils.get_layer_vars(layer_idx, var_scope=var_scope)
    graph_mu = rad_graph_conv(h, spT)

    h_out = left_mult(h - graph_mu, W) + B
    return h_out

#=============================================================================
# Network and model functions
#=============================================================================

#==== helpers
def get_layer(args):
    # hacky dispatch on num of args. set uses no extra, radgraph 1, kgraph 2
    num_args = len(args)
    if num_args > 0:
        layer = rad_graph_layer if num_args == 1 else kgraph_layer
    else:
        layer = set_layer
    return layer

def skip_connection(x_in, h, vel_coeff=None):
    # input splits
    x_coo = x_in[...,:3]
    x_vel = x_in[...,3:] if x_in.shape[-1] == h.shape[-1] else x_in[...,3:-1]
    # h splits
    h_coo = h[...,:3]
    h_vel = h[...,3:]
    # add
    h_coo += x_coo
    h_vel += x_vel

    if vel_coeff is not None:
        h_coo += vel_coeff * x_vel
    h_out = tf.concat((h_coo, h_vel), axis=-1)
    return h_out

# ==== Network fn
def network_fwd(x_in, num_layers, var_scope, *args, activation=tf.nn.relu):
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    layer = get_layer(args)
    H = x_in
    for i in range(num_layers):
        H = layer(H, i, var_scope, *args)
        if i != num_layers - 1:
            H = activation(H)
    return H

# ==== Model fn
def model_fwd(x_in, num_layers, *args, activation=tf.nn.relu, add=True, vel_coeff=False, var_scope=VAR_SCOPE):
    h_out = network_fwd(x_in, num_layers, var_scope, *args, activation=activation)
    if add:
        vcoeff = utils.get_vel_coeff(var_scope) if vel_coeff else None
        #h_out = skip_connection(x_in, h_out, vcoeff)
        h_out = h_out + x_in[...,:3] + vcoeff*x_in[...,3:-1]
    return h_out

#=============================================================================
# Multi-step model functions
#=============================================================================
# multi fns for single step trained models
def aggregate_multiStep_fwd(x_rs, num_layers, var_scopes, nn_graph, *args, vel_coeff=False):
    """ Multi-step function for aggregate model
    Aggregate model uses a different sub-model for each redshift
    """
    # Helpers
    concat_rs = lambda h, i: tf.concat((h, x_rs[i,:,:,-1:]), axis=-1)
    fwd = lambda h, i: get_readout(model_fwd(h, num_layers, nn_graph[i], *args, var_scope=var_scopes[i], vel_coeff=vel_coeff))
    loss = lambda h, x: pbc_loss(h, x[...,:-1], )#True)

    # forward pass
    h = fwd(x_rs[0], 0)
    error = loss(h, x_rs[1])
    for i in range(1, len(var_scopes)):
        h_in = concat_rs(h, i)
        h = fwd(h_in, i)
        error += loss(h, x_rs[i+1])
    return h, error

def aggregate_multiStep_fwd_validation(x_rs, num_layers, var_scopes, graph_fn, *args, vel_coeff=False):
    """ Multi-step function for aggregate model
    Aggregate model uses a different sub-model for each redshift
    WILL NOT WORK FOR SPARSETENSOR, since tf.py_func does not allow for sparse
    return
    """
    preds = []
    # Helpers
    concat_rs = lambda h, i: tf.concat((h, x_rs[i,:,:,-1:]), axis=-1)
    fwd = lambda h, g, i: get_readout(model_fwd(h, num_layers, g, *args, var_scope=var_scopes[i], vel_coeff=vel_coeff))

    # forward pass
    g = tf.py_func(graph_fn, [x_rs[0]], tf.float32)
    h = fwd(x_rs[0], g, 0)
    preds.append(h)
    for i in range(1, len(var_scopes)):
        h_in = concat_rs(h, i)
        g = tf.py_func(graph_fn, [h_in], tf.float32)
        h = fwd(h_in, g, i)
        preds.append(h)
    return preds

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# END TF-related ops
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

#=============================================================================
# adj list ops for shift inv data
#=============================================================================

def get_input_features_TF(X_in, coo, dims):
    """ get edges and nodes with TF ops
    TESTED EQUIVALENT TO numpy-BASED FUNC

    get relative distances of each particle from its M neighbors
    Args:
        X_in (tensor): (b, N, 6), input data
        coo (tensor): (3,b*N*M)
    """
    cols = coo[1]
    b, N, M = dims
    # split input
    X = tf.reshape(X_in, (-1, 6))
    edges = X[...,:3] # loc
    nodes = X[...,3:] # vel

    # get edges (neighbors)
    edges = tf.reshape(tf.gather(edges, cols), [b, N, M, 3])
    edges = edges - tf.expand_dims(X_in[...,:3], axis=2) # (b, N, M, 3) - (b, N, 1, 3)

    return tf.reshape(edges, [-1, 3]), nodes


def get_input_edge_features_batch(X_in, lst_csrs, M, offset_idx=False):
    """ get relative distances of each particle from its M neighbors
    Args:
        X_in (ndarray): (b, N, 6), input data
        lst_csrs (list(csr)): len b list of csrs
    """
    x = X_in[...,:3] # (b, N, 3)
    b, N, k = x.shape

    #x_out = np.zeros((b, N*M, k)).astype(np.float32)
    x_out = np.zeros((b, N, M, k)).astype(np.float32)
    # have to loop, since mem issues with full X_in[adj_list]
    for i in range(b):
        h = x[i]     # (N, 3)
        a = lst_csrs[i].indices.reshape(N, M) # (N, M)
        h_sel = h[a] # (N, M, 3)
        h_out = h_sel - h[:,None,:] # (N, M, 3) - (N, 1, 3)
        x_out[i] = h_out
    return x_out.reshape(-1, k)


def get_input_node_features(X_in):
   """ get node values (velocity vectors) for each particle
   X_in.shape == (b, N, 6)
   """
   return X_in[...,3:].reshape(-1, 3)

def sinv_dim_change(X_in):
    return np.moveaxis(X_in, 0, -1) # now (N,...,b)


#=============================================================================
# numpy adjacency list func wrappers
#=============================================================================

def get_adj_graph(X_in, k, pbc_threshold=None):
    """ neighbor graph interface func
    NB: kgraph and radgraph have VERY different returns. Not exchangeable.

    Args:
        X_in (ndarray): data
        k (int or float): nearest neighbor variable, int is kgraph, float is rad
        pbc_threshold (float): boundary threshold for pbc
    """
    graph_fn = get_kgraph if isinstance(k, int) else get_radgraph
    return graph_fn(X_in, k, pbc_threshold)

def get_kgraph(X, k, pbc_threshold=None):
    return alist_to_indexlist(get_kneighbor_alist(X, k))

def get_radgraph(X, k, pbc_threshold=None):
    return radius_graph_fn(X, k)


def get_pbc_graph(x, graph_fn, threshold):
    """
    x.shape = (N, 3)
    """
    assert False # TODO


def alist_to_indexlist(alist):
    """ Reshapes adjacency list for tensorflow gather_nd func
    alist.shape: (B, N, K)
    ret.shape:   (B*N*K, 2)
    """
    batch_size, N, K = alist.shape
    id1 = np.reshape(np.arange(batch_size),[batch_size,1])
    id1 = np.tile(id1,N*K).flatten()
    out = np.stack([id1,alist.flatten()], axis=1).astype(np.int32)
    return out

def get_kneighbor_alist(X_in, K=14, offset_idx=False, inc_self=True):
    """ search for K nneighbors, and return offsetted indices in adjacency list
    No periodic boundary conditions used

    Args:
        X_in (numpy ndarray): input data of shape (mb_size, N, 6)
    """
    mb_size, N, D = X_in.shape
    adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
    for i in range(mb_size):
        # this returns indices of the nn
        graph_idx = kneighbors_graph(X_in[i, :, :3], K, include_self=inc_self).indices
        graph_idx = graph_idx.reshape([N, K]) #+ (N * i)  # offset idx for batches
        if offset_idx:
            graph_idx += (N*i)
        adj_list[i] = graph_idx
    return adj_list

def get_kneighbor_list(X_in, M, offset_idx=False, inc_self=False):
    b, N, D = X_in.shape
    lst_csrs = []
    for i in range(b):
        kgraph = kneighbors_graph(X_in[i,:,:3], M, include_self=inc_self).astype(np.float32)
        if offset_idx:
            kgraph.indices = kgraph.indices + (N * i)
        lst_csrs.append(kgraph)
    return lst_csrs


def to_coo_batch(A):
    """ Get row and column indices from csr
    DOES NOT LIKE OFFSET IDX, tocoo() method will complain about index being
    greater than matrix size

    Args:
        A (csr): list of csrs of shape (N, N)
    """
    # Dims
    # ----------------
    b = len(A) # batch size
    N = A[0].shape[0] # (32**3)
    M = A[0].indices.shape[0] // N

    # Get COO feats
    # ----------------
    COO_feats = np.zeros((3, b*N*M)).astype(np.int32)
    for i in range(b):
        coo = A[i].tocoo()

        # Offset coo feats
        row = coo.row + i*N
        col = coo.col + i*N
        cube = np.zeros_like(row) + i

        # Assign coo feats
        k, q = i*N*M, (i+1)*N*M
        COO_feats[0, k:q] = row
        COO_feats[1, k:q] = col
        COO_feats[2, k:q] = cube

    # sanity check
    #confirm_CSR_to_COO_index_integrity(A, COO_feats) # checked out
    return COO_feats

def confirm_CSR_to_COO_index_integrity(A, COO_feats):
    """ CSR.indices compared against COO.cols
    Sanity check to ensure that my indexing algebra is correct
    """
    CSR_feats = get_indices_from_list_CSR(A)
    cols = COO_feats[1]
    assert np.all(CSR_feats == cols)

def get_indices_from_list_CSR(A, offset=True):
    # Dims
    # ----------------
    b = len(A) # batch size
    N = A[0].shape[0] # (32**3)
    M = A[0].indices.shape[0] // N

    # Get CSR feats (indices)
    # ----------------
    CSR_feats = np.zeros((b*N*M)).astype(np.int32)
    for i in range(b):
        # Offset indices
        idx = A[i].indices + i*N

        # Assign csr feats
        k, q = i*N*M, (i+1)*N*M
        CSR_feats[k:q] = idx
    return CSR_feats



def pre_process_adjacency(A):
    """
    Batch of adjacency matrices <> row, col offset indices.
    Nodes can have different number of neighbors (e.g., density based).

    NOTE: this returns DIFFERENT col than coo-matrix conversion
     - cols with nonzero are sorted. Whether this is needed for the pooling op
       I'm not sure. Cols are always set-wise same as coo.
     - cols in coo are based on csr.indices and csr.indptr, which means indices
       are not sorted in scalar order, but either arbitrary or NN distance

    This is also FAR MORE computationally expensive than coo conversion

    Args:
        A (numpy array): adjacency batch, has shape (b, N, N)

    Returns:
        row_idx (numpy array), col_idx (numpy array), all_idx(numpy array):
            all have shape (c), where c = sum_{i=0..b} n_i, and n_i is the number of non-zero entries of
            the i-th adjacency in the batch.
                - if all matrices in the batch have the same number n of non zero-entries, c = b*n
                - if the number of neighbors is fixed to M, then n = N*M and c = b*N*M

        row_idx, col_idx are indices of non-zero entries
        all_idx keeps track of how many non-zero entries are in each element of the batch - needed for _pool_all operation
    """
    b = len(A)
    N = A[0].shape[0]

    row_idx = []
    col_idx = []
    all_idx = []
    for i in range(b):
        a = A[i].toarray()
        r_idx, c_idx = np.nonzero(a)
        row_idx.extend(i * N + r_idx)  # offset indices
        col_idx.extend(i * N + c_idx)  # offset indices
        all_idx.extend(i + np.zeros_like(r_idx))  # offset indices

    return np.array(row_idx), np.array(col_idx), np.array(all_idx)


#=============================================================================
# RADIUS graph ops
#=============================================================================

def radius_graph_fn(x, R):
    """ Wrapper for sklearn.Neighbors.radius_neighbors_graph function

    Args:
        x (ndarray): input data of shape (N, D), where x[:,:3] == coordinates
        R (float): radius for search
    """
    return radius_neighbors_graph(x[:,:3], R, include_self=True).astype(np.float32)

def get_radNeighbor_coo(X_in, R):
    N = X_in.shape[0]
    # just easier to diff indptr for now
    # get csr
    rad_csr = radius_graph_fn(X_in, R)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    rad_coo = rad_csr.tocoo()

    # diff data for matmul op select
    div_diff = np.diff(rad_csr.indptr)
    coo_data_divisor = np.repeat(div_diff, div_diff).astype(np.float32)
    coo_data = rad_coo.data / coo_data_divisor

    coo = coo_matrix((coo_data, (rad_coo.row, rad_coo.col)), shape=(N, N)).astype(np.float32)
    return coo

def get_radNeighbor_coo_batch(X_in, R):
    b, N = X_in.shape[:2]

    # accumulators
    coo = get_radNeighbor_coo(X_in[0], R)
    rows = coo.row
    cols = coo.col
    data = coo.data

    for i in range(1, b):
        # get coo, offset indices
        coo = get_radNeighbor_coo(X_in[i], R)
        row = coo.row + (N * i)
        col = coo.col + (N * i)
        datum = coo.data

        # concat to what we have
        rows = np.concatenate((rows, row))
        cols = np.concatenate((cols, col))
        data = np.concatenate((data, datum))

    coo = coo_matrix((data, (rows, cols)), shape=(N*b, N*b)).astype(np.float32)
    return coo

def get_radNeighbor_sparseT_attributes(coo):
    idx = np.mat([coo.row, coo.col]).transpose()
    return idx, coo.data, coo.shape

def get_radius_graph_input(X_in, R):
    coo = get_radNeighbor_coo_batch(X_in, R)
    sparse_tensor_attributes = get_radNeighbor_sparseT_attributes(coo)
    return sparse_tensor_attributes



#=============================================================================
# boundary utils
#=============================================================================
def face_outer(particle, bound): # ret shape (1,3)
    # face only has one coordinate in boundary, so only one relocation
    ret = bound + particle
    return ret[None,:]

def edge_outer(particle, bound):
    # edge has two coordinates in boundary, so 3 relocations (edge, face, face)
    zero_idx = list(bound).index(0)
    edge = np.roll(np.array([[0,1,1],[0,1,0],[0,0,1]]), zero_idx, 1)
    return (edge * bound) + particle

def corner_outer(particle, bound): # ret shape (7, 3)
    # corner has 3 coordinates in boundary, so 7 relocations:
    # (corner, edge, edge, edge, face, face, face)
    corner = np.array([[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1]])
    return (corner * bound) + particle

def get_outer(particle, bound, num_boundary):
    assert num_boundary > 0
    if num_boundary == 1:
        return face_outer(particle, bound)
    elif num_boundary == 2:
        return edge_outer(particle, bound)
    else:
        return corner_outer(particle, bound)

def pad_cube_boundaries(x, boundary_threshold):
    """ check all particles for boundary conditions and
    relocate boundary particles
    I wonder if you could just do one corner_outer over the extracted corners
    in x, edge_outer on extracted edges, and so forth, while saving indices?
    Args:
        x (ndarray): data array, shape (n_P, 3)
    Returns: expanded x, index_list
    """
    N, D = x.shape
    idx_list = np.array([], dtype=np.int32)

    # boundary
    lower = boundary_threshold
    upper = 1 - boundary_threshold
    bound_x = np.where(x >= upper, -1, np.where(x <= lower, 1, 0))
    bound_x_count = np.count_nonzero(bound_x, axis=-1)

    # get bound and add to clone
    for idx in range(N):
        num_boundary = bound_x_count[idx]
        if num_boundary > 0:
            # get particles to add to clone
            outer_particles = get_outer(x[idx], bound_x[idx], num_boundary)
            # add indices
            idx_list = np.append(idx_list, [idx] * outer_particles.shape[0])
            # concat to clone
            x = np.concatenate((x, outer_particles), axis=0)
    return x, idx_list

def get_pcube_adjacency_list(x, idx_map, N, K):
    """ get kneighbor graph from padded cube
    x is padded cube of shape (M, 3),
    where M == (N + number of added boundary particles)
    Args:
        x (ndarray): padded cube, of shape (M, 3)
        idx_map (ndarray): shape (M-N,) indices
        N: number of particles in original cube
        K: number of nearest neighbors
    """
    kgraph = kneighbors_graph(x, K, include_self=True)[:N].indices
    kgraph_outer = kgraph >= N
    for k_idx, is_outer in enumerate(kgraph_outer):
        if is_outer:
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            outer_idx = kgraph[k_idx]
            kgraph[k_idx] = idx_map[outer_idx - N]
    return kgraph.reshape(N,K)

def get_pbc_kneighbors(X, K, boundary_threshold):
    """
    """
    # get boundary range
    lower = boundary_threshold
    upper = 1 - boundary_threshold
    mb_size, N, D = X.shape

    # graph init
    adjacency_list = np.zeros((mb_size, N, K), dtype=np.int32)

    for b in range(mb_size):
        # get expanded cube
        clone = np.copy(X[b,:,:3])
        padded_cube, idx_map = pad_cube_boundaries(clone, boundary_threshold)

        # get neighbors from padded_cube
        kgraph_idx = get_pcube_adjacency_list(padded_cube, idx_map, N, K)
        adjacency_list[b] = kgraph_idx
    return adjacency_list


def get_pcube_csr(x, idx_map, N, K, include_self=False):
    """ get kneighbor graph from padded cube
    x is padded cube of shape (M, 3),
    where M == (N + number of added boundary particles)
    Args:
        x (ndarray): padded cube, of shape (M, 3)
        idx_map (ndarray): shape (M-N,) indices
        N: number of particles in original cube
        K: number of nearest neighbors
    """
    kgraph = kneighbors_graph(x, K, include_self=include_self)[:N]
    kgraph_outer_idx = kgraph.indices >= N
    for k_idx, is_outer in enumerate(kgraph_outer_idx):
        if is_outer:
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            outer_idx = kgraph.indices[k_idx]
            kgraph.indices[k_idx] = idx_map[outer_idx - N]
    return kgraph

def get_pbc_kneighbors_csr(X, K, boundary_threshold, include_self=False):
    """
    """
    # get boundary range
    lower = boundary_threshold
    upper = 1 - boundary_threshold
    mb_size, N, D = X.shape

    # graph init
    #adjacency_list = np.zeros((mb_size, N, K), dtype=np.int32)
    csr_list = []
    clone = np.copy(X[...,:3])

    for b in range(mb_size):
        # get expanded cube
        clone_cube = clone[b]
        padded_cube, idx_map = pad_cube_boundaries(clone_cube, boundary_threshold)

        # get neighbors from padded_cube
        kgraph = get_pcube_csr(padded_cube, idx_map, N, K, include_self)
        csr_list.append(kgraph)
    return csr_list


#=============================================================================
# periodic boundary conditions, loss
#=============================================================================
def get_readout(h_out):
    M = h_out.get_shape().as_list()[-1]

    # bounding coo
    h_out_coo = h_out[...,:3]
    gt_one  = (tf.sign(h_out_coo - 1) + 1) / 2
    ls_zero = -(tf.sign(h_out_coo) - 1) / 2
    rest = 1 - gt_one - ls_zero
    readout = rest*h_out_coo + gt_one*(h_out_coo - 1) + ls_zero*(1 + h_out_coo) ########################################## THIS WAS 1 + h_out...
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

    if M > 3: # then vel was predicted as well, concat
        readout = tf.concat([readout, h_out[...,3:]], axis=-1)
    return readout

def get_readout_mod(h_out):
    M = h_out.get_shape().as_list()[-1]

    if M <= 3:
        readout = tf.mod(h_out, 1.0)
    else:
        loc = tf.mod(h_out[...,:3], 1.0)
        vel = h_out[...,3:]
        readout = tf.concat([loc, vel], axis=-1)
    return readout

def periodic_boundary_dist(readout_full, x_truth):
    """ minimum distances between particles given periodic boundary conditions
    Normal squared distance would penalize for large difference between particles
    on opposite sides of cube
    """
    readout = readout_full[...,:3]
    x_truth_coo = x_truth[...,:3]
    d1 = tf.squared_difference(readout, x_truth_coo)
    d2 = tf.squared_difference(readout, (1 + x_truth_coo))
    d3 = tf.squared_difference((1 + readout),  x_truth_coo)
    dist = tf.minimum(tf.minimum(d1, d2), d3)
    return dist


def pbc_loss(readout, x_truth, vel=False):
    """ MSE over full dims with periodic boundary conditions
    Args:
        readout (tensor): model prediction which has been remapped to inner cube
        x_truth (tensor): ground truth (mb_size, N, 6)
        vel: if vel, then include vel error in loss
    """
    pbc_dist  = periodic_boundary_dist(readout, x_truth)
    error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1))

    if vel:
        assert readout.get_shape().as_list()[-1] > 3
        dist_vel = tf.squared_difference(readout[...,3:], x_truth[...,3:])
        error += tf.reduce_mean(tf.reduce_sum(dist_vel, axis=-1))
    return error
#------------------------------------------------------------------------------
# Rescaled MSE loss functions, for velocity predictions
#------------------------------------------------------------------------------
# ==== Helpers
def pbc_squared_diff(x, y):
    # Squared difference function with PeriodicBoundaryConditions (pbc)
    d1 = tf.squared_difference(x, y)
    d2 = tf.squared_difference(x, (1+y))
    d3 = tf.squared_difference((1+x), y)
    dist = tf.minimum(tf.minimum(d1, d2), d3)
    return dist

def mse(x, y, vel=False):
    # Mean Squared Error function
    # velocity not bounded, so no pbc dist if vel
    sqr_err = tf.squared_difference(x, y) if vel else pbc_squared_diff(x, y)
    sum_sqr_err = tf.reduce_sum(sqr_err, axis=-1) # eg (b, N, 3) -> (b, N)
    mean_sum_sqr_err = tf.reduce_mean(sum_sqr_err)
    return mean_sum_sqr_err

# ==== Loss func
def pbc_loss_scaled(x_input, x_pred, x_truth, vel=True):
    """ MSE loss (with pbc) rescaled by difference between x_input and truth
    Args:
        x_input (tensor): (b, N, 6), input data
        x_pred  (tensor): (b, N, 6), model prediction
        x_truth (tensor): (b, N, 6), true data
    """
    # Split data
    # ========================================
    # not a fan of interpreted split over explicit splitting idx
    split_div = 2
    loc_input, vel_input = tf.split(x_input, split_div, axis=-1)
    loc_truth, vel_truth = tf.split(x_truth, split_div, axis=-1)
    loc_pred = x_pred[...,:3]
    #loc_pred,  vel_pred  = tf.split(x_pred,  split_div, axis=-1)
    #loc_input, vel_input = x_input[...,:3], x_input[...,3:]
    #loc_truth, vel_truth = x_truth[...,:3], x_truth[...,3:]
    #loc_pred,  vel_pred  = x_pred[...,:3], x_pred[...,3:]

    # Scalars
    # ========================================
    loc_scalar = mse(loc_input, loc_truth)

    # Prediction error
    # ========================================
    loc_error = mse(loc_pred, loc_truth)

    # output
    error = (loc_error / loc_scalar)

    # Vel pred
    # ========================================
    if vel:
        vel_pred = x_pred[...,3:]
        vel_scalar = mse(vel_input, vel_truth, vel=True)
        vel_error = mse(vel_pred, vel_truth, vel=True)
        error += (vel_error / vel_scalar)
    return error


#------------------------------------------------------------------------------
# Numpy-based loss
#------------------------------------------------------------------------------
def npy_periodic_boundary_dist(readout_full, x_truth):
    readout = readout_full[...,:3]
    x_truth_coo = x_truth[...,:3]
    d1 = np.square(readout - x_truth_coo)
    d2 = np.square(readout - (1 + x_truth_coo))
    d3 = np.square((1 + readout) - x_truth_coo)
    dist = np.minimum(np.minimum(d1, d2), d3)
    return dist

def npy_pbc_loss(readout, x_truth, mu_axis=None):
    pbc_dist  = npy_periodic_boundary_dist(readout, x_truth)
    error = np.mean(np.sum(pbc_dist, axis=-1), axis=mu_axis)
    return error
