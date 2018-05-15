import os, code, sys, time

import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import coo_matrix
import tensorflow as tf

import utils
from utils import VAR_SCOPE, VAR_SCOPE_MULTI
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

RADIUS = 0.08

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# TF-related ops
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><

#=============================================================================
# LAYER OPS, New perm-eqv, shift-inv model
#=============================================================================
def _pool(X, idx, num_segs, broadcast):
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
        tensor of shape (c, k) if broadcast is True
        tensor of shape (b*N, k) if broadcast is False
    """
    X_pooled = tf.unsorted_segment_mean(X, idx, num_segs)

    if broadcast:
        return tf.gather_nd(X_pooled, tf.expand_dims(idx, axis=1))

    else:
        return X_pooled

'''
For every layer in this model, there are 4 weights and 1 bias
        W1: (k, q) no-pooling
        W2: (k, q) pooling rows
        W3: (k, q) pooling cols
        W4: (k, q) pooling all
        B: (q,) bias
'''

def left_mult_sinv(X, W):
    return tf.einsum("bpk,kq->bpq", X, W)

'''
def shift_inv_layer(X, rows, cols, layer_idx, var_scope, is_last=False, N=32**3):
    """
    X: (b, N, M, k)
    L: (b*N*M, 2) # adjacency list, tiled for tf.gather_nd
    layer_idx/var_scope: both for tf.get_variable
    W*: Weights of shape (k, q)
    B: bias of shape (q,)

    Returns:
        tensor of shape (b, N, M, q)
    """
    # helpers
    def _pool_cols(X, broadcast=True):
        return _pool(X, idx=rows, N=N, broadcast=broadcast)

    def _pool_rows(X, broadcast=True):
        return _pool(X, idx=cols, N=N, broadcast=broadcast)

    # get layer weights
    W1, W2, W3, W4, B = utils.get_sinv_layer_vars(layer_idx, var_scope)

    # get dims
    dims = tf.shape(X) # (b, N, M, k)
    #N = dims[1]
    #M = dims[2]
    #k = dims[3]

    # pooling constants
    #ones_m = tf.ones([M, 1], tf.float32)
    #ones_n = tf.ones([N, 1], tf.float32)

    # Pooling and weights
    # ========================================
    # W1 - no pooling
    H1 = left_mult_sinv(X, W1)

    # W2 - pool rows - this is the trickiest
    X_pooled_rows = _pool_rows(X)
    H2 = left_mult_sinv(X_pooled_rows, W2)

    # W3 - pool columns
    X_pooled_cols = _pool_cols(X)
    H3 = left_mult_sinv(X_pooled_cols, W3)

    # W4 - pool all
    X_pooled_all = _pool_all(X)
    H4 = left_mult_sinv(X_pooled_rows_cols, W4)

    # output
    H = H1 + H2 + H3 + H4
    X_out = H + B

    if is_last:
        return _pool_cols(X_out, broadcast=False)
    else:
        return X_out
'''

def shift_inv_layer(X_in, row_idx, col_idx, all_idx, N, b, layer_idx, var_scope, is_last=False):
    """
    Args:
        X_in (tensor): has shape (c, k), stores shift-invariant edge features, row-major order.
            c = sum_{i=0..b} n_i, and n_i is the number of non-zero entries of the i-th adjacency in the batch.
                - if all matrices in the batch have the same number n of non zero-entries, c = b*n
                - if the number of neighbors is fixed to M, then n = N*M and c = b*N*M

        row_ids, col_ids, all_idx (numpy array): have shape (c), store row / column indices of adjacency non-zero entries,
            row_ids, col_ids, all_idx =  pre_process_adjacency(A), A=adjacency batch.
        N (int): number of particles
        b (int): batch size
        W* (tensor): weights with shape (k, q), q = number of output channels
        P (tensor): bias with shape q
        activation: defaults is tf.nn.relu
        is_last (bool): if True pools output over columns (neighbors), default is False

    Returns:
        tensor of shape (c, q) if is_last is False
        tensor of shape (b, N, q) if is_last is True
    """
    def _pool_cols(X, broadcast=True):
        return _pool(X, row_idx, N * b, broadcast)

    def _pool_rows(X, broadcast=True):
        return _pool(X, col_idx, N * b, broadcast)

    def _pool_all(X, broadcast=True):
        return _pool(X, all_idx, N * b, broadcast)

    # get layer weights
    W1, W2, W3, W4, P = utils.get_sinv_layer_vars(layer_idx, var_scope)

    # Pooling and weights
    # ========================================
    # W1 - no pooling
    X1 = tf.einsum("ij,jw->iw", X_in, W1)  # (c, q)

    # W2 - pool rows
    X_pooled_r = _pool_rows(X=X_in)
    X2 = tf.einsum("ij,jw->iw", X_pooled_r, W2)  # (c, q)

    # W3 - pool cols
    X_pooled_c = _pool_cols(X=X_in)
    X3 = tf.einsum("ij,jw->iw", X_pooled_c, W3)  # (c, q)

    # W4 - pool all
    X_pooled_all = _pool_all(X=X_in)
    X4 = tf.einsum("ij,jw->iw", X_pooled_all, W4)  # (c, q)

    X_all = tf.add_n([X1, X2, X3, X4])
    X_bias = tf.add(X_all, tf.reshape(P, [1, -1]))
    X_out = X_bias  # (c, q)

    # Output
    # ========================================
    if is_last:
        return tf.reshape(_pool_cols(X_out, broadcast=False), [b, N, -1])

    else:
        return X_out


def include_node_features(X_in, V, row_idx, col_idx):
    """
    Broadcast node features to edges. To be used for first layer input.

    Args:
        X_in (tensor): has shape (c, 3) - input edge features (relative positions of neighbors)
        V (tensor): has shape (b*N, 3) - input node features (velocities)

    Returns:
        tensor with shape (c, 9), with node features broadcasted to edges
    """
    R = tf.gather_nd(V, tf.expand_dims(row_idx, axis=1))
    C = tf.gather_nd(V, tf.expand_dims(col_idx, axis=1))

    return tf.concat([X_in, R, C], axis=1)  # (c, 9)


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
        h_out = skip_connection(x_in, h_out, vcoeff)
    return h_out

#=============================================================================
# new perm eqv, shift inv model funcs
#=============================================================================
# ==== Network fn
def sinv_network_fwd(num_layers, var_scope, X, V, rows, cols, all_idx, N, b, activation=tf.nn.relu):
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    #H = activation(input_shift_inv_layer(X, V, rows, cols, 0, var_scope))
    h_in = include_node_features(X, V, rows, cols)
    H = activation(shift_inv_layer(h_in, rows, cols, all_idx, N, b, 0, var_scope))
    for i in range(1, num_layers):
        is_last = i == num_layers - 1
        H = shift_inv_layer(H, rows, cols, all_idx, N, b, i, var_scope, is_last=is_last)
        if not is_last:
            H = activation(H)
    return H

# ==== Model fn
def sinv_model_fwd(num_layers, X, V, rows, cols, all_idx, N, b, activation=tf.nn.relu, vel_coeff=False, var_scope=VAR_SCOPE):
    h_out = sinv_network_fwd(num_layers, var_scope, X, V, rows, cols, all_idx, N, b)
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

def get_input_edge_features_batch(X_in, lst_csrs, M, offset_idx=False):
    """ get relative distances of each particle from its M neighbors
    Args:
        X_in (ndarray): (b, N, 6), input data
        lst_csrs (list(csr)): len b list of csrs
    """
    x = X_in[...,:3] # (b, N, 3)
    b, N, k = x.shape

    x_out = np.zeros((b, N*M, k)).astype(np.float32)
    # have to loop, since mem issues with full X_in[adj_list]
    for i in range(b):
        h = x[i]     # (N, 3)
        a = lst_csrs[i].indices.reshape(N, M) # (N, M)
        h_sel = h[a] # (N, M, 3)
        h_out = h_sel - h[:,None,:] # (N, M, 3) - (N, 1, 3)
        x_out[i] = h_out.reshape(-1, k)
    return x_out.reshape(-1, k)

def get_input_edge_features_batch_offset(X_in, lst_csrs, M):
    """ get relative distances of each particle from its M neighbors
    Args:
        X_in (ndarray): (b, N, 6), input data
        lst_csrs (list(csr)): len b list of csrs
    """
    x = X_in[...,:3] # (b, N, 3)
    b, N, k = x.shape
    x = x.reshape(-1, k)
    adj = lst_csrs[0].indices.reshape(N, M)
    for i in range(1, b):
        cur = lst_csrs[i].indices.reshape(N, M)
        adj = np.concatenate([adj, cur], axis=0)
    x_out = x[adj] - x[:,None,:] # (B*N, M, k)
    return np.reshape(x_out, (b, N*M, k))


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
        kgraph = kneighbors_graph(X_in[i,:,:3], M, include_self=inc_self)
        if offset_idx:
            kgraph.indices = kgraph.indices + (N * i)
        lst_csrs.append(kgraph)
    return lst_csrs

def to_coo_batch2(A):
    """ Get row and column indices from csr
    DOES NOT LIKE OFFSET IDX, tocoo() method will complain about index being
    greater than matrix size

    Args:
        A (csr): list of csrs of shape (N, N)
    """
    b = len(A) # batch size
    N = A[0].shape[0] # (32**3)

    # initial coo mat
    a = A[0].tocoo()
    rows = a.row # (N*M)
    cols = a.col # (N*M)
    idx = np.zeros_like(rows)
    for i in range(1, b):
        # concat each to batched rows, cols
        a = A[i].tocoo()
        r = a.row + (i*N)
        c = a.col + (i*N)
        e = np.zeroes_like(r) + i

        rows = np.concatenate([rows, r], axis=0)
        cols = np.concatenate([cols, c], axis=0)
        idx  = np.concatenate([idx,  e], axis=0)
    return rows.astype(np.int32), cols.astype(np.int32), idx.astype(np.int32)

def to_coo_batch(A):
    """ Get row and column indices from csr
    DOES NOT LIKE OFFSET IDX, tocoo() method will complain about index being
    greater than matrix size

    Args:
        A (csr): list of csrs of shape (N, N)
    """
    b = len(A) # batch size
    N = A[0].shape[0] # (32**3)

    # initial coo mat
    a = A[0].tocoo()
    rows = a.row[None,...] # (1, N*M)
    cols = a.col[None,...] # (1, N*M)
    for i in range(1, b):
        # concat each to batched rows, cols
        a = A[i].tocoo()
        r = a.row[None,...]
        c = a.col[None,...]

        rows = np.concatenate([rows, r], axis=0)
        cols = np.concatenate([cols, c], axis=0)
    return rows.astype(np.int32), cols.astype(np.int32)

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

def get_radNeighbor_coo(X_in, R=RADIUS):
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

def get_radNeighbor_coo_batch(X_in, R=RADIUS):
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

def get_radius_graph_input(X_in, R=RADIUS):
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
    readout = rest*h_out_coo + gt_one*(h_out_coo - 1) + ls_zero*(1 + h_out_coo)

    if M > 3: # then vel was predicted as well, concat
        readout = tf.concat([readout, h_out[...,3:]], axis=-1)
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
        error *=   tf.reduce_mean(tf.reduce_sum(dist_vel, axis=-1))
    return error
