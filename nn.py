import os, code, sys, time

import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import coo_matrix
import tensorflow as tf

import utils
from utils import VAR_SCOPE, VAR_SCOPE_MULTI
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use


#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
# TF-related ops
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><
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


def shift_inv_layer(H_in, COO_idx, bN, layer_id, is_last=False):
    """
    Args:
        H_in (tensor): (c, k), stores shift-invariant edge features, row-major
          - c = b*N*M, if KNN then M is fixed, and k = num_edges = num_neighbors = M
        COO_idx (tensor): (3, c), of row, column, cube-wise indices respectively
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
    row_idx, col_idx, cube_idx = tf.split(COO_idx, 3, axis=0)

    # get layer weights
    W1, W2, W3, W4, B = utils.get_ShiftInv_layer_vars(layer_id)

    # Helper funcs
    # ========================================
    def _pool(H, idx, broadcast=True):
        # row : col
        # col : row
        # cubes : cubes
        return pool_graph(H, idx, b*N)

    def _left_mult(h, W):
        return tf.einsum("ck,kq->cq", h, W)

    # Layer forward pass
    # ========================================
    # H1 - no pooling
    H1 = _left_mult(H_in, W1) # (c, q)

    # H2 - pool rows
    H_rooled_rows = _pool(H_in, col_idx)
    H2 = _left_mult(H_pooled_rows, W2) # (c, q)

    # H3 - pool cols
    H_pooled_cols = _pool(H_in, row_idx)
    H3 = _left_mult(H_pooled_rows, W3) # (c, q)

    # H4 - pool cubes
    H_pooled_all = _pool(H_in, cube_idx)
    H4 =  _left_mult(H_pooled_all, W4) # (c, q)

    # Output
    # ========================================
    H_out = (H1 + H2 + H3 + H4) + B
    if is_last:
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        H_out = tf.reshape(_pool(H_out, col_idx, broadcast=False), (b, N, -1))
    return H_out


def include_node_features(X_in_edges, X_in_nodes, COO_idx):
    """ Broadcast node features to edges for input layer
    Args:
        X_in_edges (tensor):   (c, 3), input edge features (relative pos of neighbors)
        X_in_nodes (tensor): (b*N, 3), input node features (velocities)
        COO_idx (tensor): (3, c), rows, cols, cube indices (respectively)
    Returns:
        X_in_graph (tensor): (c, 9), input with node features broadcasted to edges
    """
    # get row, col indices
    row_idx = COO_idx[0]
    col_idx = COO_idx[1]

    # get node row, columns
    node_rows = tf.gather_nd(X_in_nodes, tf.expand_dims(row_idx, axis=1))
    node_cols = tf.gather_nd(X_in_nodes, tf.expand_dims(col_idx, axis=1))

    # full node, edges graph
    X_in_graph = tf.concat([X_in_edges, node_rows, node_cols], axis=1) # (c, 9)

    return X_in_graph


#------------------------------------------------------------------------------
# Shift invariant network func
#------------------------------------------------------------------------------

#=============================================================================
# new perm eqv, shift inv model funcs
#=============================================================================
# ==== Network fn
def sinv_network_fwd(num_layers, var_scope, X, V, rows, cols, all_idx, N, b, activation=tf.nn.relu):
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    #H = activation(input_shift_inv_layer(X, V, rows, cols, 0, var_scope))

    # input layer
    h_in = include_node_features(X, V, rows, cols)
    H = activation(shift_inv_layer(h_in, rows, cols, all_idx, N, b, 0, var_scope))

    # hidden layers
    for i in range(1, num_layers):
        is_last = i == num_layers - 1
        H = shift_inv_layer(H, rows, cols, all_idx, N, b, i, var_scope, is_last=is_last)
        if not is_last:
            H = activation(H)
    return H


# ==== Model fn
def sinv_model_fwd(num_layers, X, V, rows, cols, all_idx, N, b, activation=tf.nn.relu, vel_coeff=False, var_scope=VAR_SCOPE):
    h_out = sinv_network_fwd(num_layers, var_scope, X, V, rows, cols, all_idx, N, b)
    if vel_coeff:
        theta = utils.get_vel_coeff(var_scope)
        h_out = theta * h_out
    return h_out

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

def get_input_edge_features_batch_rad(X_in, lst_csrs, M, offset_idx=False):
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

def _get_input_edge_features_batch_offset(X_in, lst_csrs, M):
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
        e = np.zeros_like(r) + i
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

        rows = np.concatenate([rows, r], axis=0)
        cols = np.concatenate([cols, c], axis=0)
        idx  = np.concatenate([idx,  e], axis=0)
    return rows.astype(np.int32), cols.astype(np.int32), idx.astype(np.int32)


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
        error *= tf.reduce_mean(tf.reduce_sum(dist_vel, axis=-1))
    return error


##### Numpy based loss

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
