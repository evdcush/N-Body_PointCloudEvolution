import os, code, sys, time

import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import coo_matrix
import tensorflow as tf

import utils
from utils import VARIABLE_SCOPE as VAR_SCOPE
#from utils import VARIABLE_SCOPE, SEGNAMES_3D
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use


'''
TODO:
 - think of bette way to standardize model and network func arguments, so
   that layers and networks can be swapped and interfaced more simply
   - Solutions:
     - wrapper funcs
     - higher level interface funcs
     - a model func class? or model func args class, or dict
 - maybe make a trainer?
 - segment_mean stuff no longer needs explicit shape!

 - TEST:
   - any difference between doing the edges/nodes preprocessing in TF sess
     or before feeding (does it affect backprop)
   - Does wrapping functions in a class affect performance (backprop)
     - most of the tf code I've seen is purely functional
'''
#==============================================================================
# Globals
#==============================================================================
SUBSCRIPT_VANILLA  = 'ijk,kq->ijq'
SUBSCRIPT_SHIFTINV = 'ck,kq->cq'
SUBSCRIPT_ROTINV   = 'bek,kq->beq'

#==============================================================================
# Data processing funcs
#==============================================================================
#------------------------------------------------------------------------------
# Shift invariant ops
#------------------------------------------------------------------------------
# Shift invariant nodes
# ========================================
def include_node_features(X_in_edges, X_in_nodes, COO_feats, redshift=None):
    """ Broadcast node features to edges for input layer
    Args:
        X_in_edges (tensor):   (c, 3), input edge features (relative pos of neighbors)
        X_in_nodes (tensor): (b*N, 3), input node features (velocities)
        COO_feats (tensor): (3, c), rows, cols, cube indices (respectively)
    Returns:
        X_in_graph (tensor): (c, 9), input with node features broadcasted to edges
    """
    # ==== get row, col indices
    row_idx = COO_feats[0]
    col_idx = COO_feats[1]

    # ==== get node row, columns
    node_rows = tf.gather_nd(X_in_nodes, tf.expand_dims(row_idx, axis=1))
    node_cols = tf.gather_nd(X_in_nodes, tf.expand_dims(col_idx, axis=1))

    # ==== full node, edges graph
    X_in_graph = tf.concat([X_in_edges, node_rows, node_cols], axis=1) # (c, 9)

    # ==== broadcast redshifts
    if redshift is not None:
        X_in_graph = tf.concat([X_in_graph, redshift], axis=1) # (c, 10)
    return X_in_graph


# Shift invariant edges
# ========================================
def get_input_features_ShiftInv(X_in, coo, dims):
    """ get edges and nodes with TF ops
    TESTED EQUIVALENT TO numpy-BASED FUNC

    get relative distances of each particle from its M neighbors
    Args:
        X_in (tensor): (b, N, 6), input data
        coo (tensor): (3,b*N*M)
    """
    # ==== split input
    b, N, M = dims
    X = tf.reshape(X_in, (-1, 6))
    edges = X[...,:3] # loc
    nodes = X[...,3:] # vel

    # ==== get edges (neighbors)
    cols = coo[1]
    edges = tf.reshape(tf.gather(edges, cols), [b, N, M, 3])
    # weight edges
    edges = edges - tf.expand_dims(X_in[...,:3], axis=2) # (b, N, M, 3) - (b, N, 1, 3)
    return tf.reshape(edges, [-1, 3]), nodes



#==============================================================================
# NN functions
#==============================================================================
#------------------------------------------------------------------------------
# Base nn functions
#------------------------------------------------------------------------------
# Weight transformations
# ========================================
def left_mult(h, W, subscript=None):
    """ batch matmul for set-based data
    """
    if subscript is None:
        ndims = len(h.shape)
        lhs = 'ij'[:ndims-1] # h at most 3D in our usage
        subscript = '{0}k,kq->{0}q'.format(lhs)
    return tf.einsum(subscript, h, W)

#------------------------------------------------------------------------------
# Graph convolutions
#------------------------------------------------------------------------------
# KNN graph conv
# ========================================
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
    conv = tf.reduce_mean(tf.reshape(tf.gather_nd(h, adj), rdim), axis=2)
    return conv


# Radius graph conv
# ========================================
def rad_graph_conv(h, spT, *args):
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


# Pooled graph conv
# ========================================
def pool_ShiftInv_graph_conv(h, pool_idx, num_segs, broadcast):
    """
    Args:
        h (tensor): has shape (c, k), row-major order
        pool_idx (numpy array): has shape (c),
            must be row idx of non-zero entries to pool over columns
            must be column idx of non-zero entries to pool over rows
        num_segs (int): number of segments in h (how many particles)
        broadcast (bool): if True, after pooling re-broadcast to original shape

    Returns:
        tensor of shape (c, k) if broadcast, else (b*N, k)
    """
    pooled_conv = tf.unsorted_segment_mean(h, pool_idx, num_segs)
    if broadcast:
        pooled_conv = tf.gather_nd(pooled_conv, tf.expand_dims(pool_idx, axis=1))
    return pooled_conv





#==============================================================================
# Layer ops
#==============================================================================
#------------------------------------------------------------------------------
# Vanilla
#------------------------------------------------------------------------------
# Set layer (no longer used)
# ========================================
def set_layer(h, layer_idx, *args):
    """ Set layer
    *args just for convenience, set_layer has no additional
    Args:
        h: data tensor, (mb_size, N, k_in)
        layer_idx (int): layer index for params
        var_scope (str): variable scope for get variables from graph
    RETURNS: (mb_size, N, k_out)
    """
    W, B = utils.get_vanilla_layer_vars(layer_idx)
    mu = tf.reduce_mean(h, axis=1, keepdims=True)
    h_out = left_mult(h - mu, W, subscript=SUBSCRIPT_VANILLA) + B
    return h_out

# KNN graph layer (vanilla, not in use)
# ========================================
def kgraph_layer(h, layer_idx, G, K):
    """ Graph layer for KNN

    Args:
        h: data tensor, (mb_size, N, k_in)
        layer_idx (int): layer index for params
        G: adjacency index list tensor (*, 2), of tf.int32
        K (int): number of nearest neighbors in KNN
    RETURNS: (mb_size, N, k_out)
    """
    W, B = utils.get_vanilla_layer_vars(layer_idx)
    edge_mean = kgraph_conv(h, G, K)
    h_out = left_mult(h - edge_mean, W, SUBSCRIPT_VANILLA) + B
    return h_out

# Radius graph layer
# ========================================
def rad_graph_layer(h, layer_idx, spT, *args):
    """ Radius graph layer
    NB: considerably more memory consumption than any other layer type
    Args:
        h (tensor): activation
        layer_idx (int): layer index for retrieving parameters
        spT (SparseTensor): the radius graph, in pseudo-COO-matrix
                            form as a tf.SparseTensor
    Returns: (mb_size, N, k_out) tensor
    """
    W, B = utils.get_vanilla_layer_vars(layer_idx)
    cluster_mean = rad_graph_conv(h, spT)
    h_out = left_mult(h - cluster_mean, W, SUBSCRIPT_VANILLA) + B
    return h_out


#------------------------------------------------------------------------------
# Shift Invariant layers
#------------------------------------------------------------------------------
def ShiftInv_layer(H_in, COO_feats, bN, layer_id, is_last=False):
    """ Shift-invariant network layer
    # pooling relations
    # row : col
    # col : row
    # cubes : cubes
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
    row_idx  = COO_feats[0]
    col_idx  = COO_feats[1]
    cube_idx = COO_feats[2]

    # get layer weights
    weights, B = utils.get_ShiftInv_layer_vars(layer_id)
    W1, W2, W3, W4 = weights

    # Helper funcs
    # ========================================
    def _pool(H, idx, broadcast=True):

        return pool_ShiftInv_graph_conv(H, idx, b*N, broadcast)

    def _left_mult(h, W):
        return left_mult(h, W, SUBSCRIPT_SHIFTINV)

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
        H_out = tf.reshape(_pool(H_out, row_idx, broadcast=False), (b, N, -1))
    return H_out




#==============================================================================
# Network ops
#==============================================================================
#------------------------------------------------------------------------------
# Network Functions # TODO: at somepoint make a network func interface or class
#------------------------------------------------------------------------------
# Shift invariant network
# ========================================
def network_func_ShiftInv(X_in_edges, X_in_nodes, COO_feats,
                          num_layers, dims, activation, redshift=None):
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



#==============================================================================
# Model ops
#==============================================================================
#------------------------------------------------------------------------------
# Shift invariant model funcs
#------------------------------------------------------------------------------
def model_func_ShiftInv(X_in, COO_feats, model_specs, redshift=None):
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
    activation = model_specs.activation # default tf.nn.relu
    dims = model_specs.dims

    # Get graph inputs
    # ========================================
    edges, nodes = get_input_features_ShiftInv(X_in, COO_feats, dims)

    # Network forward
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        # ==== Split input
        X_in_loc, X_in_vel = X_in[...,:3], X_in[...,3:]
        # ==== Network output
        net_out = network_func_ShiftInv(edges, nodes, COO_feats, num_layers,
                                        dims[:-1], activation, redshift)
        #num_feats = net_out.get_shape().as_list()[-1]

        # ==== Scale network output and compute skip connections
        loc_scalar, vel_scalar = utils.get_scalars()
        H_out = net_out[...,:3]*loc_scalar + X_in_loc + X_in_vel*vel_scalar

        # ==== Concat velocity predictions
        if net_out.get_shape().as_list()[-1] > 3:
            H_vel = net_out[...,3:]*vel_scalar + X_in_vel
            H_out = tf.concat([H_out, H_vel], axis=-1)
        return get_readout(H_out)


#=============================================================================
# Graph, adjacency functions
#=============================================================================
#------------------------------------------------------------------------------
# Adjacency utils
#------------------------------------------------------------------------------
# Kgraph
# ========================================
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


# Sparse matrix conversions
# ========================================
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

def confirm_CSR_to_COO_index_integrity(A, COO_feats):
    """ CSR.indices compared against COO.cols
    Sanity check to ensure that my indexing algebra is correct
    """
    CSR_feats = get_indices_from_list_CSR(A)
    cols = COO_feats[1]
    assert np.all(CSR_feats == cols)


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

#------------------------------------------------------------------------------
# Graph func wrappers
#------------------------------------------------------------------------------
# Graph gets
# ========================================
def get_kneighbor_list(X_in, M, offset_idx=False, inc_self=False):
    b, N, D = X_in.shape
    lst_csrs = []
    for i in range(b):
        kgraph = kneighbors_graph(X_in[i,:,:3], M, include_self=inc_self).astype(np.float32)
        if offset_idx:
            kgraph.indices = kgraph.indices + (N * i)
        lst_csrs.append(kgraph)
    return lst_csrs


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


def get_graph_csr_list(h_in, args):
    M = args.graph_var
    pbc = args.pbc_graph == 1
    if pbc:
        return get_pbc_kneighbors_csr(h_in, M, 0.03)
    else:
        return get_kneighbor_list(h_in, M)


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


def pbc_loss(x_pred, x_truth, *args, **kwargs):
    """ MSE over full dims with periodic boundary conditions
    Args:
        readout (tensor): model prediction which has been remapped to inner cube
        x_truth (tensor): ground truth (mb_size, N, 6)
        vel: if vel, then include vel error in loss
    """
    pbc_dist  = periodic_boundary_dist(x_pred, x_truth)
    error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1))
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
def pbc_loss_scaled(x_pred, x_truth, x_input, vel=True):
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


def get_loss_func(args):
    mstep = args.model_type == utils.MULTI_STEP
    loss_func = pbc_loss_scaled if mstep else pbc_loss
    return loss_func


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



#==============================================================================
# Layer class
#==============================================================================
# SOMEDAY! No time right now
'''
# functions < Layer < Network < Model
class GraphConv:
    """ All layers use some kind of graph convolution
    - KNN: simplest, just mean(X[G], M_axis)
    - Radius: trickier
      - requires SparseTensor
      - makes assumptions about the SparseTensor to allow for
        tf.sparse_tensor_dense_matmul
    # SPECIAL CASES
    - ShiftInv: uses KNN graph currently, but performs convolution
      using unsorted_segment_mean instead of vanilla mean
        - Ideally, should be able to swap out ShiftInv/pooling convs
    - RotInv: probably same shit
    """
    def __init__(self, X_in, G, M):
        self.X = X_in
        self.G = G
        self.M = M
        self.graph_conv

    def graph_conv(self): # MUST BE OVERRIDDEN
        pass

class KGraphConv(GraphConv):
    """ KNN graph convolution
    The simplest type of graph convolution
    Just take the mean over each node's edges
    """
    def __init__(self, X_in, G, M):
        super(KGraphConv, self).__init__(X_in, G, M)

    def graph_conv(self):
        X = self.X; G = self.G; M = self.M;
        dims = tf.shape(X)
        mb = dims[0]; n  = dims[1]; d  = dims[2];
        rdim = [mb,n,M,d]
        conv = tf.reduce_mean(tf.reshape(tf.gather_nd(X, G), rdim), axis=2)
        return conv



class Layer:
    """ Layers wrap functions (layers are functions too...)
    What do layers need?
      - Variables (weights, biases)
      - a layer index, to retrieve said Variables
        - Here we can probably use tf.get_variable within scope
      - a variable scope, for all the above
      - Data:
        - input data: typically activations and an adjacency graph
    So:
      - Placeholders?
    """
    def __init__(self, h_input, adjacency):
        self.H = h_input
        self.A = adjacency

class ShiftInvariantLayer(Layer):
    def __init__(self,):

class Network:
    """ Network wraps all layer ops (activations)
    ASSUME PLACEHOLDERS, can still feed as usual
    """
    def __init__(self, ):
'''

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#                       ## Rotation Invariant ##
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


#------------------------------------------------------------------------------
# PREPROCESSING
#------------------------------------------------------------------------------
def get_batch_2D_segmentID(batch_graph):
    """
    Return row, col, indices of a list of 2D sparse adjacencies with batch indices too.
        Sorry, using a different indexing system from 3D adjacency case. TODO: make indexing consistent for clarity.

    Args:
        batch_graph. List of csr (or any other sparse format) adjacencies.

    Returns:
        array of shape (2, b * N * (M-1), 2). Each pair in the third axis is a batch idx - row idx or
            batch idx - col idx for non-zero entries of 2D adjacency. 0-axis is rows/cols respectively.
    """
    b = len(batch_graph)
    rows = []
    cols = []

    for i in range(b):
        adj = batch_graph[i]
        adj.setdiag(0)
        r, c = adj.nonzero()
        batch_idx = np.zeros_like(r) + i
        rows.append(np.transpose([batch_idx, r]))
        cols.append(np.transpose([batch_idx, c]))

    rows = np.reshape(np.array(rows), (-1, 2))
    cols = np.reshape(np.array(cols), (-1, 2))

    return np.array([rows, cols])


def get_3D_segmentID(adj_graph, M):
    """ Build 3D adjacency from csr_matrix (though any matrix from scipy.sparse works)
    Args:
        adj_graph (csr_matrix): neighbor graph (assumes kneighbors_graph)
        M (int): number of neighbors
    Returns:
        row, col, depth. npy arrays for indices of non-zero entries. Diagonals removed.
    """
    adj_graph.setdiag(0)  # Don't need diagonal elements
    m_eff = M - 1 # Wouldn't need this if include_self=False, but keep for now

    # get COO features
    # NB: this is not the same as COO conversion, since nonzero() will return neighbor indices sorted
    rows, cols = adj_graph.nonzero()
    num_elements = rows.size # N*(M-1)

    # 3D-projected features for row, col, depth
    r = np.repeat(rows, m_eff-1) # tested equivalent
    c = np.repeat(cols, m_eff-1) # tested equivalent

    # depth indexing algebra is more complicated, # tested equivalent
    d = np.reshape(np.repeat(np.reshape(cols, [-1, m_eff]), m_eff, axis=0), -1)
    del_idx = np.array([(i%m_eff) + (i*m_eff) for i in range(num_elements)])
    d = np.delete(d, del_idx)

    return r, c, d


def get_segmentID(lst_csrs, M):
    """ preprocess batch of adjacency matrices for segment ids
    Total of 7 segments for 3D graph, in order:
    col-depth, row-depth, row-col, depth, col, row, all

    Args:
        lst_csrs (list(csr_matrix)): list of csr_matrix for neighbor graph, of len num_batches
        M (int): number of neighbors
    Returns:
        ndarray (b, 7, e)
          where e=N*(M-1)*(M-2), num of edges in 3D adjacency matrix (no diags)
          N: num_particles
    """
    # Helper funcs
    # ========================================
    def _combine_segment_idx(idx1, idx2):
        combined_idx = np.transpose(np.array([idx1, idx2])) # pair idx
        vals, idx = np.unique(combined_idx, axis=0, return_inverse=True) # why not return_index?
        return idx

    # process each adjacency in batch
    # ========================================
    seg_idx = []
    for adj in batch:
        # get row, col, depth segment idx (pools col-depth, row-depth, row-col, respectively)
        r, c, d = get_3D_segmentID(adj, M)

        # combine segment idx (pools rc->d, rd->c, cd->r, respectively)
        rc = _combine_segment_idx(r, c)
        rd = _combine_segment_idx(r, d)
        cd = _combine_segment_idx(c, d)

        # idx for pooling over all
        a = np.zeros_like(r)

        # order seg ids
        seg_idx.append(np.array([cd, rd, rc, d, c, r, a])) # ['CD', 'RD', 'RC', 'D', 'C', 'R', 'A']
    seg_idx = np.array(seg_idx)

    # Offset indices
    # ========================================
    for i in range(1, seg_idx.shape[0]): # batch_size
        for j in range(seg_idx.shape[1]): # 7
            seg_idx[i][j] += np.max(seg_idx[i-1][j]) + 1
    return seg_idx



# Pre-process input
# ========================================
def get_RotInv_input(X, V, lst_csrs, M):
    """
    Args:
         X. Shape (b, N, 3). Coordinates.
         V. Shape (b, N, 3), Velocties.
         batch_A. List of csr adjacencies.
         m (int). Number of neighbors.

    Returns:
        numpy array of shape (b, e, 10)
            e=N*(M-1)*(M-2), number of edges in 3D adjacency (diagonals removed), N=num of particles, M=num of neighbors
            10 input channels corresponding to 1 edge feature + 9 broadcasted surface features, those are broken
            down into 3 surfaces x (1 scalar distance + 1 row velocity projected onto cols + 1 col velocity
            projected onto rows)
    """
    # Helpers
    _norm    = lambda dist: np.linalg.norm(dist)
    _angle   = lambda dist1, dist2: np.dot(dist1, dist2) / (_norm(dist1) * _norm(dist2))
    _project = lambda v, dist: np.dot(v, dist) / _norm(dist)

    # out dims
    batch_size, N = X.shape[:2]
    e = N*(M-1)*(M-2)
    #X_out = np.zeros((batch_size, e, 10)).astype(np.float32)
    X_out = []

    # Iterate over each cube in batch
    # ========================================
    for i in range(batch_size):
        x = X[i] # (N, 3)
        v = V[i] # (N, 3)
        adj = lst_csrs[i]

        # Get 3D seg ID (coo features)
        rows, cols, depth = get_3D_segmentID(adj, M)

        # Relative dist vectors
        dist_cr = x[cols]  - x[rows]
        dist_dr = x[depth] - x[rows]
        dist_dc = x[depth] - x[cols]

        # Velocities
        v_rows  = V[rows]
        v_cols  = V[cols]
        v_depth = V[depth]

        #   Input features
        #-------------------
        #===== Edge
        features = [_angle(dist_cr, dist_dr)]

        #===== RC surface
        # scalar distance + projection of row vel to rc vectors + projection of col vel to cr vectors
        features.extend([_norm(dist_cr), _project(v_rows, dist_cr), _project(v_cols, -dist_cr)])

        #===== RD surface
        # scalar distance + projection of row vel to rd vectors + projection of depth vel to dr vectors
        features.extend([_norm(dist_dr), _project(v_rows, dist_dr), _project(v_depth, -dist_dr)])

        #===== CD surface
        # scalar distance + projection of col vel to cd vectors + projection of depth vel to dc vectors
        features.extend([_norm(dist_dc), _project(v_cols, dist_dc), _project(v_depth, -dist_dc)])

        X_out.append(features)

    return np.array(X_out)


#------------------------------------------------------------------------------
# POST-PROCESSING OUTPUT
#------------------------------------------------------------------------------

# Post-process output
# ========================================
def get_final_position(X_in, segment_idx_2D, H_out, M):
    """ Calculates the final position of particles, where

    final_position : displacement_vectors + initial_position
        displacement_vectors : linear_combination(relative_pos_neighbors, H_out)

    Calculate displacement vectors = linear combination of neighbor relative positions, with weights = last layer
    outputs (pooled over depth), and add diplacements to initial position to get final position.

    Args:
        X_in. Shape (b, N, 3). Initial positions.
        segment_idx_2D . Shape (2, b * N * (M-1), 2). Each pair in the third axis is a batch idx - row idx or
            batch idx - col idx for non-zero entries of 2D adjacency.
            0-axis is rows/cols respectively. Get it from get_segment_idx_2D()
        H_out. Shape (b, N, M - 1, 1). Outputs from last layer (pooled over depth dimension).
        N (int). Number of neighbors.
    dX_reshaped = tf.reshape(dX, [tf.shape(X_in)[0], tf.shape(X_in)[1], M - 1, tf.shape(X_in)[2]])  # (b, N, M - 1, 3)
    Returns:
        Tensor of shape (b, N, 3). Final positions.
    """

    # Relative position of neighbors (neighbor - node)
    # ========================================
    dX = tf.gather_nd(X_in, segment_idx_2D[1]) - tf.gather_nd(X_in, segment_idx_2D[0])


    # Normalize relative positions (dX)
    # ========================================
    # Note: we want to normalize the dX vectors to be of length one
    #  ie, for any i,j,k  : dX[i,j,k,0]^2 + dX[i,j,k,1]^2 + dX[i,j,k,2]^2 = 1
    dX_reshaped = tf.reshape(dX, [tf.shape(X_in)[0], tf.shape(X_in)[1], M - 1, tf.shape(X_in)[2]])  # (b, N, M - 1, 3)
    dX_norm = tf.reshape(tf.linalg.norm(dX_reshaped[-1,3], axis=1), tf.shape(dX_shaped)[:-1] + (1,))
    dX_out = dX_reshape / dX_norm

    # Final position of particles
    # ========================================
    # Return initial pos, plus scaled and weighted (by H_out) displacement
    scalar = utils.get_scalars(1)[0] # single scalar
    displacement = tf.reduce_sum(tf.multiply(dX_out, H_out), axis=2)
    return X_in + scalar * displacement



#------------------------------------------------------------------------------
# ROTINV layer ops
#------------------------------------------------------------------------------
# Pooled graph conv
# ========================================
def pool_RotInv_graph_conv(X, idx, num_segs, broadcast=True):
    """
    Pooling for rotation invariant model, assumes everything in row-major order
    Args:
        X (tensor): (b,e,k)
        idx (tensor): (b,e)
        num_segs (int): b * N
        broadcast (bool): if True, after pooling re-broadcast to original shape
    Returns:
        tensor of shape (b,e,k) if broadcast, else (b,N*(M-1),k)
    """
    num_segs = tf.reduce_max(idx) + 1 # number of segments
    X_pooled = tf.unsorted_segment_mean(X, idx, num_segs)

    if broadcast: # same shape as X
        X_pooled = tf.gather_nd(X_pooled, tf.expand_dims(idx, axis=2))
    else:
        X_pooled = tf.reshape(X_pooled, [tf.shape(X)[0], -1, tf.shape(X)[2]])
    return X_pooled


#------------------------------------------------------------------------------
# Rotation Invariant layers
#------------------------------------------------------------------------------
def RotInv_layer(H_in, segID_3D, bN, layer_id, is_last=False):
    """
    Args:
        H_in (tensor): (b, e, k)
            b = minibatch size
            e = N*(M-1)*(M-2), number of edges in 3D adjacency (no diagonals)
              N = num_particles
              M = num neighbors
            k = input channels
        segID_3D (tensor): (b, 7, e) segment ids for pooling, 7 total:
            [col-depth, row-depth, row-col, depth, col, row, all]
        layer_id (int): layer id in network, for retrieving layer vars
    Returns:
        tensor of shape (b, e, q) if not is_last else (b, N*(M-1), q)
    """
    b, N = bN
    num_segs = b*N
    # get layer weights
    wmap, B = utils.get_RotInv_layer_vars(layer_id)
    '''
    wmap = {'CD': W1, # col-depth
            'RD': W2, # row-depth
            'RC': W2, # row-col
            'D' : W3, # depth
            'C' : W3, # col
            'R' : W4, # row
            'A' : W5, # all
            'Z' : W6} # none (no pooling)
    '''

    # Helper funcs
    # ========================================
    def _left_mult(h, pool_op):
        W = wmap[pool_op]
        return left_mult(h, W, SUBSCRIPT_ROTINV) #'bek,kq->beq'

    # Forward pass
    # ========================================
    # No pooling
    H = _left_mult(H_in, 'Z')

    # Pooling ops, ORDER MATTERS
    for i, pool_op in enumerate(SEGNAMES_3D): # ['CD', 'RD', 'RC', 'D', 'C', 'R', 'A']
        pooled_H = pool_RotInv_graph_conv(H_in, segID_3D[:,i], num_segs, broadcast=True)
        H = H + _left_mult(pooled_H, pool_op)

    # Output
    # ========================================
    H_out = H + B # (b, e, q)
    if is_last:
        # pool over depth dimension: (b, e, q) --> (b, N*(M-1), q)
        H_out = pool_RotInv_graph_conv(H_out, segID_3D[:,3], num_segs, broadcast=False)
    return H_out



#------------------------------------------------------------------------------
#                # Rotation Invariant Network function #
#------------------------------------------------------------------------------
def network_func_RotInv(edges, segID_3D, num_layers, dims, activation, redshift=None):
    """
    Args:
        edges (tensor): (b, e, k)
            b = minibatch size
            e = N*(M-1)*(M-2), number of edges in 3D adjacency (no diagonals)
              N = num_particles
              M = num neighbors
            k = input channels
        segID_3D (tensor): (b, 7, e) segment ids for pooling, 7 total:
            [col-depth, row-depth, row-col, depth, col, row, all]
        num_layers (int): network depth, for retrieving layer variables
        dims tuple(int): (b, N)
        activation (tf.function): activation function, (defaults tf.nn.relu)
        redshift (tensor): (-1, 1) vector of broadcasted redshifts
    Returns:
        tensor of shape (b, e, q) if not is_last else (b, N*(M-1), q)
    """

    # Input layer
    # ========================================
    H = activation(RotInv_layer(edges, segID_3D, dims, 0))

    # Hidden layers
    # ========================================
    for layer_idx in range(1, num_layers):
        is_last = layer_idx == num_layers - 1
        H = RotInv_layer(H, segID_3D, dims, layer_idx, is_last=is_last)
        if not is_last:
            H = activation(H)
    return H


#------------------------------------------------------------------------------
#                # Rotation Invariant Model function #
#------------------------------------------------------------------------------
def model_func_RotInv(X_in, edges, segID_3D, segID_2D, model_specs, redshift=None):
    """ Rotation invariant model function, wraps network function
    Inputs
    ------
    X_in  : tensor, original input data
        shape : (b, N, 3)
              : b batch_size, N particles

    edges : tensor, processed input data
        shape : (b, e, 10)
            b : batch size
            e : N*(M-1)*(M-2), num of edges in 3D adjacency
                : N particles, M neighbors
           10 : input channels
                [0]  : 1 edge feature
                [1:] : 9 broadcasted surface features
                        : 3 surfaces
                            : 1 scalar distance
                            : 1 row velocity projected onto cols
                            : 1 col velocity projected onto rows

    segID_3D : tensor, segment ids for pooling
        shape : (b, 7, e)
            7 : number of segment ids, ordered
                : [col-depth, row-depth, row-col, depth, col, row, all]

    segID_2D : tensor, row and col indices of 2D sparse adjacency matrix
        shape : (2, b*N*(M-1), 2)
            shape[i] : row, col indices, respectively
            shape[:, :, j] : batch offset for nonzero entries in  rows, cols

    model_specs : utils.AttrDict (dict), container for network configuration
         var_scope : str, the variable scope of this model, used for getting vars
        num_layers : int, number of layers (depth) of network
              dims : tuple int, dimensions of original data
                       : (b, N, M)
        activation : tf func, defaults to tf.nn.relu

    redshift : tensor, optional tensor vector of broadcasted reshift values
        shape : UNDETERMINED ???

    Returns
    -------
    tensor : (b, N, 3)
        Final positions (displacement ???) of particles

    """
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope
    num_layers = model_specs.num_layers
    activation = model_specs.activation # default tf.nn.relu
    dims = model_specs.dims
    b, N, M = dims


    # Network forward
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        #==== Network output
        net_out = network_func_RotInv(edges, segID_3D, num_layers, (b, N),
                                      activation, redshift=Redshift)

        #==== Final position of particles
        H_out = get_final_position(X_in, segID_2D, net_out, M)

        return H_out


