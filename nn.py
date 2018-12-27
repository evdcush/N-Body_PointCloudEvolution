import code, sys
import numpy as np
import tensorflow as tf
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import coo_matrix
"""
MAKE SURE include_self=True FOR ZA DATASET
"""


# ALL NN OPS TO BE UPDATED

"""
* passing num_layers no longer req'd if you passing an Initializer for var gets
  * also, passing var_scope is unncessary with Initializer
"""

def include_node_features(X_in_edges, X_in_nodes, COO_feats, redshift=None):
    """ Broadcast node features to edges for input layer
    Params
    ------
    X_in_edges : tensor; (c,3)
        input edges features (relative pos of neighbors)
    X_in_nodes : tensor; (b*N, 3)
        input node features (velocities)
    COO_feats : tensor; (3, c)
        rows, cols, cube indices

    Returns
    -------
    X_in : tensor; (c,9)
        model graph input, with node features broadcasted to edges
    """
    # ==== get row, col indices
    row_idx = COO_feats[0]
    col_idx = COO_feats[1]

    # ==== get node row, columns
    node_rows = tf.gather_nd(X_in_nodes, tf.expand_dims(row_idx, axis=1))
    node_cols = tf.gather_nd(X_in_nodes, tf.expand_dims(col_idx, axis=1))

    # ==== full node, edges graph
    X_in = tf.concat([X_in_edges, node_rows, node_cols], axis=1) # (c, 9)

    # ==== broadcast redshifts
    if redshift is not None:
        X_in = tf.concat([X_in, redshift], axis=1) # (c, 10)
    return X_in

    """
    There will be ONE edge in edges that is zeros because difference between
    self
    - we want to insert ZA_displacements here

    For a particle, there is one neighbor index in adjacency that is equal
    to its index in cube, when include_self=True
      - thus it will be zero if we diff
      - THIS IS WHERE YOU INSERT ZA_displacements
    """


def get_input_features_shift_inv_ZA(init_pos, ZA_displacement, coo, diag, dims):
    """ get edges and nodes with TF ops
    get relative distances of each particle from its M neighbors
    use diff tensor

    Params
    ------
    init_pos : tensor; (b, N, 3)
        initial positions on the grid

    ZA_displacement : tensor; (b, N, 3)
        za displacement vector, X[...,1:4]

    coo : tensor; (3, b*N*M)
        the sparse adjacency matrix, MADE FROM init_pos
        # Diagonal indices notes:
        -
    diag : tensor; (b*N)
        the diagonal indices; the indices where the particles in init_pos are
        neighboring nodes with themself (via `include_self=True)

    Returns
    -------
    edges : tensor; (C, 3)
        NO NODES
    """
    def _broadcast_to_diag(h, broadcast_idx, shape):
        """Broadcast values to diagonal.
        Args:
            h(tensor). Values to be broadcasted to a diagonal.
            broadcast_idx(tensor). Diagonal indices, shape = (b*N)
            shape(tensor). The shape of the output, should be (S, q)
        Returns:
            tensor with specified shape
        """
        return tf.scatter_nd(tf.expand_dims(broadcast_idx, axis=1), h, shape)

    b, N, M = dims
    #==== get edges (neighbors)
    flattened_pos = tf.reshape(init_pos, (-1, 3))
    cols = coo[1]
    #gath_edges = tf.gather(flattened_pos, cols)
    #code.interact(local=dict(globals(), **locals()))
    #edges = tf.reshape(gath_edges, [b, N, M, 3])
    edges = tf.reshape(tf.gather(flattened_pos, cols), [b, N, M, 3])

    #=== weight edges
    edges = edges - tf.expand_dims(init_pos, axis=2) # (b, N, M, 3) - (b, N, 1, 3)

    #=== broadcast ZA_displacements to diagonal
    flattened_za_disp = tf.reshape(ZA_displacement, (-1, 3))
    out_shape = (b*N*M, 3)
    diagonal_za = _broadcast_to_diag(flattened_za_disp, diag, out_shape)
    features_out = tf.reshape(edges, [-1, 3]) + diagonal_za
    return features_out


def get_input_features_shift_inv(X_in, coo, dims):
    """ get edges and nodes with TF ops
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


def shift_inv_conv(h, pool_idx, num_segs, broadcast):
    """
    Params
    ------
    h : tensor; (c, k)
        data to be avgd, row-major order
    pool_idx : tensor; (c,)
        indices for pooling over segments in data
        - column indices ---> pools rows
        - row indices    ---> pools cols
        - cube indices   ---> rows and cols
    num_segs : int
        number of segments in h (eg, num of particles in h)
    broadcast : bool
        re-broadcast to original shape after pooling

    Returns
    -------
    pooled_conv : tensor
        shape (c, k) if broadcast else (num_segs, k)
    """
    pooled_conv = tf.unsorted_segment_mean(h, pool_idx, num_segs)
    if broadcast:
        pooled_conv = tf.gather_nd(pooled_conv, tf.expand_dims(pool_idx, axis=1))
    return pooled_conv


def shift_inv_layer(H_in, COO_feats, bN, layer_vars, is_last=False):
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

    # split vars
    weights, B = layer_vars
    W1, W2, W3, W4 = weights

    # Helper funcs
    # ========================================
    def _pool(H, idx, broadcast=True):
        return shift_inv_conv(H, idx, b*N, broadcast)

    def _left_mult(h, W):
        return tf.einsum('ck,kq->cq', h, W)

    # Layer forward pass
    # ========================================
    # H1 : no pooling
    #code.interact(local=dict(globals(), **locals()))
    # H1.shape = (1835008, 3)
    # W1.shape = (9, 32) <----- input chans off
    H1 = _left_mult(H_in, W1) # (c, q)

    # H2 : pool rows
    H_pooled_rows = _pool(H_in, col_idx)
    H2 = _left_mult(H_pooled_rows, W2) # (c, q)

    # H3 : pool cols
    H_pooled_cols = _pool(H_in, row_idx)
    H3 = _left_mult(H_pooled_cols, W3) # (c, q)

    # H4 : pool cubes
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

def network_func_shift_inv_za(edges, coo, num_layers, dims, activation, model_vars):
    # Input layer
    # ========================================
    H = activation(shift_inv_layer(edges, coo, dims, model_vars.get_layer_vars(0),))

    # Hidden layers
    # ========================================
    for layer_idx in range(1, num_layers):
        is_last = layer_idx == num_layers - 1
        layer_vars = model_vars.get_layer_vars(layer_idx)
        H = shift_inv_layer(H, coo, dims, layer_vars, is_last=is_last)
        if not is_last:
            H = activation(H)
    return H


def model_func_shift_inv_za(init_pos, COO_feats, ZA_displacement, ZA_diagonal,
                            model_vars, dims, activation=tf.nn.relu):
    """

    Params
    ------
    init_pos : tensor; (b, N, 3)
        the initial positions of particles on the grid

    coo_feats : tensor; (3, c) -- where c = b * N * M
        segment IDs for rows, cols, all

    ZA_displacement : tensor; (b, N, 3)
        za displacement vector, X[...,1:4]

    ZA_diagonal : tensor; (b*N,)
        diagonal indices of the initial positions

    model_vars : Initializer
        Initializer instance that has model config and variable utils
    """
    var_scope = model_vars.var_scope
    num_layers = len(model_vars.channels) - 1

    # Get graph inputs
    # ========================================
    #edges, nodes = get_input_features_shift_inv(X_in, COO_feats, dims)
    edges = get_input_features_shift_inv_ZA(init_pos, ZA_displacement,
                                            COO_feats, ZA_diagonal, dims)

    # Network forward
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        # ==== Network output
        pred_error = network_func_shift_inv_za(edges, COO_feats, num_layers,
                                               dims[:-1], activation, model_vars)
        return pred_error

'''
def _network_func_shift_inv(X_in_edges, X_in_nodes, COO_feats, num_layers,
                           dims, activation, model_vars, redshift=None):
    # Input layer
    # ========================================
    H_in = include_node_features(X_in_edges, X_in_nodes, COO_feats, redshift=redshift)
    H = activation(shift_inv_layer(H_in, COO_feats, dims, model_vars.get_layer_vars(0),))

    # Hidden layers
    # ========================================
    for layer_idx in range(1, num_layers):
        is_last = layer_idx == num_layers - 1
        layer_vars = model_vars.get_layer_vars(layer_idx)
        H = shift_inv_layer(H, COO_feats, dims, layer_vars, is_last=is_last)
        if not is_last:
            H = activation(H)
    return H


def _model_func_shift_inv(X_in, COO_feats, model_vars, dims, activation=tf.nn.relu, redshift=None):
    """
    Args:
        X_in (tensor): (b, N, 6)
        COO_feats (tensor): (3, B*N*M), segment ids for rows, cols, all
        redshift (tensor): (b*N*M, 1) redshift broadcasted
    """
    var_scope = model_vars.var_scope
    num_layers = len(model_vars.channels) - 1

    # Get graph inputs
    # ========================================
    edges, nodes = get_input_features_shift_inv(X_in, COO_feats, dims)

    # Network forward
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        # ==== Split input
        X_in_loc, X_in_vel = X_in[...,:3], X_in[...,3:]
        # ==== Network output
        net_out = network_func_shift_inv(edges, nodes, COO_feats, num_layers,
                                        dims[:-1], activation, model_vars, redshift)
        # ==== Scale network output
        loc_scalar, vel_scalar = model_vars.get_scalars()
        H_out = net_out[...,:3]*loc_scalar + X_in_loc + X_in_vel*vel_scalar

        # ==== Concat velocity predictions
        if net_out.get_shape().as_list()[-1] > 3:
            H_vel = net_out[...,3:]*vel_scalar + X_in_vel
            H_out = tf.concat([H_out, H_vel], axis=-1)
        return H_out
'''


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


def to_coo_batch_ZA_diag(A):
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
    dia = []

    # Get COO feats
    # ----------------
    COO_feats = np.zeros((3, b*N*M)).astype(np.int32)
    for i in range(b):
        #coo = A[i].tocoo()
        r, c = A[i].nonzero()

        # Offset coo feats
        row = r + i*N
        col = c + i*N
        cube = np.zeros_like(row) + i

        # Assign coo feats
        k, q = i*N*M, (i+1)*N*M
        COO_feats[0, k:q] = row
        COO_feats[1, k:q] = col
        COO_feats[2, k:q] = cube

        # Make diagonals
        d = np.array(np.where(r == c)[0])
        dia.extend(d + i * len(r))

    #code.interact(local=dict(globals(), **locals()))
    diagonals = np.array(dia)
    # sanity check
    #confirm_CSR_to_COO_index_integrity(A, COO_feats) # checked out
    return COO_feats, diagonals

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
def get_kneighbor_list(X_in, M, offset_idx=False, include_self=True):
    b, N, D = X_in.shape
    lst_csrs = []
    #print('nn.get_kneighbor_list\n M: {}, include_self: {}'.format(M, include_self))
    for i in range(b):
        kgraph = kneighbors_graph(X_in[i,:,:3], M, include_self=include_self).astype(np.float32)
        if offset_idx:
            kgraph.indices = kgraph.indices + (N * i)
        lst_csrs.append(kgraph)
    return lst_csrs


#=============================================================================
# RADIUS graph ops
#=============================================================================

def radius_graph_fn(x, R, include_self=True):
    """ Wrapper for sklearn.Neighbors.radius_neighbors_graph function

    Params
    ------
    x : ndarray.float32; (N, D)
        input data, where x[:,:3] == particle coordinates
    R : float
        neighborhood search radius

    Returns
    -------
    xR_ngraph : scipy.CSR; (N,N)
        sparse matrix representing each particle's neighboring
        particles within radius R
    """
    xR_ngraph = radius_neighbors_graph(x[...,:3], R, include_self=include_self)
    return xR_ngraph.astype(np.float32)

def get_radius_graph_COO(X_in, R):
    """ Normalize radius neighbor graph by number of neighbors

    This function prepares a single sample for direct conversion from
    scipy CSR format to tensorflow's SparseTensor, which is structured
    much like a modified scipy COO matrix.

    The matrix data is divided by the number of neighbors for each respective
    particle for the graph convolution operation in the network layer.
    """
    N = X_in.shape[0]
    # just easier to diff indptr for now
    # get csr
    rad_csr = radius_graph_fn(X_in, R)
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

# diff for ZA, temp try somethin else
def pbc_loss(x_pred, x_truth, scale_error=True):
    """ MSE over full dims with periodic boundary conditions
    Args:
        readout (tensor): model prediction which has been remapped to inner cube
        x_truth (tensor): ground truth (mb_size, N, 6)
        vel: if vel, then include vel error in loss
    """
    pbc_dist  = periodic_boundary_dist(x_pred, x_truth)
    error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1))
    if scale_error:
        error = error * 1e5
    return error


def loss_ZA(predicted_error, true_error):
    """
    We want net to predict the error from ZA approx to truth (FPM)
    thus:
        loss = predicted_error - true_error
    where
        predicted_error = network output
        # We want this AS close to possible as actual_error

        true_error == FPM_disp - ZA_disp

    predicted_error.shape == true_error.shape == (b, N, 3)
    """
    err_diff = tf.squared_difference(predicted_error, true_error) # (b, N, 3)
    error = tf.reduce_mean(tf.reduce_sum(err_diff, axis=-1))
    return error


#=============================================================================#
#                                                                             #
#      _ __    _   _   _ __ ___    _ __    _   _      ___    _ __    ___      #
#     | '_ \  | | | | | '_ ` _ \  | '_ \  | | | |    / _ \  | '_ \  / __|     #
#     | | | | | |_| | | | | | | | | |_) | | |_| |   | (_) | | |_) | \__ \     #
#     |_| |_|  \__,_| |_| |_| |_| | .__/   \__, |    \___/  | .__/  |___/     #
#                                 |_|      |___/            |_|               #

def mse_za(fpm_displacement, za_displacement):
    err_diff = np.square(fpm_displacement - za_displacement)
    error = np.mean(np.sum(err_diff, axis=-1))
    return error


def get_init_pos(za_disp):
    b, N, k = za_disp.shape
    mg = range(2, 130, 4)
    q = np.einsum('ijkl->kjli', np.array(np.meshgrid(mg, mg, mg)))
    qr = q.reshape(-1, 3)
    init_pos = za_disp + qr
    return init_pos







#=============================================================================#
#                                                                             #
#                   d888888P  .88888.  888888ba   .88888.                     #
#                      88    d8'   `8b 88    `8b d8'   `8b                    #
#                      88    88     88 88     88 88     88                    #
#                      88    88     88 88     88 88     88                    #
#                      88    Y8.   .8P 88    .8P Y8.   .8P                    #
#                      dP     `8888P'  8888888P   `8888P'                     #
#                                                                             #
#=============================================================================#
# https://arxiv.org/abs/1506.02025 # spatial trans
# https://arxiv.org/abs/1706.03762 # attn all u need (nlp)
# >>-----> https://arxiv.org/pdf/1710.10903.pdf  # graph attention nets GATs

def attn_layer(foo):
    """ see p.3,4 of GATs, eqns 1,2,6, fig1
        auth's TF code: https://github.com/PetarV-/GAT
    """
    pass
