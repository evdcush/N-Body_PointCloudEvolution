import os, code, sys, time
from tabulate import tabulate
import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.sparse import coo_matrix, csr_matrix
import tensorflow as tf

#=============================================================================
# NOTES
#=============================================================================
'''
 - Are you sure you need to offset the segment ids in prep_RotInv_adjacency_batch?

'''
#import utils
#from utils import VAR_SCOPE
#=============================================================================
# Globals
#=============================================================================
'''
# segment labels
SEGNAMES_3D = ['CD', 'RD', 'RC', 'D', 'C', 'R', 'A']

# weight mapping
WMAP_3D = {'CD': 1, # col-depth
           'RD': 2, # row-depth
           'RC': 2, # row-col
           'D' : 3, # depth
           'C' : 3, # col
           'R' : 4, # row
           'A' : 5, # all
           'Z' : 6} # none (no pooling)

#=============================================================================
# ROTATION INVARIANT UTILS (normally in utils.py)
#=============================================================================
#------------------------------------------------------------------------------
# Network params init
#------------------------------------------------------------------------------

# @@@@@@@@@@@@@@@@@@@@@@@@@@@ IMPLEMENTED, see utils.initialize_RotInv_params
def init_RotInv_params(channels, var_scope, restore=False, seed=None):
    """ Init parameters for perm-equivariant, rotation-invariant model
    For every layer in this model, there are 6 weights (k, q) and 1 bias (q,)
        row-depth, row-col share weight
        depth, col share weight
    """

    # Get (k_in, k_out) tuples from channels
    # ========================================
    kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]

    # Initialize all layer weights
    # ========================================
    with tf.variable_scope(var_scope):
        for layer_idx, ktup in enumerate(kdims):
            # bias
            utils.init_bias(*ktup, utils.BIAS_TAG.format(layer_idx), restore=restore) # B

            # weights
            for w_idx in set(WMAP_3D.values()): # [1, 2, 3, 4, 5, 6]
                wtag = utils.MULTI_WEIGHT_TAG.format(layer_idx, w_idx)
                utils.init_weight(*ktup, wtag, restore=restore, seed=seed)


#------------------------------------------------------------------------------
# Var getters (CALLEE ASSUMES with tf.variable_scope)
#------------------------------------------------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@ IMPLEMENTED both, see utils.get_RotInv_layer_vars
def get_scoped_RotInv_weight(layer_idx, w_idx):
    W = tf.get_variable(utils.MULTI_WEIGHT_TAG.format(layer_idx, w_idx))
    return W

def get_scoped_bias(layer_idx):
    B = tf.get_variable(utils.BIAS_TAG.format(layer_idx))
    return B

#=============================================================================
# ROTATION INVARIANT NETWORK/LAYER OPS
#=============================================================================
#------------------------------------------------------------------------------
# ROTATION invariant layer ops
#------------------------------------------------------------------------------
# @@@@@@@@@@@@@@@@@@@@@@@@@@@ IMPLEMENTED, see nn.pool_RotInv_graph_conv
def pool_RotInv(X, idx, broadcast=True):
    """
    Pooling for rotation invariant model, assumes everything in row-major order
    Args:
        X (tensor): (b,e,k)
        idx (tensor): (b,e)
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

# @@@@@@@@@@@@@@@@@@@@@@@@@@@ IMPLEMENTED, see nn.RotInv_layer
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
    # Helper funcs
    # ========================================
    def _left_mult(h, w_idx):
        W = get_scoped_RotInv_weight(layer_id, w_idx)
        return tf.einsum("bek,kq->beq", h, W)

    # Forward pass
    # ========================================
    # No pooling
    H = _left_mult(H_in, WMAP_3D['Z'])

    # Pooling ops, ORDER MATTERS
    for i, pool_op in enumerate(SEGNAMES_3D): # ['CD', 'RD', 'RC', 'D', 'C', 'R', 'A']
        pooled_H = pool_RotInv(H_in, segID_3D[:,i], broadcast=True)
        H = H + _left_mult(pooled_H, WMAP_3D[pool_op])

    # Output
    # ========================================
    H_out = H + get_scoped_bias(layer_id) # (b, e, q)
    if is_last:
        # pool over depth dimension: (b, e, q) --> (b, N*(M-1), q)
        H_out = pool_RotInv(H_out, segID_3D[:,3], broadcast=False)
    return H_out


#=============================================================================
# ROTATION INVARIANT PRE/POST PROCESSING
#=============================================================================
#------------------------------------------------------------------------------
# Segment IDs
#------------------------------------------------------------------------------
def get_batch_2D_segmentID(batch_graph): #@@@@@@@@@@@@@@@@@@@@@@@@@@@ IMPLEMENTED
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


def get_3D_segmentID(adj_graph, M):#@@@@@@@@@@@@@@@@@@@@@@@@@@@ IMPLEMENTED, see nn.get_3D_segmentID
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


def get_segmentID(lst_csrs, M):#@@@@@@@@@@@@@@@@@@@@@@@@@@@ IMPLEMENTED, see nn.get_segmentID
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


#------------------------------------------------------------------------------
# Make rotational invariant input
#------------------------------------------------------------------------------
def get_RotInv_input(X, V, lst_csrs, M):#@@@@@@@@@@@@@@@@@@@@@@@@@@@ IMPLEMENTED, see nn.get_RotInv_features
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
    # out dims
    batch_size, N = X.shape[:2]
    e = N*(M-1)*(M-2)
    X_out = np.zeros((batch_size, e, 10)).astype(np.float32)

    # Iterate over each cube in batch
    # ========================================
    for i in range(batch_size):
        x = X[i]
        v = V[i]
        adj = lst_csrs[i]

        # Get 3D seg ID (coo features)
        rows, cols, depth = get_3D_segmentID(adj, M)

        # Relative dist vectors
        dist_cr = x[cols] - x[rows]
        dist_dr = x[depth] - x[rows]
        dist_dc = x[depth] - x[cols]

        # Edge features
'''


# STOPPED HERE
#=============================================================================
#=============================================================================
#=============================================================================
#=============================================================================

# Post-process output
# ========================================
def get_final_position(X_in, segment_idx_2D, weights, m, scalar):
    """
    Calculate displacement vectors = linear combination of neighbor relative positions, with weights = last layer
    outputs (pooled over depth), and add diplacements to initial position to get final position.

    Args:
        X_in. Shape (b, N, 3). Initial positions.
        segment_idx_2D . Shape (2, b * N * (M-1), 2). Each pair in the third axis is a batch idx - row idx or
            batch idx - col idx for non-zero entries of 2D adjacency.
            0-axis is rows/cols respectively. Get it from get_segment_idx_2D()
        weights. Shape (b, N, M - 1, 1). Outputs from last layer (pooled over depth dimension).
        m (int). Number of neighbors.
    dX_reshaped = tf.reshape(dX, [tf.shape(X_in)[0], tf.shape(X_in)[1], m - 1, tf.shape(X_in)[2]])  # (b, N, M - 1, 3)
    Returns:
        Tensor of shape (b, N, 3). Final positions.
    """

    # Find relative position of neighbors (neighbor - node)
    # (b, E, )
    dX = tf.gather_nd(X_in, segment_idx_2D[1]) - tf.gather_nd(X_in, segment_idx_2D[0])


    # Note: we want to normalize the dX vectors to be of length one,
    #  i.e. dX_reshaped[i, j, k, 0]^2 + dX_reshaped[i, j, k, 1]^2 + dX_reshaped[i, j, k, 2]^2 = 1 for any i, j, k.
    dX_reshaped = tf.reshape(dX, [tf.shape(X_in)[0], tf.shape(X_in)[1], m - 1, tf.shape(X_in)[2]])  # (b, N, M - 1, 3)[-1, 3]  --> (b*N*M, )
    # dX_norm = np.linalg.norm([b*N*M, 3], axis=1) --> resshape (b, N, M-1, 1)
    dX_out = dX_reshaped / dX_norm

    # Return initial position + displacement (=weighted combination of neighbor relative distances)
    # Note: we want to rescale the second term by a learnable scalar parameter
    #       and add the linear displacement, same as in shift_invariant setup.
    #return X_in + tf.reduce_sum(tf.multiply(dX_reshaped, weights), axis=2)
    return X_in + scalar * tf.reduce_sum(tf.multiply(dX_reshaped, weights), axis=2)


#=============================================================================
# ROTATION INVARIANT PRE/POST PROCESSING
#=============================================================================
#=============================================================================
# ROTATION INVARIANT PRE/POST PROCESSING
#=============================================================================
#================================s============================================
# ROTATION INVARIANT PRE/POST PROCESSING
#=============================================================================
#=============================================================================
# ROTATION INVARIANT PRE/POST PROCESSING
#=============================================================================
#=============================================================================
# ROTATION INVARIANT PRE/POST PROCESSING
#=============================================================================









def pre_process_adjacency_batch(batch, m, sparse=True):
    """
    Process batch of adjacency matrices and return segment_idx.

    Args:
        batch. List of adjacency matrices. Each matrix can be dense NxN or any scipy sparse format, like csr.
        m (int). Number of neighbors.
        sparse (bool). If True, matrices in the batch must be in sparse format.

    Returns:
        numpy array with shape (b, 7, e)
            e=N*(M-1)*(M-2), number of edges in 3D adjacency (diagonals removed), N=num of particles, M=num of neighbors
            The seven arrays correpond to segment_idx for pooling over col-depth, row-depth, row-col, depth, col,
            row, all, respectively
    """

    def _combine_segment_idx(idx_1, idx_2):
        """
        Combine pairs of segment idx.
        # Why use return_inverse instead of return_index?
        """
        idx_12 = np.transpose(np.array([idx_1, idx_2]))  # pair up idx
        vals, idx = np.unique(idx_12, axis=0, return_inverse=True)

        return idx  # return idx of unique pairs

    out = []

    for a in batch:

        # Get all segment idx for pooling
        # row, col, depth indices correspond to segment_idx for pooling over col-depth, row-depth, row-col, respectively
        if sparse:
            r_idx, c_idx, d_idx = _make_cube_adjacency_sparse(A_sparse=a, m=m)
        else:
            r_idx, c_idx, d_idx = _make_cube_adjacency_dense(A_dense=a)

        # By combining pairs, will get segment idx for pooling over depth (combine row and col),
        # col (combine row and depth), and row (combine col and depth)
        rc_idx = _combine_segment_idx(r_idx, c_idx)
        rd_idx = _combine_segment_idx(r_idx, d_idx)
        cd_idx = _combine_segment_idx(c_idx, d_idx)

        # Get idx for pooling over all
        all_idx = np.zeros_like(r_idx)

        out.append(np.array([r_idx, c_idx, d_idx, rc_idx, rd_idx, cd_idx, all_idx]))

    out = np.array(out)

    # Offset: note that number of segments is not always N as for 2D case
    for i in range(1, out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j] += np.max(out[i - 1][j]) + 1

    return out


def _make_cube_adjacency_dense(A_dense):
    """
    Build brute-force 3D adjacency from dense NxN input.
    This is just for testing/debugging. Don't use this.
    """
    N = A_dense.shape[0]
    A_cube = np.zeros(shape=[N, N, N], dtype=np.int32)

    for i in range(N):
        for j in range(N):
            if A_dense[i, j] > 0:
                A_cube[i, j, :] = A_dense[i]

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i == j or i == k or j == k:
                    A_cube[i, j, k] = 0

    return np.nonzero(A_cube)


def _make_cube_adjacency_sparse(A_sparse, m):
    """
    Build 3D adjacency from sparse input.

    Args:
        A_sparse. csr_matrix or any other scipy sparse format.
        m (int). number of neighbors.

    Returns:
        row, col, depth. numpy arrays for indices of non-zero entries. Diagonals removed.
    """
    A_sparse.setdiag(0)  # Don't need diagonal elements
    m_eff = m - 1 # IF NO SELF

    rows, cols = A_sparse.nonzero()

    # Will fill indices for rows, columns, depth, in this order.
    r = []
    c = []
    d = []

    for i in range(len(rows)):
        r.extend([rows[i]] * (m_eff - 1))
        c.extend([cols[i]] * (m_eff - 1))
        cumulative_m = (rows[i] + 1) * m_eff
        depth_idx = cols[cumulative_m - m_eff:cumulative_m]
        depth_idx = np.delete(depth_idx, np.where(depth_idx==cols[i]))  # Remove neighbor-neighbor diagonal
        d.extend(depth_idx)

    return np.array(r), np.array(c), np.array(d)


def get_segment_idx_2D(batch_A_sparse):
    """
    Return row, col, indices of a list of 2D sparse adjacencies with batch indices too.
        Sorry, using a different indexing system from 3D adjacency case. TODO: make indexing consistent for clarity.

    Args:
        batch_A_sparse. List of csr (or any other sparse format) adjacencies.

    Returns:
        array of shape (2, b * N * (M-1), 2). Each pair in the third axis is a batch idx - row idx or
            batch idx - col idx for non-zero entries of 2D adjacency. 0-axis is rows/cols respectively.
    """
    rows = []
    cols = []

    for i in range(len(batch_A_sparse)):
        a = batch_A_sparse[i]
        a.setdiag(0)
        r, c = a.nonzero()
        batch = np.zeros_like(r) + i
        rows.append(np.transpose([batch, r]))
        cols.append(np.transpose([batch, c]))

    rows = np.reshape(np.array(rows), (-1, 2))
    cols = np.reshape(np.array(cols), (-1, 2))

    return np.array([rows, cols])


# Pre-process input
# ========================================
def rot_invariant_input(batch_X, batch_V, batch_A, m):
    """
    Args:
         batch_X. Shape (b, N, 3). Coordinates.
         batch_V. Shape (b, N, 3), Velocties.
         batch_A. List of csr adjacencies.
         m (int). Number of neighbors.

    Returns:
        numpy array of shape (b, e, 10)
            e=N*(M-1)*(M-2), number of edges in 3D adjacency (diagonals removed), N=num of particles, M=num of neighbors
            10 input channels corresponding to 1 edge feature + 9 broadcasted surface features, those are broken
            down into 3 surfaces x (1 scalar distance + 1 row velocity projected onto cols + 1 col velocity
            projected onto rows)
    """
    def _process(X, V, A):
        def _angle(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        def _norm(v):
            return np.linalg.norm(v)

        def _project(v1, v2):
            return np.dot(v1, v2) / np.linalg.norm(v2)

        rows, cols, depth = _make_cube_adjacency_sparse(A, m)

        X_out = []
        for r, c, d in zip(rows, cols, depth):

            # Relative distance vectors
            dx1 = X[c] - X[r]
            dx2 = X[d] - X[r]
            dx3 = X[d] - X[c]

            # Edge features
            features = [_angle(dx1, dx2)]

            # rc surface features
            # scalar distance + projection of row vel to rc vectors + projection of col vel to cr vectors
            features.extend([_norm(dx1), _project(V[r], dx1), _project(V[c], -dx1)])

            # rd surface features
            # scalar distance + projection of row vel to rd vectors + projection of depth vel to dr vectors
            features.extend([_norm(dx2), _project(V[r], dx2), _project(V[d], -dx2)])

            # cd surface features
            # scalar distance + projection of col vel to cd vectors + projection of depth vel to dc vectors
            features.extend([_norm(dx3), _project(V[c], dx3), _project(V[d], -dx3)])

            X_out.append(features)

        return X_out

    return np.array([_process(batch_X[i], batch_V[i], batch_A[i]) for i in range(len(batch_X))])


# Post-process output
# ========================================
def get_final_position(X_in, segment_idx_2D, weights, m):
    """
    Calculate displacement vectors = linear combination of neighbor relative positions, with weights = last layer
    outputs (pooled over depth), and add diplacements to initial position to get final position.

    Args:
        X_in. Shape (b, N, 3). Initial positions.
        segment_idx_2D . Shape (2, b * N * (M-1), 2). Each pair in the third axis is a batch idx - row idx or
            batch idx - col idx for non-zero entries of 2D adjacency.
            0-axis is rows/cols respectively. Get it from get_segment_idx_2D()
        weights. Shape (b, N, M - 1, 1). Outputs from last layer (pooled over depth dimension).
        m (int). Number of neighbors.

    Returns:
        Tensor of shape (b, N, 3). Final positions.
    """

    # Find relative position of neighbors (neighbor - node)
    dX = tf.gather_nd(X_in, segment_idx_2D[1]) - tf.gather_nd(X_in, segment_idx_2D[0])
    dX_reshaped = tf.reshape(dX, [tf.shape(X_in)[0], tf.shape(X_in)[1], m - 1, tf.shape(X_in)[2]])  # (b, N, M - 1, 3)

    # Return initial position + displacement (=weighted combination of neighbor relative distances)
    return X_in + tf.reduce_sum(tf.multiply(dX_reshaped, weights), axis=2)


# Example
# ========================================
def toy_example():

    # Parameters
    # ============================
    N = 3  # number of particles
    M = 3  # number of neighbors
    b = 2  # minibatch size

    e = N * (M - 1) * (M - 2)  # number of edges in 3D adjacency (diagonal removed)

    # Graph
    # ============================
    # Make up random weights
    # Normally there would be 8 independent weight matrices - but since the two neighbor dimensions are excheangeable
    # row-depth pooling and row-col pooling share the same weight, as well as col pooling and depth pooling.
    # So, there are 6 independent, instead of 8.
    W1 = tf.Variable(tf.constant(1., shape=[10, 1]))
    W2 = tf.Variable(tf.constant(2., shape=[10, 1]))
    W3 = tf.Variable(tf.constant(3., shape=[10, 1]))
    W4 = tf.Variable(tf.constant(4., shape=[10, 1]))
    W5 = tf.Variable(tf.constant(5., shape=[10, 1]))
    W6 = tf.Variable(tf.constant(6., shape=[10, 1]))

    W = {
        "no-pooling": W1,
        "col-depth": W2,
        "row-depth": W3,
        "row-col": W3,
        "depth": W4,
        "col": W4,
        "row": W5,
        "all": W6
    }
    B = tf.Variable(tf.constant(1., shape=[1]))

    # Inputs
    _X_edges = tf.placeholder(tf.float32, [b, e, 10])
    _segment_idx_3D = tf.placeholder(tf.int32, [b, 7, e])
    _X_in = tf.placeholder(tf.float32, [b, N, 3])
    _segment_idx_2D = tf.placeholder(tf.int32, [2, b * N * (M - 1), 2])

    layer_output = rot_inv_layer(
        X_edges=_X_edges,
        segment_idx_3D=_segment_idx_3D,
        W=W,
        B=B,
        activation=tf.nn.relu,
        is_last=True  # This should be set to True only if it's the last layer (depth is pooled)
    )

    weights = tf.reshape(layer_output, [b, N, M - 1, 1])

    final_positions = get_final_position(
        X_in=_X_in,
        segment_idx_2D=_segment_idx_2D,
        weights=weights,
        m=M
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Pre-processing
        # ============================
        # Batch of adjacencies, b=2, N=3, M=3
        # Number of edges (i.e., non-zero entries in the 3D adjacency) e = N*(M-1)*(M-2) = 6,
        # E has to be constant across the batch.
        # This is a particularly simple case but shows the expected behavior.
        A = np.array([
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        ])

        # Make adjacencies in sparse format (can be any of the scipy sparse formats)
        A_csr = [csr_matrix(a) for a in A]

        # Get segment idx that correspond to the 7 pooling operations.
        # They're automatically offset along batch.
        segment_idx_3D = pre_process_adjacency_batch(batch=A_csr, sparse=True, m=M)  # (b, 7, e), b=2, e=6

        print(segment_idx_3D[:, 0, :])

        # Print(segment_idx
        # CAN MANUALLY CHECK THAT THE SEGMENT IDX ARE INDEED THE EXPECTED ONES FOR THE CORRESPONDING POOLING OPERATION
        segment_names = ["col-depth", "row-depth", "row-col", "depth", "col", "row", "all"]

        print("First matrix")
        print(tabulate(
            [[name, segment_idx_3D[0][i], np.max(segment_idx_3D[0][i]) + 1] for i, name in enumerate(segment_names)],
            headers=["Pooling", "Segment idx", "Number of Segments"]
        ))

        # In this case poolings are the same - note offset idx.
        print("\nSecond matrix")
        print(tabulate(
            [[name, segment_idx_3D[1][i], np.max(segment_idx_3D[0][i]) + 1] for i, name in enumerate(segment_names)],
            headers=["Pooling", "Segment idx", "Number of Segments"]
        ))

        # Make up random coordinate vectors
        X_in = np.random.rand(2, 3, 3)  # (b, N, 3)

        # Make up random velocities
        V = np.random.rand(2, 3, 3)  # (b, N, 3)

        # Generate input
        # This generate 10 input channels corresponding to 1 edge feature + 9 broadcasted surface features,
        # those are broken down into 3 surfaces x (1 scalar distance + 1 row velocity projected onto cols
        # + 1 col velocity projected onto rows)
        X_edges = rot_invariant_input(batch_X=X_in, batch_V=V, batch_A=A_csr, m=M)  # (b, e, 10), b=2, e=6

        print("-----------")
        print("Input shape: %s" % str(X_edges.shape))

        segment_idx_2D = get_segment_idx_2D(batch_A_sparse=A_csr)

        # Output
        # ============================
        out = sess.run(
            final_positions,
            feed_dict={
                _X_edges: X_edges,
                _segment_idx_3D: segment_idx_3D,
                _X_in: X_in,
                _segment_idx_2D: segment_idx_2D
            }
        )

        print("Output shape: %s" % str(out.shape))


if __name__ == "__main__":
    toy_example()




def network_func_RotInv(X_in, segID_3D, num_layers, dims, activation, redshift=None):
    """
    Args:
        X_in (tensor): (b, e, k)
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
    pass

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
        num_feats = net_out.get_shape().as_list()[-1]

        # ==== Scale network output and compute skip connections
        loc_scalar, vel_scalar = utils.get_scalars()
        H_out = net_out[...,:3]*loc_scalar + X_in_loc + X_in_vel*vel_scalar

        # ==== Concat velocity predictions
        if net_out.get_shape().as_list()[-1] > 3:
            H_vel = net_out[...,3:]*vel_scalar + X_in_vel
            H_out = tf.concat([H_out, H_vel], axis=-1)
        return get_readout(H_out)

'''
#############

Hey guys,

sorry for the delay. Attached is my code for the rotational invariant model.
It required some work - but I think it should implement what we discussed.
Here are some remarks, I'd appreciate if you could share your comments.

1. I have included an end-to-end toy example, which shows how the different
parts should be used together, and how poolings (which are the tricky part) work.

2. Inputs for the rot-invariant layers have now shape X = (b, e, k),
where b=batch, k=channels and e is the number of edge features. Edge features
sit on the non-zero entries of a 3D adjacency with shape (N, M, M), which is
symmetric under exchange of the last two dimensions. The number of edge
features is e=N*(M-1)*(M-2). I'm working under the assumption of fixed number
of neighbors M for now, and the code removes the diagonals from the 3D
adjacency (given that we fill edges with the angle between <nm, nm'>, one
should have n!=m and n!=m'; additionally, broadcasting surface features to
edges (see below for details) also requires m != m', so I have removed all
diagonal elements). Note that the subsampling trick Siamak suggested last
time for dealing with variable number of neighbors, required a fixed number
of 2D edges across the batch (i.e., fixed total number of non-zero entries in
the adjacency across the batch) for the shift-invariant case; but now it would
require a fixed number of 3D edges, which is not guaranteed to be constant
even if the 2D edges are fixed. I'm postponing these issues for later
discussion, and sticking with fixed M for now.

3. As you can see, the layer is very simple in principle, all operations
are very symmetric. Here is the main idea: we have a three-dimensional
adjacency tensor, whose non-zero elements (there are e of them) are contained
in X, in row-column-depth order (generalization of row-major order). There are
7 possible pooling + no-pooling, which correspond to 8 sets of weights. Think
about the 2D case first: row indices of non-zero adjacency elements define
segments for pooling over columns, and viceversa column indices define segments
for pooling over rows. For the 3D case we have row, column and depth indices
which indicate the non-zero entries and correspond exactly to pooling over
col-and-depth, row-and-depth, row-and-col, respectively. Additionally, a
proper combination of row and col indices defines segments for pooling over
depth, a combination of row and depth indices defines segments for pooling
over col, and a combination of col and depth indices defines segments for
pooling over row (you can see the code for how the combination is calculated).
Finally, there is a pool over all e edges.

4. Because adjacency is symmetric for exchange of col and depth, I think we
should actually have 6 independent sets of weights, instead of 8. row-and-col
and row-and-depth should share the same weight. Same for col and depth, as I
did in the toy example. Also, I've assumed the batch of adjacency is in csr
(or any other scipy sparse format).

5. There are functions for preprocessing and postprocessing.
Preprocessing: the input data is a tensor of shape (b, e, 10).
There are 10 channels because: 1 true edge feature
(angle between <nm, nm'> + 9 features coming from surface broadcasting.
Assume that the surface is IJ, where I and J can be {row, col, depth},
and they are indexed by ij. There are 3 possible surfaces, and for each
surface there are 1 scalar distance for the pair ij + 1 projection of
velocity_i onto vector_ij + 1 projection of velocity_j onto vector_ji = - vector_ij.
This gives 3 surfaces x 3 features = 9 features coming from surface broadcasting.
I think this is one possible way of encoding the input (possibly redundant?),
there could be others. For example, once we have a triplet of particles
identified by an edge, we could add all 3 relative angles on the edge,
instead of only one angle and two scalar distances on the surface.

6. Postprocessing: last layer of the network (once you set is_last=True) pools
over the depth dimension, and so returns an output X_out of shape (b, N*(M-1), q).
With q=1 this can be reshaped to (b, N, M-1, 1). For each particle, we take a
linear combination of the relative distances of its neighbors with weight given
by X_out, this gives the displacement. Once summed to the initial position, you
get the final position.

7. I have tested separate pieces of the code and some overall functionalities -
but I have not done extensive tests. Some code, especially for the pre or
post processing, can definitely be cleaned up.
'''
