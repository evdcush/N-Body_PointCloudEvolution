import os, code, sys
import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
#from scipy.sparse import coo_matrix
import tensorflow as tf

import utils
from utils import VARIABLE_SCOPE as VAR_SCOPE

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

# Updated Shift invariant input features
#   Asymmetrical adjacency
# ========================================
def get_input_features_ShiftInv_numpy(X_in, A, N, redshift):
    """Generate input for first layer.

    # DANIELE Notes
    #-----------------
    This is doing what
    https://github.com/evdcush/NBPCE/blob/master/nn.py#L81-L129
    were doing, **with the crucial difference** that the adjacency is
    symmetrized and zero features are associated with new entries coming
    from symmetrization.
      Even if number of neighbors is fixed, different matrices in the batch
    can have a different number of non-zero symmetrized entries.
      Even different rows within same adjacency can have different number of
    non-zero entries. So I have to do explicit loops.

    I implemented this in numpy which I think it's fine because it's a
    preprocessing step.

    Args:
        X_in(array). (b, N, 6) coord and velocities.
        A(list). Batch of csr adjacency.
        N(int). Number of particles.
        redshift(float).

    Returns:
        array of shape (S, 9) or (S, 10) if redshift is not None.
        S is defined above in new_ShiftInv_layer.
    """
    shiftinv_input = []
    for k, a in enumerate(A):
        a_symm = a + a.transpose()
        # split input coo/vel
        coordinates = X_in[k][:, :3]
        velocities  = X_in[k][:, 3:]

        for i in range(N):
            node_coordinates = X_in[k][i, :3]
            #==== get neighbor idx
            neighbors_idx_symm = a_symm.getrow(i).nonzero()[1]
            neighbors_idx = a.getrow(i).nonzero()[1]
            for j in neighbors_idx_symm:
                if j in neighbors_idx:
                    features = coordinates[j] - node_coordinates
                    features = np.concatenate((features, velocities[i],
                                                         velocities[j]), axis=0)
                else:
                    features = np.zeros(9)
                if redshift is not None:
                    features = np.append(features, redshift)
                shiftinv_input.append(features)
    return np.array(shiftinv_input)


# Updated/Corrected Shiftinv layer
# ===================================
def ShiftInv_layer(H_in, adj, bN, layer_id, is_last=False):
    """

    New basis with 15 independent weights.
      see: https://openreview.net/pdf?id=Syx72jC9tm.

    Let S = sum_i( symmetrized_i ) for i = 0...b-1, where b = batch size.
    symmetrized_i = number of non-zero entries of symmetrized adjacency.
    If adjacency is symmetric, then symmetrized_i = N*M;
    but in general it's not.

    Also, even if all instances in the batch have a fixed number of
    neighbors (i.e. all unsymmetrized adjacencies have N*M non-zero entries),
    the corresponding symmetrized versions can contain different number
    of entries.

    Our implementation where all dimensions but channels are flattened
    is handy to deal with this.

    Args:
        H_in(tensor). Shape = (S, k)
            k is number of input channels.
        adj: dict
            adj["row"]: array, shape = (S)
                Row idx of non-zero entries.
            adj["col"]: array, shape = (S)
                Col idx of non-zero entries.
            adj["all"]: array, shape = (S)
                Idx to pool over the entire adjacency.
            adj["tra"]: array, shape = (S)
                Idx to traspose matrix.
            adj["dia"]: array, shape = (b*N)
                Idx of diagonal elements.
            adj["dal"]: array, shape = (b*N)
                Idx to pool diagonal.
            All entries are properly shifted across the batch.
        b(int). Batch size.
        N(int). Number of particles.
        layer_id (int). Id of layer in network, for retrieving variables.
        is_last (bool). If is_last, pool output over columns.

    Returns:
        H_out (tensor). Shape = (S, q) or (b, N, q) if is_last.
    """
    def _pool(h, pool_idx, num_segs):
        """Pool based on indices.

        Given row idx, it corresponds to pooling over columns, given col idx it corresponds
        to pool over rows, etc...

        Args:
            h (tensor). Shape = (S, k), row-major order.
            pool_idx (tensor). Shape = (S) or (b*N).
            num_segs (int). Number of segments (number of unique indices).
        Return:
            tensor.
        """
        return tf.unsorted_segment_mean(h, pool_idx, num_segs)

    def _broadcast(h, broadcast_idx):
        """Broadcast based on indices.

        Given row idx, it corresponds to broadcast over columns,
        given col idx it corresponds to broadcast over rows, etc...
        Note: in the old implementation _pool and _broadcast were
        done together in pool_ShiftInv_graph_conv.

        Args:
            h (tensor). Pooled data.
            broadcast_idx (tensor). Shape = (S) or (b*N).
        Return:
            tensor.
        """
        return tf.gather_nd(h, tf.expand_dims(broadcast_idx, axis=1))

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

    # -------------------------
    # FIXME: this paragraph is a placeholder for weights and biases
    # This is just a placeholder for trainable weights. Just setting identity matrices as placeholders (so
    # input_channels = output_channels).
    # This should be replaced with 15 weight matrices of shape (H_in.shape[1], number of output channels).
    #W = tf.constant(np.array([np.eye(H_in.shape[1], dtype=np.float64) for _ in range(15)], dtype=np.float64))

    # This is just a placeholder for biases. Just setting it to one as placeholder.
    # This should be replaced with 2 bias matrices of shape (number of output channels).
    #B = tf.constant(np.ones(shape=(2, W.shape[2])))
    # -------------------------

    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # FIX:
    # S = sum_i( symmetrized_i ) for i = 0...b-1, where b = batch size.
    # Updated: H_in, adj, bN, layer_id, is_last=False
    #     H_in: (S, k_in)
    # Prev:    H_in, COO_feats, bN, layer_id, is_last=False
        # COO_feats (tensor): (3, c), of row, column, cube-wise indices respectively
        # COO_feats (tensor): (3, c), of row, column, cube-wise indices respectively
    #
    # Get layer vars
    # -------------------------
    #==== Data dims
    b, N = bN # batch_size, num_particles

    #==== Weights and Biases
    # W : (15, k_in, k_out)
    # B : (2, k_out)
    W, B = utils.get_ShiftInv_layer_vars(layer_id)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    out_shape = (H_in.shape[0], W.shape[-1]) # (S, k_out)
    H_all = []

    #==== 1. No pooling
    H_all.append(tf.matmul(H_in, W[0]))

    #==== 2. Transpose
    H2 = tf.gather(H_in, adj["tra"])
    H_all.append(tf.matmul(H2, W[1]))

    #==== 3. Diagonal
    Hd = tf.gather(H_in, adj["dia"])
    H_all.append(_broadcast_to_diag(tf.matmul(Hd, W[2]), adj["dia"], out_shape))

    #==== 4. Pool rows, broadcast to rows
    Hr = _pool(H_in, adj["col"], b * N)
    H_all.append(_broadcast(tf.matmul(Hr, W[3]), adj["col"]))

    #==== 5. Pool rows, broadcast to cols
    H_all.append(_broadcast(tf.matmul(Hr, W[4]), adj["row"]))

    #==== 6. Pool rows, broadcast to diag
    H_all.append(_broadcast_to_diag(tf.matmul(Hr, W[5]), adj["dia"], out_shape))

    #==== 7. Pool cols, broadcast to cols
    Hc = _pool(H_in, adj["row"], b * N)
    H_all.append(_broadcast(tf.matmul(Hc, W[6]), adj["row"]))

    #==== 8. Pool cols, broadcast to rows
    H_all.append(_broadcast(tf.matmul(Hc, W[7]), adj["col"]))

    #==== 9. Pool cols, broadcast to diag
    H_all.append(_broadcast_to_diag(tf.matmul(Hc, W[8]), adj["dia"], out_shape))

    #==== 10. Pool all, broadcast all
    Ha = _pool(H_in, adj["all"], b)
    H_all.append(_broadcast(tf.matmul(Ha, W[9]), adj["all"]))

    #==== 11. Pool all, broadcast diagonal
    Ha_broad = _broadcast(tf.matmul(Ha, W[10]), adj["dal"])
    H_all.append(_broadcast_to_diag(Ha_broad, adj["dia"], out_shape))

    #==== 12. Pool diagonal, broadcast all
    Hp = _pool(Hd, adj["dal"], b)
    H_all.append(_broadcast(tf.matmul(Hp, W[11]), adj["all"]))

    #==== 13. Pool diagonal, broadcast diagonal
    Hp_broad = _broadcast(tf.matmul(Hp, W[12]), adj["dal"])
    H_all.append(_broadcast_to_diag(Hp_broad, adj["dia"], out_shape))

    #==== 14. Broadcast diagonal to rows
    H_all.append(_broadcast(tf.matmul(Hd, W[13]), adj["col"]))

    #==== 15. Broadcast diagonal to cols
    H_all.append(_broadcast(tf.matmul(Hd, W[14]), adj["row"]))

    # Diagonal and off diagonal bias
    # For simplicity will have a bias applied to all and a
    #  separate one to diagonal only
    #  (which is equivalent to diagonal and off-diagonal)
    B_diag = _broadcast_to_diag(tf.broadcast_to(B[0], (b * N, B[0].shape[0])), adj["dia"], out_shape)
    B_all = B[1]

    # Output
    #------------------------
    H = tf.add_n(H_all) + B_diag + B_all
    if is_last:
        return tf.reshape(_pool(H, adj["row"], b * N), (b, N, -1))
    else:
        return H


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Shift invariant network
# ========================================
def network_func_ShiftInv(X_input_features, adj_map, num_layers, dims,
                          activation, redshift=None):
    # Input layer
    # -----------
    H = activation(ShiftInv_layer(X_input_features, adj_map, dims, 0))

    # Hidden layers
    # -------------
    for layer_idx in range(1, num_layers):
        is_last = layer_idx == num_layers - 1
        H = ShiftInv_layer(H, adj_map, dims, layer_idx, is_last=is_last)
        if not is_last:
            H = activation(H)
    return H

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Shift invariant model func
# ========================================
# get_input_features_ShiftInv_numpy(X_in, A, N, redshift)
def model_func_ShiftInv_symm(X_in_features, adj_map, model_specs, redshift=None):
    """ # Shiftinv model func for symmetrical adjacency indices

    Params
    ------
    X_in_features : (S, 9)

    adj_map : dict(tf.Tensor)
        dictionary mapping shift-inv op tags to corresponding adjacency indices
            # All entries are properly shifted across the batch.
            #==== Size (S*,) S = sum of symm indices for each i in batch
            # Not static!
            adj_map["row"]: Row idx of non-zero entries.
            adj_map["col"]: Col idx of non-zero entries.
            adj_map["all"]: Idx to pool over the entire adjacency.
            adj_map["tra"]: Idx to traspose matrix.
            #==== shape (b*N)
            adj_map["dia"]: Idx of diagonal elements.
            adj_map["dal"]: Idx to pool diagonal.

    model_specs : dict
        key & vals: var_scope, num_layers, activation, dims

    redshift : tensor

    Args:
        X_in (tensor): (b, N, 6)
        adj_map dict(tensor): (3, B*N*M), segment ids for rows, cols, all
        redshift (tensor): (b*N*M, 1) redshift broadcasted
    """
    # Get relevant model specs
    # ========================================
    var_scope  = model_specs.var_scope
    num_layers = model_specs.num_layers
    activation = model_specs.activation # default tf.nn.relu
    dims = model_specs.dims

    # Now a numpy preprocessing step
    # X Get graph inputs
    # ========================================
    #edges, nodes = get_input_features_ShiftInv(X_in, adj_map, dims)
    '''
    Args:
        X_in(array). (b, N, 6) coord and velocities.
        A(list). Batch of csr adjacency.
        N(int). Number of particles.
        redshift(float).

    Returns:
    #    ;;;;;                                                                   ;;;;;     #
    #    ;;;;;      ____   _____   _____   _____    __      _____   __  __       ;;;;;     #
    #  ..;;;;;..   |  _ \ |  _  \ /     \ |  __ \  |  |    |  ___| |  \/  |    ..;;;;;..   #
    #   ':::::'    |  __/ |     / |  |  | |  __ <  |  |__  |  ___| |      |     ':::::'    #
    #     ':`      |__|   |__|__\ \_____/ |_____/  |_____| |_____| |_|\/|_|       ':`      #
        array of shape (S, 9) or (S, 10) if redshift is not None.
        S is defined above in new_ShiftInv_layer.

    ## NOTE: cannot have array shape (S, 9) if S is not static int
    '''

    # Network forward
    # ========================================
    with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
        # ==== Split input
        X_in_loc, X_in_vel = X_in[...,:3], X_in[...,3:]
        # ==== Network output
        net_out = network_func_ShiftInv(edges, nodes, adj_map, num_layers,
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






#==============================================================================
#                       Symmeterized graph search
#==============================================================================

def get_symmetrized_idx(A):
    """Generates idx given a batch of sparse matrices (adjacencies will be symmetrized!).

    Args:
        A(list). list of csr adjacency matrices in a batch.
            Each adjacency should have 1 on diagonals.

    Returns:
        idx(dict): dict
            idx["row"]: array, shape = (S)
                Row idx of non-zero entries.
            idx["col"]: array, shape = (S)
                Col idx of non-zero entries.
            idx["all"]: array, shape = (S)
                Idx to pool over the entire adjacency.
            idx["tra"]: array, shape = (S)
                Idx to traspose matrix.
            idx["dia"]: array, shape = (b*N)
                Idx of diagonal elements.
            idx["dal"]: array, shape = (b*N)
                Idx to pool diagonal.
            All entries are properly shifted across the batch.
    """
    row, col, all_, tra, dia, dal = [], [], [], [], [], []
    for i, a in enumerate(A):
        a = a + a.transpose() # symmetrize
        N = a.shape[0]
        r, c = a.nonzero()
        row.extend(r + i * N)
        col.extend(c + i * N)
        all_.extend(np.zeros_like(r) + i)

        t = np.array([], dtype=np.int64)
        for j in range(a.shape[0]):
            t = np.append(t, np.where(c == j)[0])
        tra.extend(t + i * len(r))

        d = np.array(np.where(r == c)[0])
        dia.extend(d + i * len(r))
        dal.extend(np.zeros_like(d) + i)

    return {"row": row, "col": col, "all": all_, "tra": tra, "dia": dia, "dal": dal}
