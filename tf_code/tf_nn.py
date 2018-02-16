import os, code, sys, time
import numpy as np
import tensorflow as tf
#import chainer.functions as F
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import tf_utils as utils
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

''' TF nuggets:
 - tf.get_variable only returns an existing variable with name if it was
   created earlier by get_variable. It won't return a variable created using
   tf.Variable()
'''
#=============================================================================
''' IMPLEMENTATION LIST
 X velocity coefficient
 X save points
 - restore model
 - Graph model, kneighbors
 - Graph model, radius

'''
#=============================================================================

#=============================================================================
# layer ops
#=============================================================================
def left_mult(h, W):
    return tf.einsum('ijl,lq->ijq', h, W)

def linear_fwd(h_in, W, b=None):
    """ permutation equivariant linear transformation
    Args:
        h_in: external input, of shape (mb_size, n_P, k_in)
        W: layer weight, of shape (k_in, k_out)
        b: bias, of shape (k_out,)
    """
    mu = tf.reduce_mean(h_in, axis=1, keepdims=True)
    h = h_in - mu
    h_out = left_mult(h, W)
    if b is not None:
        h_out += b
    return h_out

def linear_layer(h, layer_idx):
    """ layer gets weights and returns linear transformation
    """
    W, B = utils.get_layer_vars(layer_idx)
    return linear_fwd(h, W, B)

def kgraph_select(h, adj, K):
    dims = tf.shape(h)
    mb = dims[0]; n  = dims[1]; d  = dims[2];
    rdim = [mb,n,K,d]
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    nn_graph = tf.reduce_mean(tf.reshape(tf.gather_nd(h, adj), rdim), axis=2)
    return nn_graph

def kgraph_layer(h, layer_idx, alist, K):
    """ layer gets weights and returns linear transformation
    """
    W, Wg, B = utils.get_layer_vars_graph(layer_idx)
    nn_graph = kgraph_select(h, alist, K)
    h_w = linear_fwd(h, W)
    h_g = linear_fwd(nn_graph, Wg)
    h_out = h_w + h_g + B
    return h_out

def graph_fwd(x_in, num_layers, alist, K=14, activation=tf.nn.relu):
    H = x_in
    for i in range(num_layers):
        H = kgraph_layer(H, i, alist, K)
        if i != num_layers - 1:
            H = activation(H)
    return H

def set_fwd(x_in, num_layers, activation=tf.nn.relu):
    H = x_in
    for i in range(num_layers):
        H = linear_layer(H, i)
        if i != num_layers - 1:
            H = activation(H)
    return H

def network_graph_fwd(x_in, num_layers, alist, activation=tf.nn.relu, add=True, vel_coeff=None, K=14):
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    h_out = graph_fwd(x_in, num_layers, alist, activation=activation, K=K)
    if add:
        x_coo = x_in[...,:3]
        h_out += x_coo
    if vel_coeff is not None:
        h_out += vel_coeff * x_in[...,3:]
    return h_out

def network_fwd(x_in, num_layers, *args, activation=tf.nn.relu, mtype_key=0, add=True, vel_coeff=None, **kwargs):
    if mtype_key == 0: # set
        h_out = set_fwd(x_in, num_layers, activation)
    else:
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        h_out = graph_fwd(x_in, num_layers, *args, activation=activation, **kwargs)
    if add:
        x_coo = x_in[...,:3]
        h_out += x_coo
    if vel_coeff is not None:
        h_out += vel_coeff * x_in[...,3:]
    return h_out

#=============================================================================
# graph ops
#=============================================================================
def alist_to_indexlist(alist):
    batch_size, N, K = alist.shape
    id1 = np.reshape(np.arange(batch_size),[batch_size,1])
    id1 = np.tile(id1,N*K).flatten()
    out = np.stack([id1,alist.flatten()], axis=1).astype(np.int32)
    return out
'''
def get_kneighbor_alist2(X_in, K=14, shell_fraction=0.1):
    batch_size, N, D = X_in.shape
    csr_list = get_csr_periodic_bc(X_in, K, shell_fraction=shell_fraction)
    adj_list = np.zeros((batch_size, N, K)).astype(np.int32)
    for i in range(batch_size):
        adj_list[i] = csr_list[i].indices.reshape(N, K)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    return alist_to_indexlist(adj_list)
'''

def get_kneighbor_alist(X_in, K=14, shell_fraction=0.1):
    batch_size, N, D = X_in.shape
    adj_list = get_pbc_adjacency_list(X_in, K, shell_fraction=shell_fraction)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    return alist_to_indexlist(adj_list)


#=============================================================================
# periodic boundary condition neighbor graph stuff
#=============================================================================
def _get_status(coordinate, L_box, dL):
    """
    Assign a status to each coordinate (of a particle position inside the box):
    1 if 0 < coord < dL, 2 if L- dL < coord < L, 0 otherwise
    PARAMS:
        coordinate(float)
    RETURNS:
        status(int). Either 0, 1, or 2
    """
    if coordinate < dL:
        return 1
    elif L_box - dL < coordinate < L_box:
        return 2
    else:
        return 0

def _get_clone(particle, k, s, L_box, dL):
    """
    Clone a particle otuside of the box.
    PARAMS:
        particle(np array). 6-dim particle position in phase space
        k(int). Index of dimension that needs to be projected outside of the box.
        s(int). Status, either 1 or 2. Determines where should be cloned.
    RETURNS:
        clone(np array). 6-dim cloned particle position in phase space.
    """
    clone = []
    for i in range(6):
        if i == k:
            if s == 1:
                clone.append(particle[i] + L_box)
            elif s == 2:
                clone.append(particle[i] - L_box)
        else:
            clone.append(particle[i])
    return np.array(clone)

'''
def get_csr_periodic_bc2(X_in, K, shell_fraction=0.1):
    """
    Map inner chunks to outer chunks
    cant black box this anymore
    NEED TO CLEAN THIS UP, at least var names
    """
    K = K
    mb_size, N, D = X_in.shape
    L_box = 16 if N == 16**3 else 32
    dL = L_box * shell_fraction
    box_size = (L_box, dL)
    #adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
    csr_list = []
    for i in range(mb_size):
        ids_map = {}  # For this batch will map new_id to old_id of cloned particles
        new_X = [part for part in X_in[i]]  # Start off with original cube
        for j in range(N):
            status = [_get_status(X_in[i, j, k], *box_size) for k in range(3)]
            if sum(status) == 0:  # Not in the shell --skip
                continue
            else:
                for k in range(3):
                    if status[k] > 0:
                        clone = _get_clone(X_in[i, j, :], k, status[k], *box_size)
                        new_X.append(clone)
                        ids_map.update({len(new_X) - 1: j})
                        for kp in range(k + 1, 3):
                            if status[kp] > 0:
                                bi_clone = _get_clone(clone, kp, status[kp], *box_size)
                                new_X.append(bi_clone)
                                ids_map.update({len(new_X) - 1: j})
                                for kpp in range(kp + 1, 3):
                                    if status[kpp] > 0:
                                        tri_clone = _get_clone(bi_clone, kpp, status[kpp], *box_size)
                                        new_X.append(tri_clone)
                                        ids_map.update({len(new_X) - 1: j})
        new_X = np.array(new_X)
        # try LIL matrix
        #graph = rad_graph(new_X[:,:3], K, include_self=True).tolil()[:N,:]
        graph = kneighbors_graph(new_X[:,:3], K, include_self=True).tolil()[:N,:]
        for j in range(N):
            graph.rows[j] = [r if r < N else ids_map[r] for r in graph.rows[j]]
        graph_csr = graph[:,:N].tocsr()
        csr_list.append(graph_csr)#, np.diff(graph_csr.indptr)]
    return csr_list
'''

def get_pbc_adjacency_list(X_in, K, shell_fraction=0.1):
    """
    Map inner chunks to outer chunks
    cant black box this anymore
    NEED TO CLEAN THIS UP, at least var names
    """
    K = K
    mb_size, N, D = X_in.shape
    L_box = 16 if N == 16**3 else 32
    dL = L_box * shell_fraction
    box_size = (L_box, dL)
    adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
    for i in range(mb_size):
        ids_map = {}  # For this batch will map new_id to old_id of cloned particles
        new_X = [part for part in X_in[i]]  # Start off with original cube
        for j in range(N):
            status = [_get_status(X_in[i, j, k], *box_size) for k in range(3)]
            if sum(status) == 0:  # Not in the shell --skip
                continue
            else:
                for k in range(3):
                    if status[k] > 0:
                        clone = _get_clone(X_in[i, j, :], k, status[k], *box_size)
                        new_X.append(clone)
                        ids_map.update({len(new_X) - 1: j})
                        for kp in range(k + 1, 3):
                            if status[kp] > 0:
                                bi_clone = _get_clone(clone, kp, status[kp], *box_size)
                                new_X.append(bi_clone)
                                ids_map.update({len(new_X) - 1: j})
                                for kpp in range(kp + 1, 3):
                                    if status[kpp] > 0:
                                        tri_clone = _get_clone(bi_clone, kpp, status[kpp], *box_size)
                                        new_X.append(tri_clone)
                                        ids_map.update({len(new_X) - 1: j})
        new_X = np.array(new_X)
        graph_idx = kneighbors_graph(new_X[:, :3], K, include_self=True).indices
        graph_idx = graph_idx.reshape([-1, K])[:N, :]  # Only care about original box
        # Remap outbox neighbors to original ids
        for j in range(N):
            for k in range(K):
                if graph_idx[j, k] > N - 1:  # If outside of the box
                    graph_idx[j, k] = ids_map.get(graph_idx[j, k])
        graph_idx = graph_idx #+ (N * i)  # offset idx for batches
        adj_list[i] = graph_idx
    return adj_list

#=============================================================================
# periodic boundary conditions, loss
#=============================================================================
def get_readout(h_out):
    gt_one  = (tf.sign(h_out - 1) + 1) / 2
    ls_zero = -(tf.sign(h_out) - 1) / 2
    rest = 1 - gt_one - ls_zero
    readout = rest*h_out + gt_one*(h_out - 1) + ls_zero*(1 + h_out)
    return readout

def periodic_boundary_dist(readout, x_truth):
    x_truth_coo = x_truth[...,:3]
    dist = tf.minimum(tf.square(readout - x_truth_coo), tf.square(readout - (1 + x_truth_coo)))
    dist = tf.minimum(dist, tf.square((1 + readout) - x_truth_coo))
    return dist

def pbc_loss(readout, x_truth):
    pbc_dist = periodic_boundary_dist(readout, x_truth)
    pbc_error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1), name='loss')
    return pbc_error