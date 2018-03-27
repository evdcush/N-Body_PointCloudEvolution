import os, code, sys, time

import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import tensorflow as tf

import utils
from utils import VAR_SCOPE, VAR_SCOPE_MULTI
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

#=============================================================================
# LAYER OPS
#=============================================================================
def left_mult(h, W):
    """ batch matmul for set-based data
    """
    return tf.einsum('ijl,lq->ijq', h, W)

def linear(h_in, W, b=None):
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

def set_layer(h, layer_idx, var_scope, *args):
    """ Set layer
    """
    W, B = utils.get_layer_vars(layer_idx, var_scope=var_scope)
    return linear(h, W, B)

#=============================================================================
# graph
def no_tiled_kgraph_conv(h, adj):
    """ Uses gather instead of gather_nd. Idea being you can keep the adj
    list in it's "natural" shape, and you don't need to pass K as an arg anymore,
    but tf complains.

    Works, but not using due to tf dense gradient warning:
    "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "

    Args:
        h: data tensor, (mb_size, N, k_in)
        adj: adjacency index list, for gather (mb_size, N, K)
    """
    dims = tf.shape(h)
    mb = dims[0]; n  = dims[1]; d  = dims[2];
    K = tf.shape(adj)[-1]
    alist = tf.reshape(adj, [-1])
    h_flat = tf.reshape(h, [-1, d])

    rdim = [mb,n,K,d]
    nn_graph = tf.reduce_mean(tf.reshape(tf.gather(h_flat, alist), rdim), axis=2)
    return nn_graph

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
    Graph layer has two sets of weights for data:
      W: for connections to all other particles
     Wg: for connections to nearest neighbor particles
    Unlike the set layer, no biases are used. Have tried h_w + h_g + B, but
    error was worse.
    Args:
        h: data tensor, (mb_size, N, k_in)
        layer_idx (int): layer index for params
        var_scope (str): variable scope for get variables from graph
        alist: adjacency index list tensor (*, 2), of tf.int32
        K (int): number of nearest neighbors in KNN
    RETURNS: (mb_size, N, k_out)
    """
    W, Wg = utils.get_graph_layer_vars(layer_idx, var_scope=var_scope)
    nn_graph = kgraph_conv(h, alist, K)
    h_w = linear(h, W)
    h_g = linear(nn_graph, Wg)
    h_out = h_w + h_g
    return h_out

#=============================================================================
def network_fwd(x_in, num_layers, var_scope, *args, activation=tf.nn.relu):
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    layer = kgraph_layer if len(args) != 0 else set_layer
    H = x_in
    for i in range(num_layers):
        H = layer(H, i, var_scope, *args)
        if i != num_layers - 1:
            H = activation(H)
    return H

#=============================================================================
def model_fwd(x_in, num_layers, *args, activation=tf.nn.relu, add=True, vel_coeff=False, var_scope=VAR_SCOPE):
    h_out = network_fwd(x_in, num_layers, var_scope, *args)
    if add:
        if x_in.shape[-1] == h_out.shape[-1]: # when predicting velocity
            #h_out += x_in
            x_coo =  x_in[...,:3]
            x_vel =  x_in[...,3:]
            h_coo = h_out[...,:3]
            h_vel = h_out[...,3:]
            h_coo += x_coo
            if vel_coeff:
                vel_co = utils.get_vel_coeff(var_scope)
                v_scaled = vel_co * x_vel
                h_coo += v_scaled
            h_out = tf.concat((h_coo, h_vel), axis=-1)
        else:
            h_out += x_in[...,:3]
    return h_out

#=============================================================================
# multi stuff, new data
#=============================================================================
def zuni_model_fwd(x_in, num_layers, *args, activation=tf.nn.relu, add=True, vel_coeff=False, var_scope=VAR_SCOPE):
    h_out = network_fwd(x_in, num_layers, var_scope, *args)
    if add:
        '''
        # x splits
        x_coo =  x_in[...,:3]
        x_vel =  x_in[...,3:-1]
        # h splits
        h_coo = h_out[...,:3]
        h_vel = h_out[...,3:]
        # add
        h_coo += x_coo
        h_out = tf.concat((h_coo, h_vel), axis=-1)
        '''
        h_out += x_in[...,:-1]
    return h_out


def multi_fwd_sampling(x_in, num_layers, adj, K, sampling_probs, var_scope=VAR_SCOPE):
    num_rs_layers = x_in.get_shape().as_list()[0] - 1
    concat_rs = lambda x, z: tf.concat((x, z), axis=-1)
    fwd = lambda x, a: get_readout_vel(zuni_model_fwd(x, num_layers, a, K, var_scope=var_scope))
    h = fwd(x_in[0], adj[0])
    for i in range(1, num_rs_layers):
        h_in = tf.where(sampling_probs[i], concat_rs(h, x_in[i, :, :, -1:]), x_in[i])
        h = fwd(h_in, adj[i])
    return h

def multi_fwd_sampling_sumError(x_in, num_layers, adj, K, sampling_probs, var_scope=VAR_SCOPE):
    num_rs_layers = x_in.get_shape().as_list()[0] - 1
    concat_rs = lambda x, z: tf.concat((x, z), axis=-1)
    fwd = lambda x, a: get_readout_vel(zuni_model_fwd(x, num_layers, a, K, var_scope=var_scope))
    h = fwd(x_in[0], adj[0])
    error_sum = pbc_loss(h, x_in[1,:,:,:-1])
    for i in range(1, num_rs_layers):
        h_in = tf.where(sampling_probs[i], concat_rs(h, x_in[i, :, :, -1:]), x_in[i])
        h = fwd(h_in, adj[i])
        error_sum += pbc_loss(h, x_in[i+1,:,:,:-1])
    return h, error_sum


def multi_model_fwd_sampling(x_in, num_layers, adj, K, sampling_probs, var_scopes):
    concat_rs = lambda x, z: tf.concat((x, z), axis=-1)
    fwd = lambda x, a, v: get_readout_vel(zuni_model_fwd(x, num_layers, a, K, var_scope=v))
    h = fwd(x_in[0], adj[0], var_scopes[0])
    for i, vscope in enumerate(var_scopes[1:]):
        h_in = tf.where(sampling_probs[i], concat_rs(h, x_in[i, :, :, -1:]), x_in[i])
        h = fwd(h_in, adj[i], vscope)
    return h

def multi_model_fwd_val(x_in, num_layers, adj_fn, K, var_scopes):
    concat_rs = lambda x, z: tf.concat((x, z), axis=-1)
    fwd = lambda x, a, v: get_readout_vel(zuni_model_fwd(x, num_layers, a, K, var_scope=v))
    adj = tf.py_func(adj_fn, [x_in[0]], tf.int32)
    h = fwd(x_in[0], adj, var_scopes[0])
    for i, vscope in enumerate(var_scopes[1:]):
        h_in = concat_rs(h, x_in[i,:,:,-1:])
        adj = tf.py_func(adj_fn, [h_in], tf.int32)
        h = fwd(h_in, adj, vscope)
    return h

#=============================================================================
# multi fns for single step trained models

def zuni_multi_single_fwd(x_rs, num_layers, rs_adj_list, K, var_scopes):
    concat_rs = lambda x, z: tf.concat((x, z), axis=-1)
    fwd = lambda x, a, v: get_readout_vel(zuni_model_fwd(x, num_layers, a, K, var_scope=v))
    h = fwd(x_rs[0], rs_adj_list[0], var_scopes[0])
    for i, vscope in enumerate(var_scopes[1:]):
        h_in = concat_rs(h, x_rs[i,:,:,-1:])
        h = fwd(h_in, rs_adj_list[i], vscope)
    return h

def zuni_multi_single_fwd_val(x_rs, num_layers, adj_fn, K, var_scopes):
    # helpers
    concat_rs = lambda x, z: tf.concat((x, z), axis=-1)
    fwd = lambda x, a, v: get_readout_vel(zuni_model_fwd(x, num_layers, a, K, var_scope=v))
    adj = tf.py_func(adj_fn, [x_rs[0]], tf.int32)

    h = fwd(x_rs[0], adj, var_scopes[0])
    for i, vscope in enumerate(var_scopes[1:]):
        h_in = concat_rs(h, x_rs[i,:,:,-1:])
        adj = tf.py_func(adj_fn, [h_in], tf.int32)
        h = fwd(h_in, adj, vscope)
    return h

def zuni_multi_single_fwd_val_all(x_rs, num_layers, adj_fn, K, var_scopes):
    preds = []
    # helpers
    concat_rs = lambda x, z: tf.concat((x, z), axis=-1)
    fwd = lambda x, a, v: get_readout_vel(zuni_model_fwd(x, num_layers, a, K, var_scope=v))
    adj = tf.py_func(adj_fn, [x_rs[0]], tf.int32)

    h = fwd(x_rs[0], adj, var_scopes[0])
    preds.append(h)
    for i, vscope in enumerate(var_scopes[1:]):
        h_in = concat_rs(h, x_rs[i,:,:,-1:])
        adj = tf.py_func(adj_fn, [h_in], tf.int32)
        h = fwd(h_in, adj, vscope)
        preds.append(h)
    return preds


#=============================================================================
# graph ops
#=============================================================================
def alist_to_indexlist(alist):
    """ reshapes adjacency list for tf.gather_nd
    """
    batch_size, N, K = alist.shape
    id1 = np.reshape(np.arange(batch_size),[batch_size,1])
    id1 = np.tile(id1,N*K).flatten()
    out = np.stack([id1,alist.flatten()], axis=1).astype(np.int32)
    return out

def get_kneighbor_alist(X_in, K=14):
    """ search for K nneighbors, and return offsetted indices in adjacency list

    Args:
        X_in (numpy ndarray): input data of shape (mb_size, N, 6)
    """
    mb_size, N, D = X_in.shape
    adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
    for i in range(mb_size):
        # this returns indices of the nn
        graph_idx = kneighbors_graph(X_in[i, :, :3], K, include_self=True).indices
        graph_idx = graph_idx.reshape([N, K]) #+ (N * i)  # offset idx for batches
        adj_list[i] = graph_idx
    return adj_list


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
    idx_list = np.array([], dtype=np.int32) # keep in mind idx need to be offset by N

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
            #kgraph[k_idx] = idx_map[k_idx - N]
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
    gt_one  = (tf.sign(h_out - 1) + 1) / 2
    ls_zero = -(tf.sign(h_out) - 1) / 2
    rest = 1 - gt_one - ls_zero
    readout = rest*h_out + gt_one*(h_out - 1) + ls_zero*(1 + h_out)
    return readout

def get_readout_vel(h_out):
    """ For when the network also predicts velocity
    velocities remain unchanged
    """
    h_out_coo = h_out[...,:3]
    h_out_vel = h_out[...,3:]
    gt_one  = (tf.sign(h_out_coo - 1) + 1) / 2
    ls_zero = -(tf.sign(h_out_coo) - 1) / 2
    rest = 1 - gt_one - ls_zero
    readout_coo = rest*h_out_coo + gt_one*(h_out_coo - 1) + ls_zero*(1 + h_out_coo)
    readout = tf.concat([readout_coo, h_out_vel], -1)
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
    #dist = tf.minimum(tf.square(readout - x_truth_coo), tf.square(readout - (1 + x_truth_coo)))
    #dist = tf.minimum(dist, tf.square((1 + readout) - x_truth_coo))
    return dist

def periodic_boundary_dist_vel(readout, x_truth):
    """ minimum distances between particles given periodic boundary conditions
    Normal squared distance would penalize for large difference between particles
    on opposite sides of cube
    """
    # pbc coo dist
    dist_coo = periodic_boundary_dist(readout[...,:3], x_truth[...,:3]) # (mb_size, N, 3)
    # dist vel
    dist_vel = tf.squared_difference(readout[...,3:], x_truth[...,3:])
    # combined dist
    dist = tf.concat([dist_coo, dist_vel], -1)
    return dist

def pbc_loss(readout, x_truth):
    """ MSE (coo only) with periodic boundary conditions
    Args:
        readout (tensor): model prediction which has been remapped to inner cube
        x_truth (tensor): ground truth (mb_size, N, 6)
    """
    pbc_dist = periodic_boundary_dist(readout, x_truth) # (mb_size, N, 3)
    pbc_error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1), name='loss')
    return pbc_error

def pbc_loss_vel(readout, x_truth):
    """ MSE over full dims with periodic boundary conditions
    Args:
        readout (tensor): model prediction which has been remapped to inner cube
        x_truth (tensor): ground truth (mb_size, N, 6)
    """
    pbc_dist  = periodic_boundary_dist(readout, x_truth)
    dist_vel = tf.squared_difference(readout[...,3:], x_truth[...,3:])

    error_coo = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1), name='loss')
    error_vel = tf.reduce_mean(tf.reduce_sum(dist_vel, axis=-1), name='loss')
    return error_coo + error_vel

def _pbc_loss_vel(readout, x_truth):
    """ MSE over full dims with periodic boundary conditions
    Args:
        readout (tensor): model prediction which has been remapped to inner cube
        x_truth (tensor): ground truth (mb_size, N, 6)
    """
    pbc_dist  = periodic_boundary_dist_vel(readout, x_truth)
    pbc_error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1), name='loss')
    return pbc_error



#=============================================================================
# NUMPY ERROR, for scaling losses in multi-model
#=============================================================================
def np_periodic_boundary_dist(readout_full, x_truth):
    """ minimum distances between particles given periodic boundary conditions
    Normal squared distance would penalize for large difference between particles
    on opposite sides of cube
    """
    readout = readout_full[...,:3]
    x_truth_coo = x_truth[...,:3]
    d1 = np.square(readout - x_truth_coo)
    d2 = np.square(readout - (1 + x_truth_coo))
    d3 = np.square((1 + readout) -  x_truth_coo)
    dist = np.minimum(np.minimum(d1, d2), d3)
    return dist

def np_periodic_boundary_dist_vel(readout, x_truth):
    """ minimum distances between particles given periodic boundary conditions
    Normal squared distance would penalize for large difference between particles
    on opposite sides of cube
    """
    # pbc coo dist
    dist_coo = np_periodic_boundary_dist(readout[...,:3], x_truth[...,:3]) # (mb_size, N, 3)
    # dist vel
    dist_vel = np.square(readout[...,3:] - x_truth[...,3:])
    # combined dist
    dist = np.concatenate([dist_coo, dist_vel], -1)
    return dist

def np_pbc_loss_vel(readout, x_truth):
    """ numpy ops for scaling losses in multi
    """
    pbc_dist  = np_periodic_boundary_dist_vel(readout, x_truth)
    pbc_error = np.mean(np.sum(pbc_dist, axis=-1))
    return pbc_error

def np_pbc_loss(readout, x_truth):
    """ numpy ops for scaling losses in multi
    """
    pbc_dist  = np_periodic_boundary_dist(readout, x_truth)
    pbc_error = np.mean(np.sum(pbc_dist, axis=-1))
    return pbc_error

def error_scales(x_in):
    """ x_in shape (num_rs, mb_size, N, 6)
    OUT: (num_rs,)
    """
    num_rs = x_in.shape[0]
    scale_weights = np.zeros((num_rs-1)).astype(np.float32)
    for i in range(num_rs - 1):
        #scale_weights[i] = np_pbc_loss_vel(x_in[i], x_in[i+1])
        scale_weights[i] = np_pbc_loss(x_in[i], x_in[i+1])
    return 1 / scale_weights