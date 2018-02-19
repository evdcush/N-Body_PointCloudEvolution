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
 - EXTREMELY sensitive to random seed? I made the seed none for weight init
   and the error per iteration went from 770000.0 to 0.22 ??
 -
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
# LAYER OPS
#=============================================================================
def left_mult(h, W):
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

def set_layer(h, layer_idx, *args):
    """ layer gets weights and returns linear transformation
    """
    W, B = utils.get_layer_vars(layer_idx)
    return linear(h, W, B)

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
    #W, Wg, B = utils.get_layer_vars_graph(layer_idx)
    W, Wg = utils.get_layer_vars_graph(layer_idx)
    nn_graph = kgraph_select(h, alist, K)
    h_w = linear(h, W)
    h_g = linear(nn_graph, Wg)
    h_out = h_w + h_g #+ B
    return h_out

def network_fwd(x_in, num_layers, *args, activation=tf.nn.relu):
    layer = kgraph_layer if len(args) > 0 else set_layer
    H = x_in
    for i in range(num_layers):
        H = layer(H, i, *args)
        if i != num_layers - 1:
            H = activation(H)
    return H

#=============================================================================
def model_fwd(x_in, num_layers, *args, activation=tf.nn.relu, add=True, vel_coeff=None):
    h_out = network_fwd(x_in, num_layers, *args)
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
'''
def get_kneighbor_alist(X_in, K=14, shell_fraction=0.1):
    batch_size, N, D = X_in.shape
    adj_list = get_pbc_adjacency_list(X_in, K, shell_fraction=shell_fraction)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    return alist_to_indexlist(adj_list)
'''
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
# periodic boundary condition neighbor graph stuff
#=============================================================================
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

def get_pbc_kneighbors(X, K, boundary_threshold=0.1):
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

def periodic_boundary_dist(readout, x_truth):
    x_truth_coo = x_truth[...,:3]
    dist = tf.minimum(tf.square(readout - x_truth_coo), tf.square(readout - (1 + x_truth_coo)))
    dist = tf.minimum(dist, tf.square((1 + readout) - x_truth_coo))
    return dist

def pbc_loss(readout, x_truth):
    pbc_dist = periodic_boundary_dist(readout, x_truth)
    pbc_error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1), name='loss')
    return pbc_error