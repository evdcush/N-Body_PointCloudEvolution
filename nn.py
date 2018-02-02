import os, sys, code

import cupy
import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

import chainer
import chainer.links as L
import chainer.functions as F

#=============================================================================
# graph ops
#=============================================================================
''' # for reference
def adjacency_list_tf(X_in,k):
    shape_in = X_in.shape
    X_out = np.zeros([shape_in[0],shape_in[1],k],dtype=np.int32)
    for b in range(shape_in[0]):
        # this returns indices of the nn
        X_out[b] = kneighbors_graph(X_in[b,:,:3],k,include_self=True).indices.reshape([shape_in[1],k])
        #print(X_out[b].shape)
    return X_out
'''

''' # WORK IN PROGRESS, currently no need for interface, since densityNN too inefficient for use
class GraphNN():
    """ Interface for knn and density-based neighbor graphs
    """
    def __init__(self, X_in, k):
        self.k = k
        neighbor_metric = DensityNN if isinstance(k, float) else KNN
        self.graph = neighbor_metric(X_in, k)

    def __call__(self, X):
        return self.graph(X)


class DensityNN():
    def __init__(self, X_in, rad):
        self.k = rad
        self.adjacency_list = self.get_adjacency_list(X_in)
        
    def get_adjacency_list(self, X_in):
        #how to deal with variable size neighborhood?
        #gravity clumps will have large neighborhood, while isolated points will have small (possibly a single point)
        
        rad = self.k
        mb_size, N, D = X_in.shape
        #X_out = np.zeros([mb_size, N, N],dtype=np.bool)
        X_out = np.zeros([mb_size, N, N], dtype=np.float32)
        for b in range(mb_size):
            graph_idx = radius_neighbors_graph(X_in[b,:,:3], rad, include_self=True)
            gidx = graph_idx.toarray().astype(np.float32)
            X_out[b] = gidx
        X_out = chainer.Variable(cuda.to_gpu(X_out))
        X_out = F.scale(X_out, 1/F.sum(X_out, axis=-1),axis=0)
        return X_out

    def __call__(self, x):
        # NEED TO DO SPARSE OPS HERE
        mb_size, N, D = x.shape
        graph = F.batch_matmul(self.adjacency_list, x)
        return graph
'''
'''
class KNN():
    def __init__(self, X_in, K):
        self.K = K
        self.adjacency_list = self.get_adjacency_list(X_in)
        
    def get_adjacency_list(self, X_in):
        """ search for K nneighbors, and return offsetted indices in adjacency list
        
        Args:
            X_in (numpy ndarray): input data of shape (mb_size, N, 6)
        """
        K = self.K
        mb_size, N, D = X_in.shape
        adj_list = np.zeros([mb_size, N, K],dtype=np.int32)
        for i in range(mb_size):
            # this returns indices of the nn
            graph_idx = kneighbors_graph(X_in[i,:,:3], K, include_self=True).indices
            graph_idx = graph_idx.reshape([N, K]) + (N * i) # offset idx for batches
            adj_list[i] = graph_idx
        return adj_list

    def __call__(self, x):
        alist = np.copy(self.adjacency_list) # loss worse without copying
        alist = alist.flatten()
        mb_size, N, D = x.shape
        xr = F.reshape(x, (-1,D))
        graph = F.reshape(F.get_item(xr, alist), (mb_size, N, self.K, D))
        return F.mean(graph, axis=2)
'''

class KNN(): # Daniele's periodic bounding knn
    def __init__(self, X_in, K, L_box):
        self.K = K
        self.L_box = L_box  # Box size

        # Change shell_fraction for tuning the thickness of the shell to be replicated, must be in range 0-1
        # 0: no replication, 1: entire box is replicated to the 26 neighbouring boxes
        self.shell_fraction = 0.1
        self.dL = self.L_box * self.shell_fraction

        # self.adjacency_list = self.get_adjacency_list(X_in)
        self.adjacency_list = self.get_adjacency_list_periodic_bc(X_in)

    def get_adjacency_list(self, X_in):
        """ search for K nneighbors, and return offsetted indices in adjacency list

        Args:
            X_in (numpy ndarray): input data of shape (mb_size, N, 6)
        """
        K = self.K
        mb_size, N, D = X_in.shape
        adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
        for i in range(mb_size):
            # this returns indices of the nn
            graph_idx = kneighbors_graph(X_in[i, :, :3], K, include_self=True).indices
            graph_idx = graph_idx.reshape([N, K]) + (N * i)  # offset idx for batches
            adj_list[i] = graph_idx
        return adj_list

    def get_adjacency_list_periodic_bc(self, X_in):
        """
        Same as get_adjacency_list but includes periodic boundary conditions.

        PARAMS:
            X_in(numpy ndarray). Input data of shape (mb_size, N, 6).

        RETURNS:
            adj_list(numpy ndarray). Adjacency list of shape (mb_size, N, K)
        """
        K = self.K
        mb_size, N, D = X_in.shape
        adj_list = np.zeros([mb_size, N, K], dtype=np.int32)
        for i in range(mb_size):
            ids_map = {}  # For this batch will map new_id to old_id of cloned particles
            new_X = [part for part in X_in[i]]  # Start off with original cube
            for j in range(N):
                if self._is_in_inbox_shell(X_in[i, j, :3]):  # Consider only inbox particles in a shell close to the boundary
                    # Map this particle to the 26 neighbouring boxes, but keep only clones that leave in the
                    # corresponding outside shell close to the boundary.
                    for x0 in [X_in[i, j, 0], X_in[i, j, 0] + self.L_box, X_in[i, j, 0] - self.L_box]:
                        for x1 in [X_in[i, j, 1], X_in[i, j, 1] + self.L_box, X_in[i, j, 1] - self.L_box]:
                            for x2 in [X_in[i, j, 2], X_in[i, j, 2] + self.L_box, X_in[i, j, 2] - self.L_box]:
                                # Check if it's in the outside shell and remove mapping to itself
                                if self._is_in_outbox_shell([x0, x1, x2]) and not self._is_in_box([x0, x1, x2]):
                                    new_X.append(np.array([x0, x1, x2, X_in[i, j, 3], X_in[i, j, 4], X_in[i, j, 5]]))
                                    ids_map.update({len(new_X) - 1: j}) # Update map {new_id: old_id}

            new_X = np.array(new_X)
            graph_idx = kneighbors_graph(new_X[:, :3], K, include_self=True).indices
            graph_idx = graph_idx.reshape([-1, K])[:N, :]  # Only care about original box
            # Remap outbox neighbors to original ids
            for j in range(N):
                for k in range(K):
                    if graph_idx[j, k] > N - 1:  # If outside of the box
                        graph_idx[j, k] = ids_map.get(graph_idx[j, k])

            #print graph_idx.shape

            graph_idx = graph_idx + (N * i)  # offset idx for batches
            adj_list[i] = graph_idx

        return adj_list

    # Bunch of helper methods
    def _is_in_inbox_shell(self, x):
        """
        x must be a particle INSIDE the original box, then returns true if it's in a cubic shell of thickness dL
        from the boundary of the box

        PARAMS:
            x(list or numpy array). Coordinates.

        RETURNS:
            is_in_shell(bool).
        """
        is_in_shell = False
        for i in range(3):
            if x[i] < self.dL or x[i] > self.L_box - self.dL:
                is_in_shell = True
                break
        return is_in_shell

    def _is_in_outbox_shell(self, x):
        """
        x must be a particle OUTSIDE the original box, then returns true if it's in a cubic shell of thickness dL
        from the boundary of the box

        PARAMS:
            x(list or numpy array). Coordinates.

        RETURNS:
            is_in_shell(bool).
        """
        is_in_shell = True
        for i in range(3):
            if x[i] > self.L_box + self.dL or x[i] < -self.dL:
                is_in_shell = False
                break
        return is_in_shell

    def _is_in_box(self, x):
        """
        x can be either outside or inside the original box. Returns true if it is inside the box.

        PARAMS:
            x(list or numpy array). Coordinates.

        RETURNS:
            is_in_box(bool).
        """
        is_in_box = True
        for i in range(3):
            if x[i] > self.L_box or x[i] < 0:
                is_in_box = False
                break
        return is_in_box

    def __call__(self, x):
        alist = np.copy(self.adjacency_list) # loss worse without copying
        alist = alist.flatten()
        mb_size, N, D = x.shape
        xr = F.reshape(x, (-1,D))
        graph = F.reshape(F.get_item(xr, alist), (mb_size, N, self.K, D))
        return F.mean(graph, axis=2)

#=============================================================================
# network layers
#=============================================================================
class SetLinear(chainer.Chain):
    """ Permutation equivariant linear layer for set data

    Args:
        kdim tuple(int): channel size (k_in, k_out)
        nobias (bool): if True, no bias weights used
    """
    def __init__(self, kdim, nobias=False):
        self.kdim = kdim
        super(SetLinear, self).__init__(
            lin = L.Linear(kdim[0], kdim[1], nobias=nobias),
            )

    def __call__(self, x, add=False):
        mb_size, N, D = x.shape
        k_in, k_out  = self.kdim
        x_mean = F.broadcast_to(F.mean(x, axis=1, keepdims=True), x.shape)
        x_r = F.reshape(x - x_mean, (mb_size*N, k_in))
        x_out = self.lin(x_r)
        if add and k_in == k_out:
            x_out += x_r
        x_out = F.reshape(x_out, (mb_size,N,k_out))
        return x_out

#=============================================================================
class SetLayer(chainer.Chain):
    """ Set layer, interface to SetLinear
    
    Args:
        kdim: channel size tuple (k_in, k_out)
        nobias: if True, no bias weights used
    """
    def __init__(self, kdim, nobias=False):
        self.kdim = kdim
        super(SetLayer, self).__init__(
            input_linear = SetLinear(kdim, nobias=nobias),
            )
        
    def __call__(self, x_in, add=True):
        x_out = self.input_linear(x_in, add=add)
        return x_out

#=============================================================================
class GraphLayer(chainer.Chain):
    """ Graph layer
    Consists of two sets of weights, one for the data input, the other for the 
    neighborhood graph
    
    Args:
        kdim: channel size tuple (k_in, k_out)
        nobias: if True, no bias weights used
    """
    def __init__(self, kdim, nobias=True):
        self.kdim = kdim
        super(GraphLayer, self).__init__(
            input_linear = SetLinear(kdim, nobias=nobias),
            graph_linear = SetLinear(kdim, nobias=nobias),
            )
        
    def __call__(self, x_in, graphNN, add=True):
        #graphNN = graph_arg[0]
        x_out     = self.input_linear(x_in, add=False)
        graph_out = self.graph_linear(graphNN(x_in), add=False)
        x_out += graph_out
        if add and x_in.shape == x_out.shape:
            x_out += x_in
        return x_out

#=============================================================================
# Loss related ops
'''
Need cleanup here really bad
Currently, only functions used are get_bounded_MSE and mean_squared_error_full
Wait until you have fully refactored train script and utils, and current models are finished training
'''
#=============================================================================

def mean_squared_error_full(x_hat, x_true):
    return F.mean(F.sum(F.squared_difference(x_hat[...,:3], x_true[...,:3]), axis=-1))

def mean_squared_error(x_hat, x_true, boundary=(0.095, 1-0.095)):
    if boundary is None:
        return get_min_readout_MSE(x_hat, x_true)
    else:
        return get_bounded_MSE(x_hat, x_true, boundary)

def get_readout(x_hat):
    readout = x_hat[...,:3]
    gt_one  = (F.sign(readout - 1) + 1) // 2
    ls_zero = -(F.sign(readout) - 1) // 2
    rest = 1 - gt_one - ls_zero
    readout_xhat = rest*readout + gt_one*(readout-1) + ls_zero*(1-readout)
    return readout_xhat

def get_min_readout_MSE(x_hat, x_true):
    '''x_hat needs to be bounded'''
    readout = get_readout(x_hat)
    x_true_loc = x_true[...,:3]
    dist = F.minimum(F.square(readout - x_true_loc), F.square(readout - (1 + x_true_loc)))
    dist = F.minimum(dist, F.square((1 + readout) - x_true_loc))
    mse = F.mean(F.sum(dist, axis=-1))
    return mse


def get_bounded(x, bound):
    # need to select xp based on x array module
    xp = chainer.cuda.get_array_module(x)
    lower, upper = bound
    return xp.all(xp.logical_and(lower<x.data, x.data<upper),axis=-1)

def get_bounded_MSE(x_hat, x_true, boundary):
    x_hat_loc  = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bidx = get_bounded(x_true_loc, boundary)
    bhat  = F.get_item(x_hat_loc, bidx)
    btrue = F.get_item(x_true_loc, bidx)
    return F.mean(F.sum(F.squared_difference(bhat, btrue), axis=-1))

def get_bounded_squared_error(x_hat, x_true, boundary):
    x_hat_loc  = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bidx = get_bounded(x_true_loc, boundary)
    bhat  = F.get_item(x_hat_loc, bidx)
    btrue = F.get_item(x_true_loc, bidx)
    return F.squared_difference(bhat, btrue)

def get_bounded_MSE_vel(x_hat, x_true, boundary):
    #x_hat_loc  = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bidx = get_bounded(x_true_loc, boundary)
    bhat  = F.get_item(x_hat, bidx)
    btrue = F.get_item(x_true, bidx)
    mse_loc = F.mean(F.sum(F.squared_difference(bhat[...,:3], btrue[...,:3]), axis=-1))
    mse_vel = F.mean(F.sum(F.squared_difference(bhat[...,3:], btrue[...,3:]), axis=-1))
    return mse_loc*mse_vel, mse_loc, mse_vel


def get_combined_MSE(x_input, x_hat, x_true, boundary): # EXPERIMENTAL
    x_input_loc  = x_input[...,:3]
    x_hat_loc = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bidx = get_bounded(x_true_loc, boundary)
    binput = F.get_item(x_input_loc, bidx)
    bhat   = F.get_item(x_hat_loc, bidx)
    btrue  = F.get_item(x_true_loc, bidx)

    dist_in_true  = F.sum(F.squared_difference(binput, btrue), axis=-1)
    dist_in_hat   = F.sum(F.squared_difference(binput,  bhat), axis=-1)
    dist_hat_true = F.sum(F.squared_difference(bhat,   btrue), axis=-1)
    input_dist = F.squared_difference(dist_in_true, dist_in_hat)
    combined = F.mean(input_dist * dist_hat_true)
    normal = F.mean(dist_hat_true.data).data

    return combined, normal


