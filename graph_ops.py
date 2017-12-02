import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import chainer
import chainer.functions as F
from chainer import cuda
import cupy

def adjacency_list_tf(X_in,k):
    shape_in = X_in.shape
    X_out = np.zeros([shape_in[0],shape_in[1],k],dtype=np.int32)
    for b in range(shape_in[0]):
        # this returns indices of the nn
        X_out[b] = kneighbors_graph(X_in[b,:,:3],k,include_self=True).indices.reshape([shape_in[1],k])
        #print(X_out[b].shape)
    return X_out


def get_k_alist(X_in,k):
    """ search for k nneighbors, and return offsetted indices in adjacency list
    
    Args:
        X_in: input data of shape (mb_size, N, 6)
        k: number of nearest neighbors
    """
    mb_size, N, D = X_in.shape
    X_out = np.zeros([mb_size, N, k],dtype=np.int32)
    for b in range(mb_size):
        # this returns indices of the nn
        #graph_idx = kneighbors_graph(X_in[b,:,:3],k,include_self=True).indices.reshape([N,k]) + (N * b)
        graph_idx = kneighbors_graph(X_in[b,:,:3], k, include_self=True).indices
        graph_idx = graph_idx.reshape([N,k]) + (N * b)
        X_out[b] = graph_idx
    return X_out

def get_rad_alist(X_in, rad):
    '''
    how to deal with variable size neighborhood?
    gravity clumps will have large neighborhood, while isolated points will have small (possibly a single point)
    '''
    mb_size, N, D = X_in.shape
    #X_out = np.zeros([mb_size, N, N],dtype=np.bool)
    X_out = np.zeros([mb_size, N, N], dtype=np.float32)
    for b in range(mb_size):
        graph_idx = radius_neighbors_graph(X_in[b,:,:3], rad, include_self=True)
        #gidx = graph_idx.toarray().astype(np.bool)
        gidx = graph_idx.toarray().astype(np.float32)
        #gidx = gidx / np.sum(gidx, axis=-1, keepdims=True)
        X_out[b] = gidx
    X_out = chainer.Variable(cuda.to_gpu(X_out))
    X_out = F.scale(X_out, 1/F.sum(X_out, axis=-1),axis=0)
    return X_out


def get_alist(X_in, p, search_type):
    x_in = cuda.to_cpu(X_in.data)
    if search_type == 'rad':
        return get_rad_alist(x_in, p)
    else:
        return get_k_alist(x_in, p)


def nneighbors_graph(X_in, alist, n_NN=30):
    """ gets nneighbors for data, indexing from adjacency list indices
    
    Args:
        X_in: input data of shape (mb_size, N, 6)
        alist: adjacency list for X_in of shape (mb_size, N, num_NN)
    """
    mb_size, N, D = X_in.shape
    n_NN = alist.shape[-1]
    xr = F.reshape(X_in, (-1,D))
    graph = F.reshape(F.get_item(xr, alist.flatten()), (mb_size, N, n_NN, D))
    return graph

def knneighbors_graph(X, alist):
    """ gets nneighbors for data, indexing from adjacency list indices
    
    Args:
        X_in: input data of shape (mb_size, N, 6)
        alist: adjacency list for X_in of shape (mb_size, N, num_NN)
    """   
    mb_size, N, D = X.shape
    n_NN = alist.shape[-1]
    xr = F.reshape(X, (-1,D))
    graph = F.reshape(F.get_item(xr, alist.flatten()), (mb_size, N, n_NN, D))
    return F.mean(graph, axis=2)

def rad_nneighbors_graph(X, alist, xp=cupy):
    """ gets nneighbors for data, indexing from adjacency list indices
    
    Args:
        X_in: input data of shape (mb_size, N, 6)
        alist: adjacency list for X_in of shape (mb_size, N, num_NN)
    """
    mb_size, N, D = X.shape
    """
    ar = F.broadcast_to(F.reshape(alist, (mb_size,N,N,1)), (mb_size,N,N,D))
    xr = F.broadcast_to(F.reshape(X, (mb_size, 1, N, D)), (mb_size, N, N, D))
    zeros = chainer.Variable(xp.zeros_like(ar.data).astype(xp.float32))
    graph = F.where(ar, xr, zeros)
    """
    graph = F.batch_matmul(alist, X)
    return graph


class AdjacencyList():
    def __init__(self, X_in, search_type, p):
        self.search_type = search_type
        self.p = p
        self.alist = get_alist(X_in, p, search_type)

    def __call__(self, X):
        if self.search_type == 'rad':
            return rad_nneighbors_graph(X, self.alist)
        else:
            return knneighbors_graph(X, self.alist)
