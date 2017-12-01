import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import chainer
import chainer.functions as F


def adjacency_list_tf(X_in,k):
    shape_in = X_in.shape
    X_out = np.zeros([shape_in[0],shape_in[1],k],dtype=np.int32)
    for b in range(shape_in[0]):
        # this returns indices of the nn
        X_out[b] = kneighbors_graph(X_in[b,:,:3],k,include_self=True).indices.reshape([shape_in[1],k])
        #print(X_out[b].shape)
    return X_out


def get_adjacency_list(X_in,k):
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
        graph_idx = kneighbors_graph(X_in[b,:,:3],k,include_self=True).indices
        graph_idx = graph_idx.reshape([N,k]) + (N * b)
        X_out[b] = graph_idx
    return X_out

def get_adjacency_list_rad(X_in,rad):
    '''
    how to deal with variable size neighborhood?
    gravity clumps will have large neighborhood, while isolated points will have small (possibly a single point)
    '''
    mb_size, N, D = X_in.shape
    X_out = np.zeros([mb_size, N, k],dtype=np.int32)
    for b in range(mb_size):
        # this returns indices of the nn
        #graph_idx = kneighbors_graph(X_in[b,:,:3],k,include_self=True).indices.reshape([N,k]) + (N * b)
        graph_idx = radius_neighbors_graph(X_in[b,:,:3],k,include_self=True).indices
        graph_idx = graph_idx.reshape([N,k]) + (N * b)
        X_out[b] = graph_idx
    return X_out


# adjacency list to proper index list for get_item
def alist_to_indexlist(alist):
    """ tiles batch indices to adjacency list for tf.gather
    """
    b, n, k = alist.shape
    #b = alist.shape[0] # batch size
    #n = alist.shape[1] # set size
    #k = alist.shape[2] # number of nn
    id1 = np.reshape(np.arange(b),[b,1])
    id1 = np.tile(id1,n*k).flatten()
    out = np.stack([id1,alist.flatten()],axis=1)
    return out

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

def rad_nneighbors_graph(X_in, alist, n_NN=30):
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