import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import chainer
import chainer.functions as F
from chainer import cuda
import cupy

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