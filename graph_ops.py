import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import chainer
import chainer.functions as F
from chainer import cuda
import cupy

xp = cupy

def adjacency_list_tf(X_in,k):
    shape_in = X_in.shape
    X_out = np.zeros([shape_in[0],shape_in[1],k],dtype=np.int32)
    for b in range(shape_in[0]):
        # this returns indices of the nn
        X_out[b] = kneighbors_graph(X_in[b,:,:3],k,include_self=True).indices.reshape([shape_in[1],k])
        #print(X_out[b].shape)
    return X_out


class GraphNN():
    def __init__(self, X_in, search_type, k):
        self.search_type = search_type
        self.k = k
        graph_init = DensityNN if search_type == 'rad' else KNN
        self.graph = graph_init(X_in, k)

    def __call__(self, X):
        return self.graph(X)


class DensityNN():
    def __init__(self, X_in, rad):
        self.k = rad
        self.alist = self.get_alist(X_in)
        
    def get_alist(self, X_in):
        '''
        how to deal with variable size neighborhood?
        gravity clumps will have large neighborhood, while isolated points will have small (possibly a single point)
        '''
        rad = self.k
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

    def __call__(self, X):
        mb_size, N, D = X.shape
        """
        ar = F.broadcast_to(F.reshape(alist, (mb_size,N,N,1)), (mb_size,N,N,D))
        xr = F.broadcast_to(F.reshape(X, (mb_size, 1, N, D)), (mb_size, N, N, D))
        zeros = chainer.Variable(xp.zeros_like(ar.data).astype(xp.float32))
        graph = F.where(ar, xr, zeros)
        """
        graph = F.batch_matmul(self.alist, X)
        return graph
        

class KNN():
    def __init__(self, X_in, n_NN):
        self.k = n_NN
        self.alist = self.get_alist(X_in)
        
    def get_alist(self, X_in):
        """ search for k nneighbors, and return offsetted indices in adjacency list
        
        Args:
            X_in: input data of shape (mb_size, N, 6)
            k: number of nearest neighbors
        """
        n_NN = self.k
        mb_size, N, D = X_in.shape
        X_out = np.zeros([mb_size, N, n_NN],dtype=np.int32)
        for b in range(mb_size):
            # this returns indices of the nn
            graph_idx = kneighbors_graph(X_in[b,:,:3], n_NN, include_self=True).indices
            graph_idx = graph_idx.reshape([N,n_NN]) + (N * b)
            X_out[b] = graph_idx
        return cuda.to_gpu(X_out)

    def __call__(self, X):
        alist = xp.copy(self.alist)
        mb_size, N, D = X.shape
        n_NN = alist.shape[-1]
        xr = F.reshape(X, (-1,D))
        graph = F.reshape(F.get_item(xr, alist.flatten()), (mb_size, N, n_NN, D))
        return F.mean(graph, axis=2)