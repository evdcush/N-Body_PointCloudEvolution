import chainer
import chainer.links as L
import chainer.functions as F
import graph_ops
import cupy
import numpy as np

#=============================================================================
# network layers
#=============================================================================
class SetLinear(chainer.Chain):
    """ Permutation equivariant linear layer for set data

    Args:
        kdim: channel size tuple (k_in, k_out)
        nobias: if True, no bias weights used
    """
    def __init__(self, kdim, nobias=False):
        self.kdim = kdim
        super(SetLinear, self).__init__(
            lin1 = L.Linear(kdim[0], kdim[1], nobias=nobias),
            )
        # Linear link wraps Wx+b function
        # can specify weight-init function with initialW=chainer.initializers.*

    def __call__(self, x, add=False):
        mb_size, N, D = x.shape
        k_in, k_out  = self.kdim
        x_mean = F.broadcast_to(F.mean(x, axis=1, keepdims=True),x.shape)
        #x_max  = F.broadcast_to(F.max(x, axis=1, keepdims=True),x.shape)
        x_r = F.reshape(x - x_mean, (mb_size*N, k_in))
        x1 = self.lin1(x_r)
        if add and k_in == k_out: x1 += x_r # shouldn't this be += x ?
        x_out = F.reshape(x1, (mb_size,N,k_out))  
        return x_out

#=============================================================================
class GraphSubset(chainer.Chain):
    """ graph subset layer, consists of two sets of permutation
        equivariant weights for graph and data
    
        Args:
            kdim: channel size tuple (k_in, k_out)
            nobias: if True, no bias weights used
        """
    def __init__(self, kdim, nobias=True):
        self.kdim = kdim
        super(GraphSubset, self).__init__(
            setlayer1 = SetLinear(kdim, nobias=nobias),
            setlayer2 = SetLinear(kdim, nobias=nobias),
            )
        
    def __call__(self, X_in, graphNN, add=False):
        # CAREFUL WITH SKIP CONNECTION REDUNDANCY
        #ngraph = graph_ops.nneighbors_graph(X_in, alist, n_NN) # (b, N, n_NN, D)
        nn_graph = graphNN(X_in)
        #x0 = F.mean(ngraph, axis=2)
        x1 = self.setlayer1(X_in)
        x2 = self.setlayer2(nn_graph)
        x_out = x1 + x2
        if add and self.kdim[0] == self.kdim[1]: x_out += X_in # is this the right place to have skips
        return x_out

#=============================================================================
# Loss related ops
#=============================================================================
def loss_fun(readout, x_true):
    # loss part
    x_true_loc = x_true[...,:3]
    dist = F.minimum(F.square(readout - x_true_loc), F.square(readout - (1 + x_true_loc)))
    dist = F.minimum(dist, F.square((1 + readout) - x_true_loc))
    # l2-dist
    loss = F.mean(F.sum(dist, axis=-1))
    return loss

def get_bounded(x, bound, xp=cupy):
    #gtb, ltb = bound, 1-bound
    lower, upper = bound, 1-bound
    gt = lower < x.data
    lt = x.data < upper
    bounded = xp.all(gt & lt, axis=-1) # shape should be (mb_size, 4096)
    return bounded

def loss_fun_bounded(x_hat, x_true, bound):
    # loss part
    readout = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bounding_idx = get_bounded(x_true_loc, bound)
    bdreadout = F.get_item(readout, bounding_idx)
    bdx_true_loc = F.get_item(x_true_loc, bounding_idx)
    dist_mse = F.mean(F.sum(F.square(bdreadout - bdx_true_loc), axis=-1))
    return dist_mse



def loss_fun_bounded2(readout, x_true, bound=0.1):
    # loss part
    x_true_loc = x_true[...,:3]
    bounded = get_bounded(x_true_loc, bound)
    unbounded = F.sum(F.square(readout - x_true_loc), axis=-1)
    zeros = xp.zeros_like(unbounded)
    bounded_diff = F.where(bounded, unbounded, zeros)
    dist_mse = F.mean(bounded_diff)
    return dist_mse

def foo(xp=cupy):
    print('cupy: {}'.format(cupy))