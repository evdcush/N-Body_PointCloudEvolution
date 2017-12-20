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
            #layer_norm = L.LayerNormalization()
            )
        # Linear link wraps Wx+b function
        # can specify weight-init function with initialW=chainer.initializers.*

    def __call__(self, x, add=False, final=False):
        mb_size, N, D = x.shape
        k_in, k_out  = self.kdim
        x_mean = F.broadcast_to(F.mean(x, axis=1, keepdims=True),x.shape)
        #x_max  = F.broadcast_to(F.max(x, axis=1, keepdims=True),x.shape)
        x_r = F.reshape(x - x_mean, (mb_size*N, k_in))
        x1 = self.lin1(x_r)
        if add and k_in == k_out: x1 += x_r # shouldn't this be += x ?
        #if not final: x1 = self.layer_norm(x1)
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
        
    def __call__(self, x_in, graphNN, add=False):
        # CAREFUL WITH SKIP CONNECTION REDUNDANCY
        #ngraph = graph_ops.nneighbors_graph(X_in, alist, n_NN) # (b, N, n_NN, D)
        nn_graph = graphNN(x_in)
        #x0 = F.mean(ngraph, axis=2)
        x1 = self.setlayer1(x_in)
        x2 = self.setlayer2(nn_graph)
        x_out = x1 + x2
        if add and self.kdim[0] == self.kdim[1]: x_out += x_in # is this the right place to have skips
        return x_out

class GatedGraphSubset(chainer.Chain):
    """ graph subset layer, consists of two sets of permutation
        equivariant weights for graph and data
    
        Args:
            kdim: channel size tuple (k_in, k_out)
            nobias: if True, no bias weights used
        """
    def __init__(self, kdim, nobias=True):
        self.kdim = kdim
        super(GatedGraphSubset, self).__init__(
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

def get_MSE(x_hat, x_true, boundary=0.095):
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


def get_bounded(x, bound, xp=cupy):
    return xp.all(xp.logical_and(bound<x.data, x.data<1-bound),axis=-1)

def get_bounded_MSE(x_hat, x_true, boundary):
    x_hat_loc  = x_hat[...,:3]
    x_true_loc = x_true[...,:3]
    bidx = get_bounded(x_true_loc, boundary)
    bhat  = F.get_item(x_hat_loc, bidx)
    btrue = F.get_item(x_true_loc, bidx)
    return F.mean(F.sum(F.squared_difference(bhat, btrue), axis=-1))


def get_combined_MSE(x_input, x_hat, x_true, boundary):
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


