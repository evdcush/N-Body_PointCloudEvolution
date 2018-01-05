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
    Consists of two set layers, one for the data input, the other for the 
    neighborhood graph
    
    Args:
        kdim: channel size tuple (k_in, k_out)
        nobias: if True, no bias weights used
    """
    def __init__(self, kdim, nobias=True):
        self.kdim = kdim
        super(SetLayer, self).__init__(
            input_linear = SetLinear(kdim, nobias=nobias),
            graph_linear = SetLinear(kdim, nobias=nobias),
            )
        
    def __call__(self, x_in, graphNN, add=True):
        x_out = 
        return self.input_layer(x_in, add=add)

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
            input_layer = SetLinear(kdim, nobias=nobias),
            graph_layer = SetLinear(kdim, nobias=nobias),
            )
        
    def __call__(self, x_in, graphNN, add=False):
        neighborhood_graph = graphNN(x_in)
        x_out     = self.input_layer(x_in)
        graph_out = self.graph_layer(neighborhood_graph)
        x_out += graph_out
        if add and self.kdim[0] == self.kdim[1]:
            x_out += x_in
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


