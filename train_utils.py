import chainer
import chainer.links as L
import chainer.functions as F
import graph_ops
import cupy
import numpy as np
from chainer import cuda
import data_utils

from chainer.datasets import TupleDataset
import chainer.training as training
from chainer.dataset import iterator
from chainer.iterators import SerialIterator
#from chainer.training import Updater

#tupset = TupleDataset(x,y) # tupset[i] == x[i],y[i]


class nBodyIterator(iterator.Iterator):
    '''
    NOTE: only modification from chainer.iterators.SerialIterator is 
     the data shifting procedure in __next__
    '''
    """Dataset iterator that serially reads the examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.

    """
    def peturb_data(self, batch):
        # batch is tuple (x,y) where x.shape == y.shape == (mb_size, N, D)
        batch_size = batch[0].shape[0]
        rands = self.xp.random.rand(6)
        shift = self.xp.random.rand(batch_size,3)
        out = []
        for arr in batch:
            tmp = arr
            if rands[0] < .5:
                tmp = tmp[:,:,[1,0,2,4,3,5]]
            if rands[1] < .5:
                tmp = tmp[:,:, [0,2,1,3,5,4]]
            if rands[2] < .5:
                tmp = tmp[:,:, [2,1,0,5,4,3]]
            if rands[3] < .5:
                tmp[:,:,0] = 1 - tmp[:,:,0]
                tmp[:,:,3] = -tmp[:,:,3]
            if rands[4] < .5:
                tmp[:,:,1] = 1 - tmp[:,:,1]
                tmp[:,:,4] = -tmp[:,:,4]
            if rands[5] < .5:
                tmp[:,:,2] = 1 - tmp[:,:,2]
                tmp[:,:,5] = -tmp[:,:,5]            
            tmploc = tmp[:,:,:3]
            tmploc += shift[:,None,:]
            gt1 = tmploc > 1
            tmploc[gt1] = tmploc[gt1] - 1
            tmp[:,:,:3] = tmploc
            out.append(tmp)
        return tuple(out)

    def __next__(self):
        batch = super(nBodyIterator, self).__next__()
        return self.peturb_data(batch)

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