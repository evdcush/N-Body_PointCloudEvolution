import chainer
import chainer.links as L
import chainer.functions as F
import graph_ops, nn


class nBodyModel(chainer.Chain):
    def __init__(self, channels, use_graph=False):
        self.channels = ch = channels
        self.use_graph = use_graph
        ch = [(ch[i],ch[i+1]) for i in range(0,len(ch)-1)]
        super(nBodyModel, self).__init__()
        layer = nn.GraphSubset if use_graph else nn.SetLinear
        # instantiate model layers
        for i in range(len(ch)):
            self.add_link('H' + str(i+1), layer(ch[i]))
    
    def get_sign(self, a):
        xp = self.xp
        ones = xp.ones(a.shape).astype(xp.float32)
        negones = -1*xp.ones(a.shape).astype(xp.float32)
        zeros = xp.zeros(a.shape).astype(xp.float32)
        return F.where(a.data < 0, negones, F.where(a.data > 0, ones, zeros))
    
    def get_readout(self, x_hat):
        readout = x_hat[...,:3]
        gt_one = ((self.get_sign(readout - 1) + 1)/2)
        ls_zero = -(self.get_sign(readout) - 1)/2
        rest = 1 - gt_one - ls_zero
        final = rest * readout + gt_one * (readout - 1) + ls_zero * (1 - readout)
        readout = final
        return readout
                
    def __call__(self, x, alist=None, n_NN=None, activation=F.relu, add=True, bounded=False):
        if self.use_graph: assert alist is not None
        h = activation(self.H1(x, alist, n_NN)) # first layer
        for i in range(2,len(self.channels)): # forward through layers
            cur_layer = getattr(self, 'H' + str(i))
            h = cur_layer(h, alist, n_NN, add=add)
            if i != len(self.channels)-1: # activation on all except output layer
                h = activation(h)
        if add: h += x[...,:3]
        if not bounded: h = self.get_readout(h)
        return h