import chainer
import chainer.links as L
import chainer.functions as F
import nn

class nBodyModel(chainer.Chain):
    def __init__(self, channels, use_graph=False):
        self.channels = ch = channels
        self.use_graph = use_graph
        ch = [(ch[i],ch[i+1]) for i in range(0,len(ch)-1)]

        super(nBodyModel, self).__init__()
        layer = nn.GraphSubset if self.use_graph else nn.SetLinear
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

    def fwd_graph(self, x, activation, alist, add=False):
        h = activation(self.H1(x, alist))
        for i in range(2, len(self.channels)):
            cur_layer = getattr(self, 'H' + str(i))
            h = cur_layer(h, alist, add=add)
            if i != len(self.channels)-1:
                h = activation(h)
        return h

    def fwd_set(self, x, activation, add=False):
        h = activation(self.H1(x))
        for i in range(2, len(self.channels)):
            cur_layer = getattr(self, 'H' + str(i))
            h = cur_layer(h, add=add)
            if i != len(self.channels)-1:
                h = activation(h)
        return h
                
    def __call__(self, x, activation=F.relu, graphNN=None, add=True, bounded=False):
        if self.use_graph:
            h = self.fwd_graph(x, activation, graphNN, add=add)
        else:
            h = self.fwd_set(x, activation, add=add)
        #fwd = self.fwd_graph if self.use_graph else self.fwd_set
        #h = fwd(x, activation, add)
        if add: h += x[...,:3]
        if not bounded: h = self.get_readout(h)
        return h