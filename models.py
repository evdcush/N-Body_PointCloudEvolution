import chainer
import chainer.links as L
import chainer.functions as F
import nn

"""
class Model(chainer.Chain):
    def __init__(self, channels):
        self.channels = ch = channels
        ch = [(ch[i],ch[i+1]) for i in range(0,len(ch)-1)]
        super(Model, self).__init__()

    def __call__(self, x, activation=F.relu, graphNN=Non)
"""



class nBodyModel(chainer.Chain):
    def __init__(self, channels, use_graph=False):
        self.channels = ch = channels
        self.use_graph = use_graph
        ch = [(ch[i],ch[i+1]) for i in range(0,len(ch)-1)]

        super(nBodyModel, self).__init__(
            #VelScalar = L.Scale(axis=0, W_shape=(1,1,1)),
            )
        layer = nn.GraphSubset if self.use_graph else nn.SetLinear
        # instantiate model layers
        for i in range(len(ch)):
            self.add_link('H' + str(i+1), layer(ch[i]))
    

    def fwd_graph(self, x, activation, graphNN, add=False):
        h = activation(self.H1(x, graphNN))
        for i in range(2, len(self.channels)):
            cur_layer = getattr(self, 'H' + str(i))
            h = cur_layer(h, graphNN, add=add)
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
                
    def __call__(self, x, activation=F.relu, graphNN=None, add=True):
        if self.use_graph:
            h = self.fwd_graph(x, activation, graphNN, add=add)
        else:
            h = self.fwd_set(x, activation, add=add)

        if add:
            # assume h.shape[-1] == 3
            h += x[...,:3] #+ self.VelScalar(x[...,3:])
        return h


class ScaleVelocity(chainer.Chain):
    def __init__(self, scalar=True):
        j = 1 if scalar else 3
        super(ScaleVelocity, self).__init__(
            theta = L.Scale(axis=0, W_shape=(1,1,j)), # theta either scalar or (3,) vector
        )

    def __call__(self, x, activation=None, graphNN=None, add=None):
        return x[...,:3] + self.theta(x[...,3:])
