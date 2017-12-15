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
            if h.shape[-1] == 3: h += x[...,:3]
            else: h += x
        return h


class VelocityBias(chainer.Chain):
    def __init__(self,):
        super(VelocityBias, self).__init__(
            theta = L.Linear(3, 3, nobias=True),
        )

    def __call__(self, x):
        mb_size, N, D = x.shape
        x_r = F.reshape(x, (mb_size*N, D))
        x_coo, x_vel = F.split_axis(x_r, 2, -1)
        vel_scaled   = self.theta(x_vel)
        x_out = x_coo + vel_scaled
        return F.reshape(x_out, (mb_size, N, x_out.shape[-1]))