import chainer
import chainer.links as L
import chainer.functions as F
import nn
import graph_ops

'''
Design notes:
 - finished base model class
 - up next is graph_ops, trainers
   - for graph_ops, make it so it accepts a chainer.Variable too
 - FINISH graph_ops refactor, next up:
   - Fix data_utils. Currently messy duplicate functions with normal and gpu,
     just dispatch in funs
   - make trainer
   - work on density base dgraph, read attention stuff
'''
#=============================================================================
# Models
#=============================================================================
class Model(chainer.Chain):
    """ Base model class, defines basic functionality all models
    """
    def __init__(self, channels, layer, theta=None):
        self.channels = channels
        self.num_layers = len(channels) - 1
        
        super(Model, self).__init__()

        if theta is not None: # scalar for timestep
            if type(theta) == float: # static theta
                self.theta = lambda x: x*theta
            else:
                self.add_link('theta', L.Scale(axis=0, W_shape=(1,1,1)))
        else:
            self.theta = None

        # build network layers
        for i in range(self.num_layers):
            cur_layer = layer((channels[i], channels[i+1]))
            self.add_link('H' + str(i), cur_layer)

    def __call__(self, x, *args, activation=F.relu, add=True):
        h = x # this may mutate x, probably need to copy x
        for i in range(self.num_layers):
            cur_layer = getattr(self, 'H' + str(i))
            h = cur_layer(h, *args, add=add)
            if i != self.num_layers - 1:
                h = activation(h)
        if add:
            if h.shape[-1] == x.shape[-1]: h+= x
            else: h += x[...,:3]
            #h += x[...,:3]
        if self.theta is not None: 
            h += self.theta(x[...,3:])
        return h

#=============================================================================
class GraphModel(Model):
    """ GraphModel uses GraphLayer
     - apparently need to pass the numpy data to graph_ops.KNN
       since the to_cpu stuff within graph_ops gives significantly worse loss
    """
    def __init__(self, channels, K=14, **kwargs):
        self.K = K
        super(GraphModel, self).__init__(channels, nn.GraphLayer, **kwargs)

    def __call__(self, x, **kwargs):
        graphNN = graph_ops.KNN(chainer.cuda.to_cpu(x.data), self.K)
        return super(GraphModel, self).__call__(x, graphNN, **kwargs)

#=============================================================================
class SetModel(Model):
    """ SetModel uses SetLayer
    """
    def __init__(self, channels, **kwargs):
        super(SetModel, self).__init__(channels, nn.SetLayer, **kwargs)

#=============================================================================


class VelocityScaled(chainer.Chain):
    def __init__(self, *args, scalar=True, **kwargs): # *args, **kwargs just for convenience
        j = 1 if scalar else 3
        super(VelocityScaled, self).__init__(
            theta = L.Scale(axis=0, W_shape=(1,1,j)), # theta either scalar or (3,) vector
        )

    def __call__(self, x, **kwargs):
        return x[...,:3] + self.theta(x[...,3:])
