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
        self.theta = None
        super(Model, self).__init__()

        if theta is not None: # scalar for timestep
            if type(theta_scale) == float: # static theta
                self.theta = lambda x: x*theta
            else:
                self.add_link('theta', L.Scale(axis=0, W_shape=(1,1,1)))

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
        '''
        16 -> 3 # (batch_size, )

        # calculate norm h
        h[...,0] # x
        h[...,1] # y
        h[...,2] # z

        take norm of output -> how much gravity should be at work here
         - whatever force is remaining is the difference

        h_init #(output of layer)
        vector_of_acceleration = h_init
         - plot: beginning of vector is x_input, end h_init + x_input -> vector
         - for each locations of input x, 
        
        force_vector = x_input -> h_init + x_input
        '''
        if add:
            h += x[...,:3]
        if self.theta is not None:
            h += self.theta(x[...,3:])
        return h

#=============================================================================
class GraphModel(Model):
    """ GraphModel uses GraphLayer
     - apparently need to pass the numpy data to graph_ops.KNN
       since the to_cpu stuff within graph_ops gives significantly worse loss
    """
    def __init__(self, channels, K, **kwargs):
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


class ScaleVelocity(chainer.Chain):
    def __init__(self, scalar=True):
        j = 1 if scalar else 3
        super(ScaleVelocity, self).__init__(
            theta = L.Scale(axis=0, W_shape=(1,1,j)), # theta either scalar or (3,) vector
        )

    def __call__(self, x, activation=None, graphNN=None, add=None):
        return x[...,:3] + self.theta(x[...,3:])
