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
'''
#=============================================================================
# Models
#=============================================================================
class Model(chainer.Chain):
    """ Base model class, defines basic functionality all models
    """
    def __init__(self, channels, layer, theta_scale=False):
        self.channels = channels
        self.num_layers = len(channels) - 1
        self.theta_scale = theta_scale
        super(Model, self).__init__()

        if theta_scale: # scalar for timestep
            self.add_link('theta', L.Scale(axis=0, W_shape=(1,1,1)))

        # build network layers
        for i in range(self.num_layers):
            cur_layer = layer((channels[i], channels[i+1]))
            self.add_link('H' + str(i), cur_layer)


    def __call__(self, x, *args, activation=F.relu, add=True, static_theta=None):
        h = x # this may mutate x, probably need to copy x
        for i in range(self.num_layers):
            cur_layer = getattr(self, 'H' + str(i))
            h = cur_layer(h, *args, add=add)
            if i != self.num_layers - 1:
                h = activation(h)
        if add:
            h += x[...,:3]
        if self.theta_scale:
            # scales output by some timestep*input_vel
            if static_theta is not None:
                h += static_theta * x[...,3:]
            else:
                h += self.theta(x[...,3:])
        return h

#=============================================================================
class GraphModel(Model):
    """ GraphModel uses GraphLayer
    """
    def __init__(self, channels, K, **kwargs):
        self.K = K
        super(GraphModel, self).__init__(channels, nn.GraphLayer, **kwargs)

    def __call__(self, x, **kwargs):
        graphNN = graph_ops.KNN(x, self.K)
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
