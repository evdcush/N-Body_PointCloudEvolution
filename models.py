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

#=============================================================================
# experimental models
#=============================================================================

class RSModel(chainer.Chain):
    BOUND = (0.095, 1-0.095)
    def __init__(self, channels, *args, layer=SetModel, theta=None, n_P=16):
        self.channels = channels
        last_channels = channels[:-1] + [3,]
        self.tags = ['6040', '4020', '2015', '1512', '1210', '1008', '0806', '0604', '0402', '0200']
        theta_rs = [6.0, 4.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        theta_val = None
        super(RSModel, self).__init__()
        for i in range(len(self.tags)):
            cur_tag = self.tags[i]
            if theta is not None:
                theta_val = theta[(n_P, theta_rs[i], theta_rs[i+1])]['W']
            self.add_link('RS_' + cur_tag, layer(channels, theta=theta_val))
        
    def fwd_pred(self, x, rs_tup=(0,10)):
        """
        # each layer receives previoous layer prediction        
        rs_tup is idx of redshift
        if rs_tup == (7, 10), then you are making prediction from redshift 0.6 to 0.0
        """
        # x.shape == (1, n_P, 6)
        rs_start, rs_target = rs_tup
        redshift_distance = rs_target - rs_start
        assert redshift_distance > 0 and rs_target <= 10
        predictions = self.xp.zeros(((redshift_distance,) + x.shape)).astype(self.xp.float32)
        cur_layer = getattr(self, 'RS_' + self.tags[rs_start])
        hat = cur_layer(x)
        predictions[0] = hat.data
        if redshift_distance == 1:
            return hat, predictions
        else:
            for i in range(rs_start+1, rs_target):
                cur_layer = getattr(self, 'RS_' + self.tags[i])
                hat = cur_layer(hat)
                predictions[i] = hat.data
            return hat, predictions
    
    def fwd_input(self, x):
        """ each layer receives external input
        """
        hat = self.RS_6040(x[0])
        error = nn.get_bounded_MSE(hat, x[1], boundary=self.BOUND)
        for i in range(1, len(self.tags)):
            cur_layer = getattr(self, 'RS_' + self.tags[i])
            hat = cur_layer(x[i])
            error += nn.get_bounded_MSE(hat, x[i+1], boundary=self.BOUND)
        return hat, error
    
    def fwd_pred_loss(self, x):
        """ each layer receives external input
        """
        hat = self.RS_6040(x[0])
        error = nn.get_bounded_MSE(hat, x[1], boundary=self.BOUND)
        for i in range(1, len(self.tags)):
            cur_layer = getattr(self, 'RS_' + self.tags[i])
            hat = cur_layer(hat)
            error += nn.get_bounded_MSE(hat, x[i+1], boundary=self.BOUND)
        return hat, error