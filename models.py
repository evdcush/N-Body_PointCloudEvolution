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
    """ Base model class, defines basic functionality all models
    """
    rs_num = 10 # 11 redshifts, so 10 possible predictions
    def __init__(self, channels, *args, theta=None):
        self.channels = channels
        last_channels = channels[:-1] + [3,]
        
        super(RSModel, self).__init__(
            RS_6040 = models.SetModel(channels, theta=theta),
            RS_4020 = models.SetModel(channels, theta=theta),
            RS_2015 = models.SetModel(channels, theta=theta),
            RS_1512 = models.SetModel(channels, theta=theta),
            RS_1210 = models.SetModel(channels, theta=theta),
            RS_1008 = models.SetModel(channels, theta=theta),
            RS_0806 = models.SetModel(channels, theta=theta),
            RS_0604 = models.SetModel(channels, theta=theta),
            RS_0402 = models.SetModel(channels, theta=theta),
            RS_0200 = models.SetModel(last_channels, theta=theta),
            )
    def fwd_pred(self, x):
        # each layer receives previoous layer prediction
        rs_40_hat = self.RS_6040(x)
        #print('input: {}, output: {}'.format(x.shape, rs_40_hat.shape))
        rs_20_hat = self.RS_4020(rs_40_hat)
        rs_15_hat = self.RS_2015(rs_20_hat)
        rs_12_hat = self.RS_1512(rs_15_hat)
        rs_10_hat = self.RS_1210(rs_12_hat)
        rs_08_hat = self.RS_1008(rs_10_hat)
        rs_06_hat = self.RS_0806(rs_08_hat)
        rs_04_hat = self.RS_0604(rs_06_hat)
        rs_02_hat = self.RS_0402(rs_04_hat)
        rs_00_hat = self.RS_0200(rs_02_hat)
        return rs_00_hat
    
    def fwd_input(self, x):
        """ each layer receives external input
        x.shape = (11, bs, n_P, 6)
        """
        # each layer receives data input
        rs_40_hat = self.RS_6040(x[0])
        rs_20_hat = self.RS_4020(x[1])
        rs_15_hat = self.RS_2015(x[2])
        rs_12_hat = self.RS_1512(x[3])
        rs_10_hat = self.RS_1210(x[4])
        rs_08_hat = self.RS_1008(x[5])
        rs_06_hat = self.RS_0806(x[6])
        rs_04_hat = self.RS_0604(x[7])
        rs_02_hat = self.RS_0402(x[8])
        rs_00_hat = self.RS_0200(x[9])
        return rs_00_hat

    def fwd_input_pred_diff(self, x):
        """ each layer receives external input
        x.shape = (11, bs, n_P, 6)
        """
        rs_40_hat = self.RS_6040(x[0])
        diff_40 = F.abs(rs_40_hat - x[1])

        rs_20_hat = self.RS_4020(diff_40)
        diff_20 = F.abs(rs_20_hat - x[2])

        rs_15_hat = self.RS_2015(diff_20)
        diff_15 = F.abs(rs_15_hat - x[3])

        rs_12_hat = self.RS_1512(diff_15)
        diff_12 = F.abs(rs_12_hat - x[4])

        rs_10_hat = self.RS_1210(diff_12)
        diff_10 = F.abs(rs_10_hat - x[5])

        rs_08_hat = self.RS_1008(diff_10)
        diff_08 = F.abs(rs_08_hat - x[6])

        rs_06_hat = self.RS_0806(diff_08)
        diff_06 = F.abs(rs_06_hat - x[7])

        rs_04_hat = self.RS_0604(diff_06)
        diff_04 = F.abs(rs_04_hat - x[8])

        rs_02_hat = self.RS_0402(diff_02)
        diff_02 = F.abs(rs_02_hat - x[9])

        rs_00_hat = self.RS_0200(diff_02)
        return rs_00_hat
    
    
    
    def __call__(self, x, *args, pred_fwd=0):
        """
        x.shape == (1, n_P, 6) # just the 6.0 sample
        returns (1, n_P, 3) # the prediction on 0.0
        """
        if pred_fwd == 0:
            return self.fwd_input(x)
        elif pred_fwd == 1:
            return self.fwd_pred(x)
        elif pred_fwd == 2:
            return self.fwd_input_pred_diff(x)