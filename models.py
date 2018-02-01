import chainer
import chainer.links as L
import chainer.functions as F
import nn
import graph_ops
import cupy
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

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
            if h.shape[-1] == x.shape[-1]: # meaning, model is also predicting velocity
                h += x
            else: # model only predict coordinates
                h += x[...,:3]
        if self.theta is not None:
            timestep_vel = self.theta(x[...,3:])
            if h.shape[-1] == x.shape[-1]:
                # need to split and concat, since direct index assignment not supported
                h_loc, h_vel = F.split_axis(h, [3], -1) # splits h[...,:3], h[...,3:]
                h_loc += timestep_vel
                h = F.concat((h_loc, h_vel), axis=-1)
            else:
                h += timestep_vel
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
        # (bs, n_p, 6)
        L_box_size = 16 if x.shape[-2] == 16**3 else 32
        graphNN = graph_ops.KNN(chainer.cuda.to_cpu(x.data), self.K, L_box_size)
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
    redshifts = [6.0, 4.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    rs_idx    = list(range(len(redshifts)))
    layer_tag = 'RS_'

    tags = ['6040', '4020', '2015', '1512', '1210', '1008', '0806', '0604', '0402', '0200']    
    def __init__(self, channels, layer=SetModel, theta=None, rng_seed=77743196):
        self.channels = channels
        theta_weight = None
        super(RSModel, self).__init__()

        for i in self.rs_idx:
            if theta is not None:
                theta_weight = theta[(self.redshifts[i], self.redshifts[i+1])]
            
            # seed before each layer to ensure same weights used at each redshift
            np.random.seed(rng_seed)
            cupy.random.seed(rng_seed)
            self.add_link('RS_{}'.format(i), layer(channels, theta=theta_weight))
        
    def fwd_pred(self, x, rs_tup=(0,10)):
        """ Model makes predictions from rs_tup[0] to rs_tup[-1]
        This is different from the forwarding used in training in that
        it only receives input for a single redshift.
        If rs_tup == (7, 10), then you are making predictions from
        redshift 0.6 to 0.0

        Args:
            x: input data, of shape (1, n_P, 6) # rs_start
            rs_tup: rs_tup[0] is the starting redshift, rs_tup[1] is target redshift
        Returns:
            hat: (rs_tup[1] - rs_tup[0], 1, n_p, 6) shaped variable for predictions
                 from rs_start to rs_target
        """
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