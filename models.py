import chainer
import chainer.links as L
import chainer.functions as F
import nn
import numpy
import cupy
import code
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

#=============================================================================
# Models
#=============================================================================
class Model(chainer.Chain):
    """ Base model class, defines basic functionality all models
    """
    def __init__(self, channels, layer, vel_coeff=None):
        self.channels = channels
        self.num_layers = len(channels) - 1

        super(Model, self).__init__()

        if vel_coeff is not None: # scalar for timestep
            if type(vel_coeff) == float: # static vel_coeff
                self.vel_coeff = lambda x: x*vel_coeff
            else:
                self.add_link('vel_coeff', L.Scale(axis=0, W_shape=(1,1,1)))
        else:
            self.vel_coeff = None

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
        if self.vel_coeff is not None:
            timestep_vel = self.vel_coeff(x[...,3:])
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
     - apparently need to pass the numpy data to nn.KNN
       since the to_cpu stuff within nn gives significantly worse loss
    """
    def __init__(self, channels, K=14, **kwargs):
        self.K = K
        super(GraphModel, self).__init__(channels, nn.GraphLayer, **kwargs)

    def __call__(self, x, **kwargs):
        # (bs, n_p, 6)
        L_box_size = 16 if x.shape[-2] == 16**3 else 32
        graphNN = nn.KNN_v2(chainer.cuda.to_cpu(x.data), self.K, L_box_size)
        #graphNN = nn.KNN(chainer.cuda.to_cpu(x.data), self.K, L_box_size)
        #graphNN = nn.NonPBKNN(chainer.cuda.to_cpu(x.data), self.K)
        return super(GraphModel, self).__call__(x, graphNN, **kwargs)

#=============================================================================
class GraphModel2(Model):
    """ GraphModel uses GraphLayer
     - apparently need to pass the numpy data to nn.KNN
       since the to_cpu stuff within nn gives significantly worse loss
    """
    def __init__(self, channels, K=14, **kwargs):
        self.K = K
        super(GraphModel2, self).__init__(channels, nn.GraphLayer, **kwargs)

    def __call__(self, x, **kwargs):
        # (bs, n_p, 6)
        L_box_size = 16 if x.shape[-2] == 16**3 else 32
        graphNN = nn.KNN_v2(chainer.cuda.to_cpu(x.data), self.K, L_box_size)
        #graphNN = nn.KNN(chainer.cuda.to_cpu(x.data), self.K, L_box_size)
        #graphNN = nn.NonPBKNN(chainer.cuda.to_cpu(x.data), self.K)
        return super(GraphModel2, self).__call__(x, graphNN, **kwargs)

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
            vel_coeff = L.Scale(axis=0, W_shape=(1,1,j)), # vel_coeff either scalar or (3,) vector
        )

    def __call__(self, x, **kwargs):
        return x[...,:3] + self.vel_coeff(x[...,3:])

#=============================================================================
# experimental models
#=============================================================================
class RSModel(chainer.Chain):
    # velocityscaled model not currently supported
    redshifts = [6.0, 4.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    ltag = 'Z_{}'
    def __init__(self, channels, layer=SetModel, vel_coeff=None, rng_seed=77743196):
        self.channels    = channels
        self.num_layers  = len(self.redshifts) - 1
        vel_coeff_weight = None
        super(RSModel, self).__init__()

        for i in range(self.num_layers):
            if vel_coeff is not None:
                vel_coeff_weight = vel_coeff[(self.redshifts[i], self.redshifts[i+1])]
            # seed before each layer to ensure same weights used at each redshift
            numpy.random.seed(rng_seed)
            cupy.random.seed(rng_seed)
            self.add_link(self.ltag.format(i), layer(channels, vel_coeff=vel_coeff_weight))

    def fwd_target(self, x, rs_tup=(0,10)):
        """ Model makes predictions from rs_tup[0] to rs_tup[-1]
        *Assumed to be used in validation only*

        Differs from fwd_prediction in that only a single redshift
        is taken as input.
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

        cur_layer = getattr(self, self.ltag.format(rs_start))
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        hat = cur_layer(x)
        predictions[0] = hat.data
        if redshift_distance == 1:
            return hat, predictions
        else:
            for i in range(rs_start+1, rs_target):
                cur_layer = getattr(self, self.ltag.format(i))
                hat = cur_layer(hat)
                predictions[i-rs_start] = hat.data
            return hat, predictions


    def fwd_predictions(self, x, loss_fun=nn.bounded_mean_squared_error):
        """ Forward predictions through network layers
        Each layer receives the previous layer's prediction as input
        The loss is calculated between truth and prediction and summed

        Args:
            x (chainer.Variable): data batch of shape (11, batch_size, num_particles, 6)
        Returns:
            hat (chainer.Variable): the last layer's prediction of shape (batch_size, num_particles, 6)
            error (chainer.Variable): summed error of layer predictions against truth
        """
        hat = self.Z_0(x[0])
        error = loss_fun(hat, x[1])
        for i in range(1, self.num_layers):
            cur_layer = getattr(self, self.ltag.format(i))
            hat = cur_layer(hat)
            error += loss_fun(hat, x[i+1])
        return hat, error