import chainer
import chainer.links as L
import chainer.functions as F
import nn
import numpy
import cupy
import code
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph as rad_graph
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
     - moved Daniele's periodic boundaries KNN here
    """
    def __init__(self, channels, K=14, **kwargs):
        self.K = K
        self.radius = 0.03
        super(GraphModel, self).__init__(channels, nn.GraphLayer, **kwargs)

    def init_periodic_boundary_conds(self, box_size, shell_fraction=0.1):
        # Change shell_fraction for tuning the thickness of the shell to be replicated, must be in range 0-1
        # 0: no replication, 1: entire box is replicated to the 26 neighbouring boxes
        self.L_box = 16 if box_size == 16**3 else 32
        self.dL = self.L_box * shell_fraction

    def _get_status(self, coordinate):
        """
        Assign a status to each coordinate (of a particle position inside the box):
        1 if 0 < coord < dL, 2 if L- dL < coord < L, 0 otherwise

        PARAMS:
            coordinate(float)
        RETURNS:
            status(int). Either 0, 1, or 2
        """
        if coordinate < self.dL:
            return 1
        elif self.L_box - self.dL < coordinate < self.L_box:
            return 2
        else:
            return 0

    def _get_clone(self, particle, k, s):
        """
        Clone a particle otuside of the box.

        PARAMS:
            particle(np array). 6-dim particle position in phase space
            k(int). Index of dimension that needs to be projected outside of the box.
            s(int). Status, either 1 or 2. Determines where should be cloned.
        RETURNS:
            clone(np array). 6-dim cloned particle position in phase space.
        """
        clone = []
        for i in range(6):
            if i == k:
                if s == 1:
                    clone.append(particle[i] + self.L_box)
                elif s == 2:
                    clone.append(particle[i] - self.L_box)
            else:
                clone.append(particle[i])
        return numpy.array(clone)

    def get_adjacency_list_periodic_bc_v2(self, X_in):
        """
        Map inner chunks to outer chunks
        """
        K = self.K
        mb_size, N, D = X_in.shape
        adj_list = numpy.zeros([mb_size, N, K], dtype=numpy.int32)

        for i in range(mb_size):
            ids_map = {}  # For this batch will map new_id to old_id of cloned particles
            new_X = [part for part in X_in[i]]  # Start off with original cube
            for j in range(N):
                status = [self._get_status(X_in[i, j, k]) for k in range(3)]
                if sum(status) == 0:  # Not in the shell --skip
                    continue
                else:
                    for k in range(3):
                        if status[k] > 0:
                            clone = self._get_clone(particle=X_in[i, j, :], k=k, s=status[k])
                            new_X.append(clone)
                            ids_map.update({len(new_X) - 1: j})
                            for kp in range(k + 1, 3):
                                if status[kp] > 0:
                                    bi_clone = self._get_clone(particle=clone, k=kp, s=status[kp])
                                    new_X.append(bi_clone)
                                    ids_map.update({len(new_X) - 1: j})
                                    for kpp in range(kp + 1, 3):
                                        if status[kpp] > 0:
                                            tri_clone = self._get_clone(particle=bi_clone, k=kpp, s=status[kpp])
                                            new_X.append(tri_clone)
                                            ids_map.update({len(new_X) - 1: j})

            new_X = numpy.array(new_X)
            graph = kneighbors_graph(new_X[:, :3], K, include_self=True)
            graph_idx = graph.indices
            #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
            graph_idx = graph_idx.reshape([-1, K])[:N, :]  # Only care about original box
            # Remap outbox neighbors to original ids
            for j in range(N):
                for k in range(K):
                    if graph_idx[j, k] >= N:  # If outside of the box
                        graph_idx[j, k] = ids_map[graph_idx[j, k]]
            graph_idx = graph_idx + (N * i)  # offset idx for batches
            adj_list[i] = graph_idx
        return adj_list

    def get_radius_adjacency_list_periodic_bc_v2(self, X_in):
        """
        Map inner chunks to outer chunks
        """
        K = self.K
        mb_size, N, D = X_in.shape
        #adj_list = numpy.zeros([mb_size, N, K], dtype=numpy.int32)
        csr_dict = {}

        for i in range(mb_size):
            ids_map = {}  # For this batch will map new_id to old_id of cloned particles
            new_X = [part for part in X_in[i]]  # Start off with original cube
            for j in range(N):
                status = [self._get_status(X_in[i, j, k]) for k in range(3)]
                if sum(status) == 0:  # Not in the shell --skip
                    continue
                else:
                    for k in range(3):
                        if status[k] > 0:
                            clone = self._get_clone(particle=X_in[i, j, :], k=k, s=status[k])
                            new_X.append(clone)
                            ids_map.update({len(new_X) - 1: j})
                            for kp in range(k + 1, 3):
                                if status[kp] > 0:
                                    bi_clone = self._get_clone(particle=clone, k=kp, s=status[kp])
                                    new_X.append(bi_clone)
                                    ids_map.update({len(new_X) - 1: j})
                                    for kpp in range(kp + 1, 3):
                                        if status[kpp] > 0:
                                            tri_clone = self._get_clone(particle=bi_clone, k=kpp, s=status[kpp])
                                            new_X.append(tri_clone)
                                            ids_map.update({len(new_X) - 1: j})

            new_X = numpy.array(new_X)
            #graph_idx = kneighbors_graph(new_X[:, :3], K, include_self=True).indices

            '''
            original_box_idx = graph_idx.reshape([-1, K])[:N, :]

            original_box_idx_flat = graph_idx[:N*K]
            og_box_flat_reshape = original_box_idx_flat.reshape([N,K])
            original_box_idx == og_box_flat_reshape is true

            # equivalent operation using indptr
            original_box_indptr = graph.indptr[:N+1]
            ptr_to_original_indices = numpy.zeros((N,K))
            for i in range(N):
                ptr_to_original_indices[i] = graph.indices[original_box_indptr[i]:original_box_indptr[i+1]]
            assert numpy.all(ptr_to_original_indices ==  original_box_idx # TRUE

            # SO, adapting for radius neighbors should use indptr
            original_box_indptr = graph.indptr[:N+1]

            # should only need to mess with indices and indptr
            '''
            # try LIL matrix
            graph = rad_graph(new_X[:,:3], self.radius, include_self=True).tolil()[:N,:]

            for j in range(N):
                graph.rows[j] = [r if r < N else ids_map[r] for r in graph.rows[j]]
            graph_csr = graph[:,:N].tocsr()
            csr_dict[i] = [graph_csr, numpy.diff(graph_csr.indptr)]

        return csr_dict

    def __call__(self, x, **kwargs):
        # (bs, n_p, 6)
        self.init_periodic_boundary_conds(x.shape[-2])
        x_in = chainer.cuda.to_cpu(x.data)
        adjacency_list = self.get_adjacency_list_periodic_bc_v2(x_in)
        #graphNN = nn.KNN_v2(chainer.cuda.to_cpu(x.data), self.K, L_box_size)
        return super(GraphModel, self).__call__(x, adjacency_list, **kwargs)

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