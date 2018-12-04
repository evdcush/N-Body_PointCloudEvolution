import tensorflow as tf
import code
class Initializer:
    """Initializes variables and provides their getters
    """
    weight_tag = 'W{}_{}'
    bias_tag   = 'B_{}'
    scalar_tag = 'T_{}'
    def __init__(self, args):
        self.seed = args.seed
        self.restore = args.restore
        self.channels = args.channels
        self.var_scope = args.var_scope.format(*args.rs_idx)
        self.scalar_val = args.scalar_val
        self.num_layer_W = args.num_layer_W

    def initialize_scalars(self):
        """ scalars initialized by const value """
        for i in range(2):
            init = tf.constant([self.scalar_val])
            tag = self.scalar_tag.format(i)
            tf.get_variable(tag, dtype=tf.float32, initializer=init)

    def initialize_bias(self, layer_idx):
        """ biases initialized to be near zero """
        args = (self.bias_tag.format(layer_idx),)
        k_out = self.channels[layer_idx + 1] # only output chans relevant
        if self.restore:
            initializer = None
            args += (k_out,)
        else:
            initializer = tf.ones((k_out,), dtype=tf.float32) * 1e-8
        tf.get_variable(*args, dtype=tf.float32, initializer=initializer)

    def initialize_weight(self, layer_idx):
        """ weights sampled from glorot normal """
        kdims = self.channels[layer_idx : layer_idx+2]
        for w_idx in range(self.num_layer_W):
            name = self.weight_tag.format(layer_idx, w_idx)
            args = (name, kdims)
            init = None if self.restore else tf.glorot_normal_initializer(None)
            tf.get_variable(*args, dtype=tf.float32, initializer=init)

    def initialize_params(self):
        tf.set_random_seed(self.seed)
        with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
            for layer_idx in range(len(self.channels) - 1):
                #==== Layer vars
                self.initialize_bias(layer_idx)
                self.initialize_weight(layer_idx)
            #==== model vars
            self.initialize_scalars()

    # - - - - - - - - - - - -

    def get_scalars(self):
        t1 = tf.get_variable(self.scalar_tag.format(0))
        t2 = tf.get_variable(self.scalar_tag.format(1))
        return t1, t2

    def get_layer_vars(self, layer_idx):
        """ Gets all variables for a layer
        NOTE: ASSUMES VARIABLE SCOPE! Cannot get vars outside of scope.
        """
        get_W = lambda w: tf.get_variable(self.weight_tag.format(layer_idx, w))
        #=== layer vars
        weights = [get_W(w_idx) for w_idx in range(self.num_layer_W)]
        bias = tf.get_variable(self.bias_tag.format(layer_idx))
        return weights, bias

    def initialize_session(self):
        sess_kwargs = {}
        #==== Check for GPU
        if tf.test.is_gpu_available(cuda_only=True):
            gpu_frac = 0.85
            gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
            sess_kwargs['config'] = tf.ConfigProto(gpu_options=gpu_opts)
        #==== initialize session
        self.sess = tf.InteractiveSession(**sess_kwargs)

    def initialize_graph(self):
        """ initializes all variables after computational graph
            has been specified (via endpoints data input and optimized error)
        """
        self.sess.run(tf.global_variables_initializer())
        print('\n\nAll variables initialized\n')

    def __call__(self):
        """ return sess """
        if not hasattr(self, 'sess'):
            print('\nin initializer __call__, no sess attrib\n')
            self.initialize_session()
        return self.sess
