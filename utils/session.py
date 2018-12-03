
import os
import time
import tensorflow as tf


# Network features
# ================
CHANNELS = [9, 32, 16, 8, 3]
SHIFT_INV_W_IDX = [1,2,3,4]


# Variable naming
# ===============
VARIABLE_SCOPE = 'params_{}-{}' # eg. 'params_0-7' for rs 9.0000 --> 1.1212
WEIGHT_TAG = 'W{}_{}'
BIAS_TAG   = 'B_{}'
SCALAR_TAG = 'T_{}'

# Initialization params
# =====================
PARAMS_SEED = 77743196


# Variable inits
# ==============
def initialize_var(args_init, initializer):
    tf.get_variable(*args_init, dtype=tf.float32, initializer=initializer)


def initialize_weight(name, kdims, restore=False):
    """ initialize weight Variable
    Args:
        name (str): variable name
        kdims tuple(int): kernel sizes (k_in, k_out)
        restore (bool): if restore, then do not user initializer
    """
    args_init = (name, kdims)
    initializer = None if restore else tf.glorot_normal_initializer(None)
    initialize_var(args_init, initializer)


def initialize_bias(name, kdims, restore=False):
    """ biases initialized to be near zero
    Args:
        name (str): variable name
        kdims tuple(int): kernel sizes (k_in, k_out), only k_out used for bias
        restore (bool): if restore, then do not user initializer
    """
    k_out = kdims[-1]
    args_init = (name,)
    if restore:
        initializer = None
        args_init += (k_out,)
    else:
        initializer = tf.ones((k_out,), dtype=tf.float32) * 1e-8
    initialize_var(args_init, initializer)


def initialize_scalars(num_scalars=2, init_val=0.002, restore=False):
    """ 1D scalars used to scale network outputs """
    initializer = tf.constant([init_val])
    for i in range(num_scalars):
        initialize_var((SCALAR_TAG.format(i),), initializer)


# Model parameter init wrappers
# =======================================
def initialize_shift_inv_params(kdims, restore=False):
    """ ShiftInv layers have 1 bias, 4 weights, scalars """
    for layer_idx, kdim in enumerate(kdims):
        #==== init bias
        bias_name = BIAS_TAG.format(layer_idx)
        initialize_bias(bias_name, kdim, restore=restore)

        #==== init weights
        for w_idx in SHIFT_INV_W_IDX:
            weight_name = WEIGHT_TAG.format(layer_idx, w_idx)
            initialize_weight(weight_name, kdim, restore=restore)
    #==== init scalars
    initialize_scalars(restore=restore)




def initialize_model_params(args):
    """ Initialize model parameters, dispatch based on layer_type
    Args:
        layer_type (str): layer-type ['vanilla', 'shift-inv', 'rot-inv']
        channels list(int): network channels
        scope (str): scope for tf.variable_scope
        seed (int): RNG seed for param inits
        restore (bool): whether new params are initialized, or just placeholders
    """
    restore = args.restore
    scope = args.var_scope
    seed = args.seed

    # Check layer_type integrity
    assert layer_type in LAYER_TYPES

    # Convert channels to (k_in, k_out) tuples
    kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]

    # Seed and initialize
    tf.set_random_seed(seed)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        layer_init(layer_type, kdims, restore)
    print('Initialized {} layer parameters'.format(layer_type))



#------------------------------------------------------------------------------
# Parameter getters
#------------------------------------------------------------------------------
# Variable gets
# ========================================
def get_var(name):
    """ Assumes within variable scope """
    return tf.get_variable(name)


def get_weight(layer_idx, w_idx=0):
    name = WEIGHT_TAG.format(layer_idx, w_idx)
    return get_var(name)


def get_bias(layer_idx, suffix=''):
    name = BIAS_TAG.format(f'{layer_idx}{suffix}')
    return get_var(name)


def get_scalars(num_scalars=2):
    scalars = [get_var(SCALAR_TAG.format(i)) for i in range(num_scalars)]
    return scalars

class Variables:
    """ Provides the initialization and getters for variables
    """
    def __init__(self, args):
        pass




# Layer var get wrappers
# ========================================
"""
Assumed to be within the tf.variable_scope of the respective network funcs
  themselves (called directly), so no dispatching layer wrapper get func
"""
def get_shift_inv_layer_vars(layer_idx, **kwargs):
    weights = []
    for w_idx in SHIFT_INV_W_IDX:
        weights.append(get_weight(layer_idx, w_idx=w_idx))
    bias = get_bias(layer_idx)
    return weights, bias




#==============================================================================
#
#                      o8o
#                      `"'
#    oooo oooo    ooo oooo  oo.ooooo.
#     `88. `88.  .8'  `888   888' `88b
#      `88..]88..8'    888   888   888
#       `888'`888'     888   888   888
#        `8'  `8'     o888o  888bod8P'
#                            888
#                           o888o
#
#==============================================================================




class TrainSaver:
    #==== directories
    params_dir  = 'Session'
    results_dir = 'Results'
    experiments_dir = 'Experiments'

    #==== Naming formats
    model_name_format = 'SI_{}{}-{}'
    cube_name_format  = 'X_{}-{}_{}'

    def __init__(self, args):
        self.rs_idx  = args.rs_idx
        self.num_iters = args.num_iters
        self.model_tag = args.model_tag
        self.dataset_type = args.dataset_type
        self.always_write_meta = args.wr_meta

        # Model params
        # ============
        self.seed = args.seed
        self.restore = args.restore
        self.channels  = args.channels
        self.var_scope = args.var_scope

        # Format naming and paths
        # =======================
        self.assign_names()
        self.make_model_dirs()

        # Init model vars
        # ===============
        self.initialize_params()

        # Paths

    def initialize_session(self):
        sess_kwargs = {}
        #==== Check for GPU
        if tf.test.is_gpu_available(cuda_only=True):
            gpu_frac= 0.85
            gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
            sess_kwargs['config'] = tf.ConfigProto(gpu_options=gpu_opts)

        #==== initialize session
        self.sess = tf.InteractiveSession(**sess_kwargs)



    def initialize_params(self):
        #restore = self.restore
        pass




    def initialize_graph(self):
        """ tf.train.Saver must be initialized AFTER the computational graph
            has been initialized via tf.global_variables_initializer
        """
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.restore:
            self.restore_model_parameters()


    def assign_names(self):
        self.start_time = time.time()
        zidx  = self.rs_idx
        dset  = self.dataset_type
        mname = self.model_name_format.format(dset, *zidx)
        if self.model_tag != '':
            mname = f'{mname}_{self.model_tag}'
        self.model_name = mname

        #==== Cube file names
        self.cube_name_truth = self.cube_name_format.format(*zidx, 'truth')
        self.cube_name_pred = self.cube_name_format.format(*zidx, 'prediction')


    def assign_pathing(self):
        """ Pathing to directories for this model """
        # Base path
        # ---------
        utils_path = os.path.abspath(os.path.dirname(__file__))
        proj_path = '/'.join(utils_path.split('/')[:-1])
        self.project_path = proj_path
        #==== model path
        epath = f'{proj_path}/{self.experiments_dir}/{self.model_name}'
        self.experiments_path = epath

        # Directory pathing
        # -----------------
        self.params_path  = f'{epath}/{self.params_dir}'
        self.results_path = f'{epath}/{self.results_dir}'

        # Create model dirs
        # -----------------
        for p in [self.params_path, self.results_path]:
            if not os.path.exists(p): os.makedirs(p)


    def restore_model_parameters(self):
        chkpt_state = tf.train.get_checkpoint_state(self.params_path)
        self.saver.restore(self.sess, chkpt_state.model_checkpoint_path)
        print(f'Restored trained model parameters from {self.params_path}')


    def save_model_error(self, error, training=False):
        name = 'training' if training else 'validation'
        path = f'{self.results_path}/error_{name}'
        np.save(path, error)
        print(f'Saved error: {path}')


    def save_model_cube(self, cube, ground_truth=False):
        name = self.cube_name_truth if ground_truth else self.cube_name_pred
        path = f'{self.results_path}/{name}'
        np.save(path, cube)
        print(f'Saved cube: {path}')


    def save_model_params(self, cur_iter):
        write_meta = self.always_write_meta
        if cur_iter == self.num_iters: # then training complete
            write_meta = True
            tsec = time.time() - self.start_time
            tmin  = tsec / 60.0
            thour = tmin / 60.0
            tS, tM, tH = f'{tsec: .3f}${tmin: .3f}${thour: .3f}'.split('$')
            print(f'Training complete!\n est. elapsed time: {tH}h, or {tM}m')
        step = cur_iter + 1
        self.saver.save(self.sess, self.experiments_path,
                        global_step=step, write_meta_graph=write_meta)


    def print_checkpoint(self, step, err):
        print(f'Checkpoint {step + 1 :>5} :  {err:.6f}')


    def print_evaluation_results(self, err):
        zx, zy = self.rs_idx
        #==== Statistics
        err_avg = np.mean(err)
        err_std = np.stdv(err)
        err_median = np.median(err)

        #==== Text
        title = f'\n# Evaluation Results:\n# {"="*78}'
        body = [f'# Location error : {:>3} ---> {:<3}',
                f'  median : {err_median : .5f}',
                f'    mean : {err_avg : .5f}',
                f'         +- {err_std : .4f} stdv',]
        #==== Print results
        print(title)
        for b in body:
            print(b)
