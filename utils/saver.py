import os
import time
import tensorflow
import numpy


class ModelSaver:
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
        self.always_write_meta = args.always_write_meta

        # Model params
        # ============
        self.restore = args.restore

        # Format naming and paths
        # =======================
        self.assign_names()
        self.assign_pathing()


    def init_sess_saver(self):
        """ tensorflow.train.Saver must be initialized AFTER the computational graph
            has been initialized via tensorflow.global_variables_initializer
        """
        self.saver = tensorflow.train.Saver()
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


    def restore_model_parameters(self, sess):
        chkpt_state = tensorflow.train.get_checkpoint_state(self.params_path)
        self.saver.restore(sess, chkpt_state.model_checkpoint_path)
        print(f'Restored trained model parameters from {self.params_path}')


    def save_model_error(self, error, training=False):
        name = 'training' if training else 'validation'
        path = f'{self.results_path}/error_{name}'
        numpy.save(path, error)
        print(f'Saved error: {path}')


    def save_model_cube(self, cube, ground_truth=False):
        name = self.cube_name_truth if ground_truth else self.cube_name_pred
        path = f'{self.results_path}/{name}'
        numpy.save(path, cube)
        print(f'Saved cube: {path}')


    def save_model_params(self, cur_iter, sess):
        write_meta = self.always_write_meta
        if cur_iter == self.num_iters: # then training complete
            write_meta = True
            tsec = time.time() - self.start_time
            tmin  = tsec / 60.0
            thour = tmin / 60.0
            tS, tM, tH = f'{tsec: .3f}${tmin: .3f}${thour: .3f}'.split('$')
            print(f'Training complete!\n est. elapsed time: {tH}h, or {tM}m')
        step = cur_iter + 1
        self.saver.save(sess, self.params_path + '/chkpt',
                        global_step=step, write_meta_graph=write_meta)


    def print_checkpoint(self, step, err):
        print(f'Checkpoint {step + 1 :>5} :  {err:.6f}')


    def print_evaluation_results(self, err):
        zx, zy = self.rs_idx
        #==== Statistics
        err_avg = numpy.mean(err)
        err_std = numpy.std(err)
        err_median = numpy.median(err)

        #==== Text
        title = f'\n# Evaluation Results:\n# {"="*78}'
        body = [f'# Location error : {zx:>3} ---> {zy:<3}',
                f'  median : {err_median : .5f}',
                f'    mean : {err_avg : .5f} +- {err_std : .4f} stdv',]
        #==== Print results
        print(title)
        for b in body:
            print(b)
