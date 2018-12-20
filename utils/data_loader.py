import os
import numpy as np

UNI_REDSHIFTS = [9.0000, 4.7897, 3.2985, 2.4950, 1.9792, 1.6141, 1.3385,
                 1.1212, 0.9438, 0.7955, 0.6688, 0.5588, 0.4620, 0.3758,
                 0.2983, 0.2280, 0.1639, 0.1049, 0.0505, 0.0000]

ZA_STEPS = ['001', '002', '003', '004', '005',
            '006', '007', '008', '009', '010']
"""

# ZA Data Features
# ================
For each simulation, the shape of data is 32*32*32*19.

32*32*32 is the number of particles and,
they are on the uniform 32*32*32 grid.


## The meaning of the 19 columns is as follows:

oneSimu[...,  1:4] : ZA displacements (Dx,Dy,Dz)
oneSimu[...,  4:7] : 2LPT displacements
oneSimu[..., 7:10] : FastPM displacements
oneSimu[...,10:13] : ZA velocity
oneSimu[...,13:16] : 2LPT velocity
oneSimu[...,16:19] : FastPM velocity

"""


class Dataset:
    seed = 12345  # for consistent splits
    def __init__(self, args):
        self.dataset_type = args.dataset_type
        self.batch_size = args.batch_size
        self.num_eval_samples = args.num_eval_samples
        self.simulation_data_path = args.simulation_data_path
        self.z_idx = args.z_idx[self.dataset_type]

        #=== Load data
        filenames = [self.fname.format(self.cube_steps[z]) for z in self.z_idx]
        paths = [f'{self.simulation_data_path}/{fname}' for fname in filenames]
        self.load_simulation_data(paths) # assigns self.X

    def split_dataset(self):
        """ split dataset into training and evaluation sets
        Both simulation datasets have their sample indices on
        the 1st axis
        """
        num_val = self.num_eval_samples
        np.random.seed(self.seed)
        ridx = np.random.permutation(self.X.shape[1])
        self.X_train, self.X_test = np.split(self.X[:, ridx], [-num_val], axis=1)
        #self.X = None # reduce memory overhead

    def get_minibatch(self):
        """ randomly select training minibatch from dataset """
        batch_size = self.batch_size
        batch_idx = np.random.choice(self.X_train.shape[1], batch_size)
        x_batch = np.copy(self.X_train[:, batch_idx])
        return x_batch

    def normalize(self):
        raise NotImplementedError

    def load_simulation_data(self, paths):
        for i, p in enumerate(paths):
            if i == 0:
                X = np.expand_dims(np.load(p), 0)
                continue
            X = np.concatenate([X, np.expand_dims(np.load(p), 0)], axis=0)
        self.X = X

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class ZA_Dataset(Dataset):
    fname = '/ZA/ZA_{}.npy'
    cube_steps = ZA_STEPS
    def __init__(self, args):
        super().__init__(args)
        self.get_za_fpm_data()
        #self.normalize()
        self.split_dataset()

    def get_za_fpm_data(self):
        """ NO NORMALIZATION """
        #=== formatting data
        reshape_dims = (1, 1000, 32**3, 3)

        #=== get ZA cubes
        ZA_disp = (self.X[...,1:4]).reshape(*reshape_dims)
        ZA_vel  = self.X[...,10:13].reshape(*reshape_dims)
        X_ZA = np.concatenate([ZA_disp, ZA_vel], axis=-1)

        #=== get FastPM cubes
        FPM_disp = (self.X[...,7:10]).reshape(*reshape_dims)
        FPM_vel  = self.X[...,16:19].reshape(*reshape_dims)
        X_FPM = np.concatenate([FPM_disp, FPM_vel], axis=-1)

        #=== Concat ZA and FastPM together, like typical redshift format
        self.X = np.concatenate([X_ZA, X_FPM], axis=0) # (2, 1000, 32**3, 6)


    def normalize(self):
        """ convert to positions and concat respective vels
        For each simulation, the shape of data is 32*32*32*19.

        32*32*32 is the number of particles and,
        they are on the uniform 32*32*32 grid.

        ## The meaning of the 19 columns is as follows:
        oneSimu[...,  1:4] : ZA displacements (Dx,Dy,Dz)
        oneSimu[...,  4:7] : 2LPT displacements
        oneSimu[..., 7:10] : FastPM displacements
        oneSimu[...,10:13] : ZA velocity
        oneSimu[...,13:16] : 2LPT velocity
        oneSimu[...,16:19] : FastPM velocity
        """
        #=== formatting data
        reshape_dims = (1, 1000, 32**3, 3)
        mrng = range(2,130,4)
        q = np.einsum('ijkl->kjli',np.array(np.meshgrid(mrng, mrng, mrng)))
        # q.shape = (32, 32, 32, 3)

        #=== get ZA cubes
        ZA_pos = (self.X[...,1:4] + q).reshape(*reshape_dims)
        ZA_vel = self.X[...,10:13].reshape(*reshape_dims)
        X_ZA = np.concatenate([ZA_pos, ZA_vel], axis=-1)

        #=== get FastPM cubes
        FPM_pos = (self.X[...,7:10] + q).reshape(*reshape_dims)
        FPM_vel = self.X[...,16:19].reshape(*reshape_dims)
        X_FPM = np.concatenate([FPM_pos, FPM_vel], axis=-1)

        #=== Concat ZA and FastPM together, like typical redshift format
        self.X = np.concatenate([X_ZA, X_FPM], axis=0) # (2, 1000, 32**3, 6)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class Uni_Dataset(Dataset):
    fname = '/uniform/X_{:.4f}_.npy'
    cube_steps  = UNI_REDSHIFTS
    def __init__(self, args):
        super().__init__(args)
        filenames = [self.name_format(self.cube_steps[z]) for z in self.z_idx]
        paths = [f'{self.simulation_data_path}/{fname}' for fname in filenames]
        self.load_simulation_data(paths) # assigns self.X
        self.split_dataset()

    def normalize(self):
        self.X[...,:3] = self.X[...,:3] / 32.0

#------------------------------------------------------------------------------

def get_dataset(args):
    dset = args.dataset_type
    return ZA_Dataset(args) if dset == 'ZA' else Uni_Dataset(args)
