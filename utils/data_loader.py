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
    path_simu  = os.environ['HOME'] + '/.Data/nbody_simulations'
    uni_format = '/uniform/X_{:.4f}_.npy'
    ZA_format  = '/ZA/ZA_{}.npy'
    def __init__(self, args):
        self.rs_idx = args.rs_idx
        self.batch_size = args.batch_size
        self.dataset_type = args.dataset_type
        self.num_eval_samples = args.num_eval_samples
        self.assign_simulation_type()

    def assign_simulation_type(self):
        """ Assigns attributes rs_val and path based on simulation type
        """
        simu_type = self.dataset_type
        if simu_type == 'uni':
            cube_type = self.uni_format
            self.rs_val = [UNI_REDSHIFTS[i] for i in self.rs_idx]
        else:
            cube_type = self.ZA_format
            self.rs_val = [ZA_STEPS[i] for i in self.rs_idx]
        self.path = f'{self.path_simu}{cube_type}'

    def split_dataset(self):
        """ split dataset into training and evaluation sets
        Both simulation datasets have their sample indices on
        the 1st axis
        """
        num_val = self.num_eval_samples
        np.random.seed(self.seed)
        ridx = np.random.permutation(self.X.shape[1])
        self.X_train, self.X_test = np.split(self.X[:, ridx], [-num_val], axis=1)
        self.X = None # reduce memory overhead

    def get_minibatch(self):
        """ randomly select training minibatch from dataset """
        batch_size = self.batch_size
        batch_idx = np.random.choice(self.X_train.shape[1], batch_size)
        x_batch = np.copy(self.X_train[:, batch_idx])
        return x_batch

    @staticmethod
    def normalize_uni(X):
        X[...,:3] = X[...,:3] / 32.0
        return X

    def load_simulation_cube(self, idx):
        #==== Format path to cube file
        val = self.rs_val[idx]
        cube_path = self.path.format(val)
        print(f'Loading cube from {cube_path}')
        #==== Load cube
        cube = np.load(cube_path).astype(np.float32)
        X = np.expand_dims(cube, 0)
        return X

    def load_simulation_data(self):
        """ Loads all simulation cubes for the
            training and evaluation datasets

        # Dataset dims
        uniform : (num_rs, 1000, 32**3, 6)
             ZA : (num_rs, 1000, 32, 32, 32, 19)
        """
        for v_idx in range(len(self.rs_idx)):
            #==== concat all cubes after first
            if v_idx == 0:
                X = self.load_simulation_cube(v_idx)
                continue
            X = np.concatenate([X, self.load_simulation_cube(v_idx)], axis=0)
        self.X = X
        print('Dataset successfully loaded')






