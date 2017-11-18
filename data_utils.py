import numpy as np
import glob
import struct
import code
'''
Data utils
'''
DATA_PATH = '/home/evan/Data/nbody_simulations/N_{0}/DM*/{1}_dm.z=0{2}000'

def read_sim(file_list, n_P):
    """ reads simulation data from disk and returns

    Args:
        file_list: (list<str>) paths to files
        n_P: (int) number of particles base (n_P**3 particles)
    """
    num_particles = n_P**3
    dataset = []
    for file_name in file_list:
        this_set = []
        with open(file_name, "rb") as f:
            for i in range(num_particles*6):
                s = struct.unpack('=f',f.read(4))
                this_set.append(s[0])
        dataset.append(this_set)
    dataset = np.array(dataset).reshape([len(file_list),num_particles,6])  
    return dataset

def load_data(zA, zB, n_P):
    """ loads two redshift datasets from proper data directory

    Args:
        zA: (float) redshift
        zB: (float) redshift
        n_P: (int) base of number of particles (n_P**3 particles)
    """
    N_P = 10000 if n_P == 32 else 1000
    Apath = glob.glob(DATA_PATH.format(N_P, 'xv', zA))
    Bpath = glob.glob(DATA_PATH.format(N_P, 'xv', zB))
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    A = read_sim(Apath, n_P)
    B = read_sim(Bpath, n_P)
    return A,B


def normalize(X_in):
    """ Normalize features
    coordinates are rescaled to (0,1)
    velocities normalized by mean/std    
    """
    X_1 = np.reshape(X_in,[-1,6])
    coo_min, coo_max = np.min(X_1[:,:3],axis=0), np.max(X_1[:,:3],axis=0)
    #coo_mean, coo_std = np.mean(X_1[:,:3],axis=0), np.std(X_1[:,:3],axis=0)
    #X_1[:,:3] = (X_1[:,:3] - coo_mean) / (coo_std)
    v_mean, v_std = np.mean(X_1[:,3:],axis=0), np.std(X_1[:,3:],axis=0)
    X_1[:,:3] = (X_1[:,:3] - coo_min) / (coo_max - coo_min) # proper rescale function?
    #X_1[:,:3] = (X_1[:,:3] - coo_min) / (coo_max) # rescale fn originally used
    X_1[:,3:] = (X_1[:,3:] - v_mean) / v_std
    return np.reshape(X_1,[X_in.shape[0],X_in.shape[1],6])

def next_minibatch(in_list,batch_size):
    if all(len(i) == len(in_list[0]) for i in in_list) == False:   
        raise ValueError('Inputs do not have the same dimension')
    index_list = np.random.permutation(len(in_list[0]))[:batch_size]#np.random.randint(len(in_list[0]), size=batch_size)
    out = list()
    rands = np.random.rand(6)
    shift = np.random.rand(batch_size,3)
    for k in range(len(in_list)):
        tmp = in_list[k][index_list]
        if rands[0] < .5:
            tmp = tmp[:,:,[1,0,2,4,3,5]]
        if rands[1] < .5:
            tmp = tmp[:,:, [0,2,1,3,5,4]]
        if rands[2] < .5:
            tmp = tmp[:,:, [2,1,0,5,4,3]]
        if rands[3] < .5:
            tmp[:,:,0] = 1 - tmp[:,:,0]
            tmp[:,:,3] = -tmp[:,:,3]
        if rands[4] < .5:
            tmp[:,:,1] = 1 - tmp[:,:,1]
            tmp[:,:,4] = -tmp[:,:,4]
        if rands[5] < .5:
            tmp[:,:,2] = 1 - tmp[:,:,2]
            tmp[:,:,5] = -tmp[:,:,5]
            
        tmploc = tmp[:,:,:3]
        tmploc += shift[:,None,:]
        gt1 = tmploc > 1
        tmploc[gt1] = tmploc[gt1] - 1
        tmp[:,:,:3] = tmploc
        out.append(tmp)
    return out