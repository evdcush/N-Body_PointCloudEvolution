import numpy as np
import cupy
import chainer
from chainer import cuda
import glob
import struct
import code
import chainer.serializers as serializers
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def load_datum(zA, n_P):
    """ loads two redshift datasets from proper data directory

    Args:
        zA: (float) redshift
        zB: (float) redshift
        n_P: (int) base of number of particles (n_P**3 particles)
    """
    N_P = 10000 if n_P == 32 else 1000
    Apath = glob.glob(DATA_PATH.format(N_P, 'xv', zA))
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    A = read_sim(Apath, n_P)
    return A

def normalize(X_in, cupy_out=False):
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
    out = np.reshape(X_1,[X_in.shape[0],X_in.shape[1],6])
    if cupy_out:
        out = cuda.to_gpu(out.astype(np.float32))
    return out


def next_minibatch(in_list,batch_size):
    if all(len(i) == len(in_list[0]) for i in in_list) == False:   
        raise ValueError('Inputs do not have the same dimension')
    index_list = np.random.permutation(len(in_list[0]))[:batch_size]#np.random.randint(len(in_list[0]), size=batch_size)
    out = []
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

def gpunext_minibatch(in_list,batch_size, xp=cupy):
    assert len(set([a.shape for a in in_list])) == 1
    M, N, D = in_list[0].shape
    index_list = xp.random.choice(M,batch_size)
    out = []
    rands = xp.random.rand(6)
    shift = xp.random.rand(batch_size,3)
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

def to_var_xp(lst_data):
    return [chainer.Variable(data) for data in lst_data]

def to_variable(lst_data, use_gpu=True):
    xp = cupy if use_gpu else np
    chainer_vars = []
    for data in lst_data:
        data = data.astype(xp.float32)
        if use_gpu: data = cuda.to_gpu(data)
        data_var = chainer.Variable(data)
        chainer_vars.append(data_var)
    return chainer_vars

def save_data_batches(batch_tuple, save_name):
    x_in, xt, xh = batch_tuple
    assert x_in.shape[0] == xt.shape[0] == xh.shape[0]
    np.save(save_name + 'input', x_in)
    np.save(save_name + 'truth', xt)
    np.save(save_name + 'hat'  , xh)
    print('data saved')

def save_model(mopt_tuple, save_name):
    model, opt = mopt_tuple
    serializers.save_npz(save_name + '.model', model)
    serializers.save_npz(save_name + '.state', opt)

#=============================================================================
# Plotting utils
#=============================================================================
def plot_3D_pointcloud(xt, xh, j, pt_size=(.9,.9), colors=('b','r'), fsize=(18,18), xin=None):
    xt_x, xt_y, xt_z = np.split(xt[...,:3], 3, axis=-1)
    xh_x, xh_y, xh_z = np.split(xh[...,:3], 3, axis=-1)
    
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xt_x[j], xt_y[j], xt_z[j], s=pt_size[0], c=colors[0])
    ax.scatter(xh_x[j], xh_y[j], xh_z[j], s=pt_size[1], c=colors[1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig

def plot_training_curve(y, cur_iter, yclip=.0004, c='b', poly=None, fsize=(16,10), title=None):
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('MSE')

    cur = str(cur_iter)
    yclipped = np.clip(y, 0, yclip) if yclip > 0 else y
    plt.grid(True)
    ax.plot(yclipped,c=c)
    if poly is not None:
        xvals = np.arange(cur_iter)
        pfit = np.poly1d(np.polyfit(xvals, yclipped, poly))
        ax.plot(poly(xvals), c='orange', linewidth=3)
    if title is None:
        title = 'Iteration: {0}, loss: {1:.4}'.format(cur_iter, y[-1])
    ax.set_title(title)
    return fig