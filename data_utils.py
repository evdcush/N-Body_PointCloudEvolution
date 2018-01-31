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


#=============================================================================
# Data utils
#=============================================================================
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

def load_datum(n_P, redshift, normalize_data=False):
    """ loads two redshift datasets from proper data directory

    Args:
        redshift: (float) redshift
        n_P: (int) base of number of particles (n_P**3 particles)
    """
    N_P = 10000 if n_P == 32 else 1000
    glob_paths = glob.glob(DATA_PATH.format(N_P, 'xv', redshift))
    X = read_sim(glob_paths, n_P)
    if normalize_data:
        X = normalize(X)
    return X

def load_data(n_P, *args, **kwargs):
    """ loads datasets from proper data directory
    # note: this function is redundant

    Args:
        n_P: (int) base of number of particles (n_P**3 particles)
    """
    data = []
    for redshift in args:
        x = load_datum(n_P, redshift, **kwargs)
        data.append(x)
    return data

def normalize(X_in, scale_range=(0,1)):
    """ Normalize features
    coordinates are rescaled to (0,1)
    velocities normalized by mean/std    
    """
    xp = chainer.cuda.get_array_module(X_in)
    x_r = xp.reshape(X_in, [-1,6])
    coo, vel = xp.split(x_r, [3], axis=-1)
    
    coo_min = xp.min(coo, axis=0)
    coo_max = xp.max(coo, axis=0)
    a,b = scale_range
    x_r[:,:3] = (b-a) * (x_r[:,:3] - coo_min) / (coo_max - coo_min) + a
    
    vel_mean = xp.mean(vel, axis=0)
    vel_std  = xp.std( vel, axis=0)    
    x_r[:,3:] = (x_r[:,3:] - vel_mean) / vel_std

    X_out = xp.reshape(x_r,X_in.shape).astype(xp.float32) # just convert to float32 here
    return X_out

def split_data_validation(X, Y, num_val_samples=200):
    """ split dataset into training and validation sets
    
    Args:        
        X, Y (ndarray): data arrays of shape (num_samples, num_particles, 6)
        num_val_samples (int): size of validation set
    """
    num_samples = X.shape[0]
    idx_list = np.random.permutation(num_samples)
    X, Y = X[idx_list], Y[idx_list]
    X_input, X_val = X[:-num_val_samples], X[-num_val_samples:]#np.split(X, [-num_val_samples])
    X_truth, Y_val = Y[:-num_val_samples], Y[-num_val_samples:]#np.split(Y, [-num_val_samples])
    return [(X_input, X_val), (X_truth, Y_val)]


def rotate_x(sin_t, cos_t, x, y, z):
    out_x = x
    out_y = cos_t * y - sin_t * z
    out_z = sin_t * y + cos_t * z
    return out_x, out_y, out_z

def rotate_y(sin_t, cos_t, x, y, z):
    out_x = cos_t * x + sin_t * z
    out_y = y
    out_z = -sin_t * x + cos_t * z
    return out_x, out_y, out_z

def rotate_z(sin_t, cos_t, x, y, z):
    out_x = cos_t * x - sin_t * y
    out_y = sin_t * x + cos_t * y
    out_z = z
    return out_x, out_y, out_z
    
def rotate_3D(thetas, input_data):
    """ Rotate data by random theta
    thetas and input_data will be split by D axis, each retaining it's split
    axis dim (1,)
    sin, cos are performed once at the caller level, and the callee rotate
    functions only perform the rotational algebra
    
    Args:
        thetas     (ndarray): (mb_size, 3) random numbers for rotation
        input_data (ndarray): (mb_size, n_P, 3) data to be rotated
    """
    xp = chainer.cuda.get_array_module(input_data) # xp either numpy or cupy, based on GPU usage
    thetas = xp.expand_dims(thetas, -1)
    
    idx = [1,2]
    xyz = xp.split(input_data, idx, axis=-1) # len 3 list, each of shape (mb_size, n_P, 1)    
    sin_t = xp.split(xp.sin(thetas), idx, axis=1) # len 3 list, each of shape (mb_size, 1, 1)
    cos_t = xp.split(xp.cos(thetas), idx, axis=1)
    
    rotated_z   = rotate_z(sin_t[0], cos_t[0], *xyz)
    rotated_yz  = rotate_y(sin_t[1], cos_t[1], *rotated_z)
    rotated_xyz = rotate_x(sin_t[2], cos_t[2], *rotated_yz)
    out_data = xp.concatenate(rotated_xyz, axis=-1)
    return out_data

def random_augmentation_rotate(batches):
    """ Randomly rotate data by theta
    This function is redundant. No need to call rotate_3D twice if we 
    spend a little more time on indexing, but for now it stays
    """
    xp = chainer.cuda.get_array_module(batches[0]) # xp either numpy or cupy, based on GPU usage
    mb_size = batches[0].shape[0]
    thetas = xp.random.rand(mb_size, 3) * np.pi
    out = []
    for tmp in batches:
        tmp_coo = rotate_3D(thetas, tmp[...,:3])
        tmp_vel = rotate_3D(thetas, tmp[...,3:])
        tmp = xp.concatenate((tmp_coo, tmp_vel), axis=-1).astype(xp.float32) # always remember float32 casting
        tmp = normalize(tmp, scale_range=(-.125, .125)) # normalize casts
        out.append(tmp)
    return out

def random_augmentation_shift(batches):
    xp = chainer.cuda.get_array_module(batches[0])
    batch_size = batches[0].shape[0]
    out = []
    rands = xp.random.rand(6)
    shift = xp.random.rand(batch_size,3)
    for tmp in batches:
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

def multi_random_augmentation_shift(batch):
    xp = chainer.cuda.get_array_module(batch)
    batch_size = batch.shape[1]
    rands = xp.random.rand(6)
    shift = xp.random.rand(1,batch_size,1,3)
    # shape (11, bs, n_P, 6)
    if rands[0] < .5:
        batch = batch[...,[1,0,2,4,3,5]]
    if rands[1] < .5:
        batch = batch[...,[0,2,1,3,5,4]]
    if rands[2] < .5:
        batch = batch[...,[2,1,0,5,4,3]]
    if rands[3] < .5:
        batch[...,0] = 1 - batch[...,0]
        batch[...,3] = -batch[...,3]
    if rands[4] < .5:
        batch[...,1] = 1 - batch[...,1]
        batch[...,4] = -batch[...,4]
    if rands[5] < .5:
        batch[...,2] = 1 - batch[...,2]
        batch[...,5] = -batch[...,5]
    batch_coo = batch[...,:3]
    batch_coo += shift
    gt1 = batch_coo > 1
    batch_coo[gt1] = batch_coo[gt1] - 1
    batch[...,:3] = batch_coo
    return batch

def next_multi_minibatch(X_in, batch_size, data_aug='shift'):
    index_list = np.random.choice(X_in.shape[1], batch_size)
    batches = X_in[:,index_list]
    if data_aug == 'shift':
        return multi_random_augmentation_shift(batches)
    else:
        return batches


def next_minibatch(in_list, batch_size, data_aug='shift'):
    num_samples, N, D = in_list[0].shape
    index_list = np.random.choice(num_samples, batch_size)
    batches = [in_list[k][index_list] for k in range(len(in_list))]
    if data_aug == 'shift':
        return random_augmentation_shift(batches)
    elif data_aug == 'rotate':
        #print('rotate aug')
        return random_augmentation_rotate(batches)
    else:
        return batches
    
    

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

def get_save_label(mname, mtype, theta_use, num_particles, zfX, zfY):
    zfX, zfY = str(zfX), str(zfY)
    zx = zfX[0] + zfX[-1]
    zy = zfY[0] + zfY[-1]
    theta_tag = 'L' if theta_use == 1 else ''
    # mname|mtype|theta_numparticles_zXzY_
    tag = '{}{}{}_{}_{}{}_'.format(mname, mtype, theta_tag, num_particles, zx, zy)
    return tag


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

def plot_training_curve(y, cur_iter, yclip=.0004, c='b', fsize=(12,6), title=None):
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('MSE')

    cur = str(cur_iter)
    yclipped = np.clip(y, 0, yclip) if yclip > 0 else y
    plt.grid(True)
    ax.plot(yclipped,c=c)
    if title is None:
        title = 'Iteration: {0}, loss: {1:.4}'.format(cur_iter, y[-1])
    ax.set_title(title)
    return fig

def save_plot(lh, save_path, mname, val=False):
    plt.clf()
    plt.figure(figsize=(16,8))
    plt.grid()
    fig_name = save_path + mname
    plot_title = mname
    if val:
        plot_title = plot_title + 'Validation'
        fig_name   = fig_name + 'Validation'
        label = 'mean val error: ' + str(np.mean(lh))
        color = 'r'        
    else:
        plot_title = plot_title + 'Training'
        fig_name   = fig_name + 'Training'
        label = 'training error'
        color = 'b'

        converged_value = np.median(lh[-150:])
        converge_line   = np.ones((lh.shape[-1])) * converged_value
        plt.plot(converge_line, c='orange', label='converge: {}'.format(converged_value)) 
        lh = np.clip(lh, 0, np.median(lh[:500]))
    plt.title(plot_title)
    plt.plot(lh, c=color, label=label)
    plt.legend()
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()
