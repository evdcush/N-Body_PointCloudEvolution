''' notebook for visualizing results in matplotlib
'''
import os, code, sys, time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf

#=============================================================================
# data vars
#=============================================================================
REDSHIFTS_ZUNI = [9.0000, 4.7897, 3.2985, 2.4950, 1.9792, 1.6141, 1.3385,
                 1.1212, 0.9438, 0.7955, 0.6688, 0.5588, 0.4620, 0.3758,
                 0.2983, 0.2280, 0.1639, 0.1049, 0.0505, 0.0000]

# paths
dpath = './X32_11-19_{}.npy'
spath = './Imgs/'
if not os.path.exists(spath): os.makedirs(spath)


# load fn
def load_cube(mname, true_data=False):
    if true_data:
        return np.load(dpath.format(mname,'true')) # dont need redshift vec
    else:
        return np.load(dpath.format(mname, 'prediction'))

# model names
mname = 'ScaledLoss_ShiftInvariant'

# redshift vars
redshift_steps = [11, 15, 19] # reverse sorted indices into redshifts. redshifts[19] == redshifts[-1] == 0.0000
redshifts = [REDSHIFTS_ZUNI[i] for i in redshift_steps] # actual redshift values
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1

# load
X_true = np.load(dpath.format('true'))
X_pred = np.load(dpath.format('prediction'))
test_error = np.load(dpath.format('loss_validation'))


#=============================================================================
# plot fns
#=============================================================================
AX_LABEL_SIZE = 20
BOUND = 0.1

def split_xyz(x, vel=False):
    return np.split(x, 3, axis=-1)

def get_samples(x_true, x_hat, rs_idx, sample_idx):
    j = sample_idx
    rs = rs_idx
    xt = x_true[rs, j]
    xh =  x_hat[rs-1, j]
    return xt, xh

def plot_3D_pointcloud_ax(ax, x_true, x_hat, rs_idx, sample_idx, pt_size=.9):
    xt, xh = get_samples(x_true, x_hat, rs_idx, sample_idx)
    xt_xyz = split_xyz(xt[...,:3])
    xh_xyz = split_xyz(xh[...,:3])

    # plot
    ax.scatter(*xt_xyz, s=pt_size, c='r', alpha=0.5)
    ax.scatter(*xh_xyz, s=pt_size, c='b', alpha=0.5)

    # Axis labels
    ax.set_xlabel('X', size=AX_LABEL_SIZE)
    ax.set_ylabel('Y', size=AX_LABEL_SIZE)
    ax.set_zlabel('Z', size=AX_LABEL_SIZE)

    # Legend labels
    labels = ['Truth', 'Pred']
    leg = ax.legend(labels, loc=(0.8,0.82))
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        line.set_linewidth(4)
        text.set_fontsize('large')


def plot_quiver_axes(ax, coo, vel, **quiver_kwargs):
    ax.quiver(*coo, *vel, **quiver_kwargs)

def get_mask(x, bound=BOUND):
    xtmp = x[...,:3]
    lower, upper = bound, 1-bound
    mask1 = np.logical_and(xtmp[:,0] < upper, xtmp[:,0] > lower)
    mask2 = np.logical_and(xtmp[:,1] < upper, xtmp[:,1] > lower)
    mask3 = np.logical_and(xtmp[:,2] < upper, xtmp[:,2] > lower)
    mask = mask1 * mask2 * mask3
    mask_nz = np.nonzero(mask)[0]
    return mask_nz

def plot_3D_quiver_ax(ax, x_true, x_hat, rs_idx, sample_idx):
    xt, xh = get_samples(x_true, x_hat, rs_idx, sample_idx)
    xin = x_true[rs_idx-1, sample_idx]
    mask = get_mask(xin)

    # mask
    xt  = xt[mask]
    xh  = xh[mask]
    xin = xin[mask]

    # data vecs
    xt_coo  = xt[...,:3]
    xh_coo  = xh[...,:3]
    xin_coo = xin[...,:3]
    xin_vel = xin[...,3:]

    # arrows
    arrow_input = split_xyz(xin_coo) + split_xyz(xin_vel)
    arrow_truth = split_xyz(xt_coo ) + split_xyz(xt_coo - xin_coo)
    arrow_pred  = split_xyz(xh_coo ) + split_xyz(xh_coo - xin_coo)

    # plot
    quiver_kwargs = {'pivot': 'middle', 'linewidths': 0.5, }#'alpha': 0.4,}

    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    ax.quiver(*arrow_input, **quiver_kwargs, color='g', alpha=0.4, length=0.01)
    ax.quiver(*arrow_truth, **quiver_kwargs, color='r', alpha=0.2) # red a bit too strong
    ax.quiver(*arrow_pred,  **quiver_kwargs, color='b', alpha=0.4)

    # Axis labels
    ax.set_xlabel('X', size=AX_LABEL_SIZE)
    ax.set_ylabel('Y', size=AX_LABEL_SIZE)
    ax.set_zlabel('Z', size=AX_LABEL_SIZE)

    # Legend labels
    labels = ['Input', 'Truth', 'Pred']
    leg = ax.legend(labels, loc=(0.8,0.82))
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        line.set_linewidth(4)
        text.set_fontsize('large')


#=============================================================================
# graph ops
#=============================================================================
# Figure/plot properties
k = 10
fsize = (k*num_rs_layers,k) # (w, h)
particle_size = (.8,.8)

xt = np.copy(X_true)
xh = np.copy(X_pred)

sample_idx = 39 #185
#rs_idx = 4
#idx_tup = (rs_idx, sample_idx)
"""
Pyplot subplot numbers:
    - 111: 1x1 grid, first subplot
    - 234: 2x3 grid, 4th subplot
"""

plt.close('all')
fig = plt.figure(figsize=fsize)
num_cols = num_rs_layers
subplots_idx = [int('1{}{}'.format(num_cols, i)) for i in range(1, num_rs)]
rs_idx = [i for i in range(1, num_rs)]



def plot_particles(cube_idx):
    for subplot, zi in zip(subplots_idx, rs_idx):
        print('ARROW redshifts[{}] = {}'.format(zi, redshifts[zi]))
        median_error = np.median(test_error[:,zi-1])
        ax_title = 'Redshift {:.4f}\nMedian error: {:.5f}'.format(redshifts[zi], median_error)
        ax = fig.add_subplot(subplot, projection='3d')
        ax.set_title(ax_title, pad=0.3)
        plot_3D_pointcloud_ax(ax, xt, xh, zi, cube_idx)
    ftitle = '{} Point Cloud'.format(mname)
    fig.suptitle(ftitle, size=24)



def plot_arrows(cube_idx):
    for subplot, zi in zip(subplots_idx, rs_idx):
        print('ARROW redshifts[{}] = {}'.format(zi, redshifts[zi]))
        median_error = np.median(test_error[:,zi-1])
        ax_title = 'Redshift {:.4f}\nMedian error: {:.5f}'.format(redshifts[zi], median_error)
        ax = fig.add_subplot(subplot, projection='3d')
        ax.set_title(ax_title, pad=0.3)
        plot_3D_quiver_ax(ax, xt, xh, zi, cube_idx)
    ftitle = '{} Displacement'.format(mname)
    fig.suptitle(ftitle, size=24)

plot_arrows(sample_idx)

plt.tight_layout()
#fig.savefig(spath + 'Scaled_no_mult2', dpi=1000, bbox_inches='tight') # warning, this makes a huge image
#fig.savefig(spath+ 'test')
print('finished')

plt.show()
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
