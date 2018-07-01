''' notebook for visualizing results in matplotlib
'''
import os, code, sys, time, argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import kneighbors_graph
import tensorflow as tf

import utils
from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, RS_TAGS

#=============================================================================
# data vars
#=============================================================================
# paths
dpath = '../multi_9k/{}X32_11-19_{}.npy'
spath = './Imgs/'

# load fn
def load_cube(mname, true_data=False):
    if true_data:
        return np.load(dpath.format(mname,'true')) # dont need redshift vec
    else:
        return np.load(dpath.format(mname, 'prediction'))

# model names
pred_mname = 'SC_NOMU_'

# redshift vars
redshift_steps = [11, 15, 19] # reverse sorted indices into redshifts. redshifts[19] == redshifts[-1] == 0.0000
redshifts = [utils.REDSHIFTS_ZUNI[i] for i in redshift_steps] # actual redshift values
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1

# load
X_true = load_cube('', True)
X_pred = load_cube(pred_mname)


#=============================================================================
# plot fns
#=============================================================================
AX_LABEL_SIZE = 20
BOUND = 0.1
split_coo = lambda x: np.split(x[...,:3],3,axis=-1)
split_vel = lambda x: np.split(x[...,3:],3,axis=-1)

def split_xyz(x, vel=False):
    if vel:
        return split_vel(x)
    else:
        return split_coo(x)

def get_samples(x_true, x_hat, rs_idx, sample_idx):
    j = sample_idx
    rs = rs_idx
    xt = x_true[rs, j]
    xh =  x_hat[rs-1, j]
    return xt, xh

def plot_3D_pointcloud_ax(ax, x_true, x_hat, rs_idx, sample_idx, pt_size=(.9,.9), colors=('r','b'), fsize=(18,18)):
    xt, xh = get_samples(x_true, x_hat, rs_idx, sample_idx)
    xt_xyz = split_xyz(xt)
    xh_xyz = split_xyz(xh)
    # fig
    #fig = plt.figure(figsize=fsize)
    #ax = fig.add_subplot(111, projection='3d')
    # plot
    ax.scatter(*xt_xyz, s=pt_size[0], c=colors[0], alpha=0.5)
    ax.scatter(*xh_xyz, s=pt_size[1], c=colors[1], alpha=0.5)
    # label
    ax.set_xlabel('X', size=AX_LABEL_SIZE)
    ax.set_ylabel('Y', size=AX_LABEL_SIZE)
    ax.set_zlabel('Z', size=AX_LABEL_SIZE)
    #return fig

def plot_quiver_axes(ax, coo, vel, **quiver_kwargs):
    #ax.quiver(*coo, *vel, **quiver_kwargs)
    xc = coo[:,:1]
    yc = coo[:,1:2]
    zc = coo[:,2:3]
    xv = vel[:,:1]
    yv = vel[:,1:2]
    zv = vel[:,2:3]

    ax.quiver(xc, yc, zc, xv, yv, zv, **quiver_kwargs)

def get_mask(x, bound=BOUND):
    xtmp = x[...,:3]
    lower, upper = bound, 1-bound
    mask1 = np.logical_and(xtmp[:,0] < upper, xtmp[:,0] > lower)
    mask2 = np.logical_and(xtmp[:,1] < upper, xtmp[:,1] > lower)
    mask3 = np.logical_and(xtmp[:,2] < upper, xtmp[:,2] > lower)
    mask = mask1 * mask2 * mask3
    mask_nz = np.nonzero(mask)[0]
    return mask_nz

def mask_data(x, mask):
    return x[mask]

def plot_3D_quiver_ax(ax, x_true, x_hat, rs_idx, sample_idx, pt_size=(.9,.9), colors=('r','b'), fsize=(18,18)):
    xt, xh = get_samples(x_true, x_hat, rs_idx, sample_idx)
    xin = x_true[rs_idx-1, sample_idx]
    mask = get_mask(xin)

    # mask
    xt = mask_data(xt, mask)
    xh = mask_data(xh, mask)
    xin = mask_data(xin, mask)

    # data vecs
    #xt_coo  = split_xyz(xt)
    #xh_coo  = split_xyz(xh)
    #xin_coo = split_xyz(xin)
    xt_coo  = xt[...,:3]
    xh_coo  = xh[...,:3]
    xin_coo = xin[...,:3]
    xin_vel = xin[...,3:]

    # fig
    #fig = plt.figure(figsize=fsize)
    #ax = fig.add_subplot(111, projection='3d')

    # arrows
    arrow_input = (xin_coo, xin_vel)
    arrow_truth = (xt_coo, xt_coo - xin_coo)
    arrow_pred  = (xh_coo, xh_coo - xin_coo)

    # plot
    quiver_kwargs = {'pivot': 'middle', 'length': 1.0, 'normalize': False, 'alpha': 0.4, 'linewidths': 0.5, 'scale':2.3}
    #arrow_true = (split_xyz(xin), spl)
    #coo = split_xyz(x)
    #vel = split_xyz(x, vel=True)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    plot_quiver_axes(ax, *arrow_truth, **quiver_kwargs, color='r',)
    #plot_quiver_axes(ax, *arrow_pred,  **quiver_kwargs, color='b', scale=2.3)
    #plot_quiver_axes(ax, *arrow_input, **quiver_kwargs, color='g', scale=np.max(xin_vel))

    # label
    ax.set_xlabel('X', size=AX_LABEL_SIZE)
    ax.set_ylabel('Y', size=AX_LABEL_SIZE)
    ax.set_zlabel('Z', size=AX_LABEL_SIZE)
    #return fig

#=============================================================================
# graph ops
#=============================================================================

k = 3
fsize = (k*num_rs_layers,k) # (w, h)
#fsize = (k*4,k*2)
particle_size = (.8,.8)
truth_color = 'red'
pred_color = 'blue'
colors = (truth_color, pred_color)

xt = np.copy(X_true)
xh = np.copy(X_pred)

sample_idx = 39 #185
#rs_idx = 4
#idx_tup = (rs_idx, sample_idx)

plt.close('all')
#fig = plot_3D_pointcloud(xt, xh, *idx_tup, colors=colors, pt_size=particle_size, fsize=fsize)
fig = plt.figure(figsize=fsize)
"""
Pyplot subplot numbers:
 - 111: 1x1 grid, first subplot
 - 234: 2x3 grid, 4th subplot
"""
#subplots_arrow = [245,246,247,248]
#subplots_arrow = [141,142,143]
num_cols = num_rs_layers
subplots_arrow = [int('1{}{}'.format(num_cols, i)) for i in range(1, num_rs)]
#subplots_point = [241,242,243,244]
rs_indices = [i for i in range(1, num_rs)]
'''
for subplot, rs_idx in zip(subplots_point, rs_indices):
    print('POINT redshifts[{}] = {}'.format(rs_idx, redshifts[rs_idx]))
    ax_title = 'Redshift: {:.4f}'.format(redshifts[rs_idx])
    ax = fig.add_subplot(subplot, projection='3d')
    ax.title.set_text(ax_title)
    plot_3D_pointcloud_ax(ax, xt, xh, rs_idx, sample_idx, colors=colors, pt_size=particle_size, fsize=fsize)
'''
test_error = np.load('./multi_9k/{}ZG_11-19_loss_validation.npy'.format(pred_mname))
for subplot, rs_idx in zip(subplots_arrow, rs_indices):
    print('ARROW redshifts[{}] = {}'.format(rs_idx, redshifts[rs_idx]))
    med_err = np.median(test_error[:,rs_idx-1])
    ax_title = 'Redshift {:.4f}\nmedian error: {:.5f}'.format(redshifts[rs_idx], med_err)
    ax = fig.add_subplot(subplot, projection='3d')
    #ax.title.set_text(ax_title, (0.5, .8))
    ax.set_title(ax_title, pad=0.3)
    plot_3D_quiver_ax(ax, xt, xh, rs_idx, sample_idx, colors=colors, pt_size=particle_size, fsize=fsize)

#test_medians = [7.699121488258243e-05, 5.8502777392277494e-05, 5.3116607887204736e-05, 4.708175401901826e-05]
#mu = np.mean(test_medians)

mu = np.median(test_error[:,-1])
fig.suptitle("Scaled Loss, Single-step only training, 9K iters", size=24)

plt.tight_layout()
#plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
#plt.gca().view_init(20, 120)
#display.display(plt.gcf())
#display.clear_output(wait=True)
#time.sleep(0.0001)
'''
rotate = False
if rotate:
    for angle in range(0,360, 60):
        #fig.view_init(30, angle)
        plt.gca().view_init(30, angle)
        display.display(plt.gcf())
        display.clear_output(wait=True)
        time.sleep(0.0001)
'''
#fig.savefig(spath + 'test', dpi=2400,bbox_inches='tight') # warning, this makes a huge image
#fig.savefig(spath + 'IndividuallyTrained_arrows', dpi=1000, bbox_inches='tight') # warning, this makes a huge image

#fig.savefig(spath + 'Scaled_no_mult2', dpi=1000, bbox_inches='tight') # warning, this makes a huge image
#fig.savefig(spath+ 'test')
print('finished')

plt.show()
code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
