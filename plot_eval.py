import time, code, os, sys
import numpy as np
#import pylab as plt
from matplotlib import pyplot as plt

''' # Notes
- Currently not looking at angles, since I was getting NaNs with the basic linalg func
- need to calculate timesteps for all possible redshifts
    - this might be better done in a separate script?
        - load cubes, same random split idx for validation
        - calculate timesteps for every possible redshift pair
'''


#plt.style.use('ggplot')
#plt.style.use('bmh')
#plt.ion()
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
# Pseudo-globals
# ========================================
SAVE_DIR = './HistPlots/'

Model_path = './Model/'
Cube_fname = '/Cubes/X32_{}-{}_{}.npy'
timesteps_fname = Model_path + 'timesteps.npy'
#base_name = 'ShiftInv_single_2coeff_7K_ZG_{}-{}'
base_name = 'Multi3_ShiftInv_7K_ZG_{}-{}'

REDSHIFTS = [9.0000, 4.7897, 3.2985, 2.4950, 1.9792, 1.6141, 1.3385,
             1.1212, 0.9438, 0.7955, 0.6688, 0.5588, 0.4620, 0.3758,
             0.2983, 0.2280, 0.1639, 0.1049, 0.0505, 0.0000]

#trained_redshifts = [(15, 19), (16, 19), (17, 19), (18, 19), (12, 19), (10, 19), (7, 19), (3, 19), (3,7), (0,1), (11,15)]


# Data IO Functions
# ========================================
def load_cube(zx, zy, model_name, truth=False):
    ctag = 'true' if truth else 'prediction'
    cube_name = Cube_fname.format(zx, zy, ctag)
    m_name = model_name.format(zx, zy)
    path = Model_path + m_name + cube_name
    return np.load(path)


def save_plot(zx, zy, save_fname=None):
    if save_fname is None:
        save_fname = 'Hist_{}-{}'.format(zx, zy)
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    plt.savefig(SAVE_DIR + save_fname, dpi=700, bbox_inches='tight')
    print('{} plot saved!'.format(save_fname))



# Data Manipulation Functions
# ========================================
def get_mask(x, bound=0.1):
    xtmp = x[...,:3].reshape([-1,3])
    lower, upper = bound, 1-bound
    mask1 = np.logical_and(xtmp[...,0] < upper, xtmp[...,0] > lower)
    mask2 = np.logical_and(xtmp[...,1] < upper, xtmp[...,1] > lower)
    mask3 = np.logical_and(xtmp[...,2] < upper, xtmp[...,2] > lower)
    mask = mask1 * mask2 * mask3
    mask_nz = np.nonzero(mask)[0]
    return mask_nz

def mask_data(x, mask):
    n,m,d = x.shape
    # Reshape for masking entire cube
    x_flat = x.reshape([-1, d])

    return np.copy(x_flat[mask,:])


# Alg/Plotting utils
# ========================================
def get_bins(dist):
    q = np.percentile(dist, 90)
    #mu = np.mean(dist)
    mu = np.median(dist)
    std = np.std(dist)
    mu, std = np.mean(dist), np.std(dist)
    lower, mu_up = 0., mu + 2*std
    upper = min(q, mu_up)
    return np.linspace(lower, upper, 500)

def angle(v1, v2):
    angle = np.degrees(np.arccos(np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))))
    #angle = np.degrees(np.arccos(np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))))
    return angle


def calculate_timestep(x_in, x_true):
    diff = x_true[...,:3] - x_in[...,:3]
    timestep = np.linalg.lstsq(x_in[...,3:].ravel()[:,None], diff.ravel())[0]
    return timestep

def get_linearVel_pred(x_in, timestep):
    displacement = x_in[...,3:] * timestep
    loc_Vlinear = x_in[...,:3] + displacement
    return loc_Vlinear

def l2_dist(x_true, x_hat):
    return np.linalg.norm(x_true[...,:3] - x_hat[...,:3], axis=-1)

# Plot vars
# ========================================
alpha = .5
LINEAR_VEL_LABEL = 'linear vel'
#pred_label = 'deep model'
#pred2_label = 'deep noisy model'
linVel_c = 'r'

plabels = ['deep vanilla', 'deep noisy']
pcolors = ['b', 'g']
#pred_c = 'b'
#pred2_c = 'g'

# Plot multi-hist
# ========================================
def label_hist_ax(ax, rs_idx, xlabel='Distance (L2)'):
    # Add some graph details
    zx, zy = rs_idx
    rsx, rsy = REDSHIFTS[zx], REDSHIFTS[zy]
    title = 'Error, single-step {:>2}-{:>2}: {:.4f} --> {:.4f}'.format(zx, zy, rsx, rsy)
    ax.set_title(title, size='medium', style='italic')
    ax.set_xlabel('Distance (L2)')
    leg = ax.legend()
    plt.setp(leg.texts, family='monospace', fontsize=11)
    for line in leg.get_lines():
        line.set_linewidth(1)
    ax.grid(True, alpha=0.5, ls='--')
    #plt.grid(True, alpha=0.5, ls='--')

def get_label(dist, tag=None):
    median = np.median(dist)
    if tag is None:
        tag = LINEAR_VEL_LABEL
    label = '{:>16}: {:.6f}'.format(tag, median)
    return label

def plot_hist_ax(dist_linVel, dist_preds, rs_idx, subplot_idx):
    # Get hist bins from statistics on l2 distance
    bins = get_bins(dist_linVel)

    # Get legend label
    label_linVel = get_label(dist_linVel)

    # Setup subplot and plot linear velocity
    ax = fig.add_subplot(subplot_idx)
    plt.hist(dist_linVel, bins= bins, label=label_linVel, color=linVel_c, alpha=alpha)

    # Plot all model predictions
    for i, dist_pred in enumerate(dist_preds):
        label_p = get_label(dist_pred, plabels[i])
        plt.hist(dist_pred, bins= bins, label=label_p, color=pcolors[i], alpha=alpha)

    # Add some graph details
    label_hist_ax(ax, rs_idx)

def plot_multi_single(X_truth, X_pred, rs_pairs, splot_idx):
    for i, pair in enumerate(rs_pairs):
        # Current redshift pair
        zx, zy = pair
        cur_splot_idx = splot_idx[i]

        # Get "input" and "truth"
        x_input = X_truth[i][0]
        x_truth = X_truth[i][1]
        x_pred  = X_pred[i]

        # Mask data
        mask = get_mask(x_input)
        x_input_masked = mask_data(x_input, mask)
        x_truth_masked = mask_data(x_truth, mask)
        x_pred_masked  = mask_data(x_pred,  mask)
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

        # Generate moving-along-velocity linear "model prediction"
        timestep = calculate_timestep(x_input_masked, x_truth_masked) #(x_in, x_true)
        linearVel_pred = get_linearVel_pred(x_input_masked, timestep) # (...,3)

        # Get l2 distance to truth
        dist_linVel = l2_dist(x_truth_masked, linearVel_pred)
        dist_pred = l2_dist(x_truth_masked, x_pred_masked)

        # Plot hist
        plot_hist_ax(dist_linVel, [dist_pred], pair, cur_splot_idx)



def plot_multiStep_comp(X_truth, X_preds, rs_pairs, splot_idx):
    for i, pair in enumerate(rs_pairs):
        # Current redshift pair
        zx, zy = pair
        cur_splot_idx = splot_idx[i]

        # Get "input" and "truth"
        x_input = X_truth[i]
        x_truth = X_truth[i+1]

        # Get model prediction
        #x_pred = X_preds[i]

        # Mask data
        mask = get_mask(x_input)
        x_input_masked = mask_data(x_input, mask)
        x_truth_masked = mask_data(x_truth, mask)
        #x_pred_masked  = mask_data(x_pred,  mask)

        # Generate moving-along-velocity linear "model prediction"
        timestep = calculate_timestep(x_input_masked, x_truth_masked) #(x_in, x_true)
        linearVel_pred = get_linearVel_pred(x_input_masked, timestep) # (...,3)

        # Get l2 distance to truth
        dist_linVel = l2_dist(x_truth_masked, linearVel_pred)

        # Get preds
        dist_preds = []
        for cur_pred in X_preds:
            x_pred = cur_pred[i]
            x_pred_masked = mask_data(x_pred, mask)
            dist_pred = l2_dist(x_truth_masked, x_pred_masked)
            dist_preds.append(dist_pred)

        # Plot hist
        plot_hist_ax(dist_linVel, dist_preds, pair, cur_splot_idx)




# Load data
# ========================================
#rs1 = [(10, 19), (12, 19), (15, 19), (16, 19), (17, 19), (18, 19), ]#(7, 19), (3, 19), (3,7), (0,1), (11,15)]
#rs2 = [(3,7), (0,1), (11,15)]
rs_multi3 = [(1,7), (7,13), (13,19)]
cur_rs = rs_multi3

#model_names = ['Multi3_ShiftInv_7K_ZG_{}-{}', 'Multi3_ShiftInv_7K_noisy018_ZG_{}-{}']
#preds_label = ['deep vanilla', 'deep noisy']
#preds_c = ['b', 'g']
#preds_attr = [(label, color, fname) for label, color, fname in zip(preds_label, preds_c, model_names)]
mname = 'sinv7K_ZG_{}-{}'


#X_truth = load_cube(1, 19, base_name, truth=True) # (4, 200, 32**3, 6)
#X_preds = [load_cube(1, 19, fname) for fname in model_names]
X_truth = [load_cube(zx, zy, mname, truth=True) for zx,zy in  cur_rs]
X_pred  = [load_cube(zx, zy, mname, truth=False) for zx,zy in cur_rs]



# Plot settings
# ========================================
# number of subplots
nr = 1
nc = len(cur_rs)
splot_idx = [int('{}{}{}'.format(nr, nc, i)) for i in range(1, len(cur_rs)+1)]

# figure
f = 5
fsize = ((f+1)*nc, f*nr)
fig = plt.figure(figsize=fsize)

plot_multi_single(X_truth, X_pred, cur_rs, splot_idx)
#fig.suptitle('Comparison of deep model (blue) against moving-along-velocity')
plt.tight_layout()
plt.show()



#plt.show()

'''
j = 5
#fsize = ((j+1)*len(cur_rs), j)
fsize = ((j+1)*nc, j*nr)
fig = plt.figure(figsize=fsize)
#plot_hist_ax(l2_dist_Vlinear, l2_dist_pred, (zx, zy), 111)

plot_multistep(cur_rs, splot_idx)
#plot_multi(cur_rs, splot_idx)
#plt.suptitle('')
#fig.suptitle('Comparison of deep model (blue) against moving-along-velocity')
plt.tight_layout()



#plt.show()
save_plot(0, 19)

def plot_multi(rs_pairs, splot_idx):
    for i, pair in enumerate(rs_pairs):
        zx, zy = pair
        cur_splot_idx = splot_idx[i]

        # Load Truth
        ground_truth = load_cube(zx, zy, truth=True)
        X_input = ground_truth[0]
        X_truth = ground_truth[1]
        ground_truth = None

        # Load Pred
        X_pred = load_cube(zx, zy)

        # Get masked, formatted data
        #loc_truth, loc_pred, loc_input, vel_input = mask_data(X_input, X_truth, X_pred)


        # Generate moving-along-velocity linear "model prediction"
        timestep = calculate_timestep(x_in, x_truth[...,:3])
        loc_velLinear = get_velLinear_pred(x_in, timestep)

        # Get l2 distance to truth
        l2_dist_Vlinear = l2_dist(loc_truth, loc_Vlinear)
        l2_dist_pred    = l2_dist(loc_truth, loc_pred)

        # Plot hist
        plot_hist_ax(l2_dist_Vlinear, l2_dist_pred, pair, cur_splot_idx)
'''

#zx = 11
#zy = 15
#rs_pair = (zx, zy)
#rsx, rsy = REDSHIFTS[zx], REDSHIFTS[zy]
#loc_input, vel_input, loc_truth, loc_pred, loc_Vlinear, timestep = load_formatted_cubes(rs_pair)
#cubes = load_formatted_cubes(rs_pair)
#timestep = cubes[-1][0]
#print('{:>2}, {:>2}: {:.4f} --> {:.4f}, TIMESTEP: {:.6f}'.format(zx, zy, rsx, rsy, timestep))
