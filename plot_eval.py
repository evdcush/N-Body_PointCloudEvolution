import time, code, os
import numpy as np
#import pylab as plt
from matplotlib import pyplot as plt

#plt.style.use('ggplot')
#plt.style.use('bmh')
#plt.ion()
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
# Pseudo-globals
# ========================================
SAVE_DIR = './HistPlots/'

Model_path = './Model/'
Cube_fname = '/Cubes/X32_{}-{}_{}.npy'
base_name = 'ShiftInv_single_2coeff_7K_ZG_{}-{}'

REDSHIFTS = [9.0000, 4.7897, 3.2985, 2.4950, 1.9792, 1.6141, 1.3385,
             1.1212, 0.9438, 0.7955, 0.6688, 0.5588, 0.4620, 0.3758,
             0.2983, 0.2280, 0.1639, 0.1049, 0.0505, 0.0000]

#trained_redshifts = [(15, 19), (16, 19), (17, 19), (18, 19), (12, 19), (10, 19), (7, 19), (3, 19), (3,7), (0,1), (11,15)]


# Data IO Functions
# ========================================
def load_cube(zx, zy, truth=False, model_name=None):
    if model_name is None:
        model_name = base_name
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
    xtmp = x[...,:3]
    lower, upper = bound, 1-bound
    mask1 = np.logical_and(xtmp[...,0] < upper, xtmp[...,0] > lower)
    mask2 = np.logical_and(xtmp[...,1] < upper, xtmp[...,1] > lower)
    mask3 = np.logical_and(xtmp[...,2] < upper, xtmp[...,2] > lower)
    mask = mask1 * mask2 * mask3
    mask_nz = np.nonzero(mask)[0]
    return mask_nz

def mask_data(x_in, x_truth, x_pred):
    # Reshape for masking entire cube
    x_in_flat    =    x_in.reshape([-1, 6])
    x_truth_flat = x_truth.reshape([-1, 6])
    x_pred_flat  =  x_pred.reshape([-1, x_pred.shape[-1]]) # (-1, 3) or (-1, 6)
    mask_nz = get_mask(x_in_flat) # flattened

    # Mask data
    loc_truth = np.copy(x_truth_flat[mask_nz, :3])
    loc_pred  = np.copy(x_pred_flat[ mask_nz, :3])
    loc_input = np.copy(x_in_flat[   mask_nz, :3])
    vel_input = np.copy(x_in_flat[   mask_nz, 3: ])

    return loc_truth, loc_pred, loc_input, vel_input

def load_formatted_cubes(rs_pair):
    zx, zy = rs_pair

    # Load Truth
    ground_truth = load_cube(zx, zy, truth=True)
    X_input = ground_truth[0]
    X_truth = ground_truth[1]
    ground_truth = None

    # Load Pred
    X_pred = load_cube(zx, zy)
    # Get masked, formatted data
    loc_truth, loc_pred, loc_input, vel_input = mask_data(X_input, X_truth, X_pred)
    # Generate moving-along-velocity linear "model prediction"
    timestep = get_timestep(vel_input, loc_input, loc_truth)
    displacement = vel_input * timestep
    loc_Vlinear = loc_input + displacement

    return loc_input, vel_input, loc_truth, loc_pred, loc_Vlinear, timestep

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

def get_timestep(vel_in, loc_in, loc_out):
    diff = loc_out - loc_in
    timestep = np.linalg.lstsq(vel_in.ravel()[:,None], diff.ravel())[0]
    return timestep

def l2_dist(loc_truth, loc_hat):
    return np.linalg.norm(loc_truth - loc_hat, axis=-1)


# Plot vars
# ========================================
alpha = .5
Vlin_label = 'vel linear'
pred_label = 'deep model'
Vlin_c = 'r'
pred_c = 'b'
'''
# Load Data
# ========================================
# Data vars
rs_tup_index = -2
zx, zy = trained_redshifts[rs_tup_index]

# Load Truth
ground_truth = load_cube(zx, zy, truth=True)
X_input = ground_truth[0]
X_truth = ground_truth[1]
ground_truth = None

# Load Pred
X_pred = load_cube(zx, zy)

# Format/mask data for plots
# ========================================
# Reshape for masking entire cube
x_input = X_input.reshape([-1, 6])
x_truth = X_truth.reshape([-1, 6])
x_pred  =  X_pred.reshape([-1, X_pred.shape[-1]]) # (-1, 3) or (-1, 6)
mask_nz = get_mask(x_input) # flattened
#
# Mask data
loc_truth = np.copy(x_truth[mask_nz,  :3])
loc_pred  = np.copy(x_pred[ mask_nz,  :3])
loc_input = np.copy(x_input[mask_nz,  :3])
vel_input = np.copy(x_input[mask_nz, 3: ])
#
# Calculate distances for hist
# ========================================
# Generate moving-along-velocity linear "model prediction"
timestep = get_timestep(vel_input, loc_input, loc_truth)
displacement = vel_input * timestep
loc_Vlinear = loc_input + displacement
#
# Get l2 distance to truth
l2_dist_Vlinear = l2_dist(loc_truth, loc_Vlinear)
l2_dist_pred    = l2_dist(loc_truth, loc_pred)
#
# Plot single hist
# ========================================
def plot_hist(dist_Vlin, dist_pred, rs_idx, subplot_idx=None):
    # Get hist bins from statistics on l2 distance
    bins = get_bins(dist_Vlin)

    # Get medians
    q50_Vlin = np.median(dist_Vlin)
    q50_pred = np.median(dist_pred)

    # format labels
    label_Vlin = '{}: {:.6f}'.format(Vlin_label, q50_Vlin)
    label_pred = '{}: {:.6f}'.format(pred_label, q50_pred)

    # plot hist
    plt.hist(dist_Vlin, bins= bins, label=label_Vlin, color=Vlin_c, alpha=alpha)
    plt.hist(dist_pred, bins= bins, label=label_pred, color=pred_c, alpha=alpha)

    # Add some graph details
    zx, zy = rs_idx
    rsx, rsy = REDSHIFTS[zx], REDSHIFTS[zy]
    title = 'Error, single-step {:>2}-{:>2}: {:.4f} --> {:.4f}'.format(zx, zy, rsx, rsy)
    plt.title(title)
    plt.xlabel('Distance (L2)')
    L = plt.legend()
    plt.setp(L.texts, family='monospace')
    plt.grid(True, alpha=0.5, ls='--')
    plt.tight_layout()

#plot_hist(l2_dist_Vlinear, l2_dist_pred, (zx, zy))
'''
# Plot multi-hist
# ========================================
def plot_hist_ax(dist_Vlin, dist_pred, rs_idx, subplot_idx):
    # Get hist bins from statistics on l2 distance
    bins = get_bins(dist_Vlin)

    # Get medians
    q50_Vlin = np.median(dist_Vlin)
    q50_pred = np.median(dist_pred)

    # format labels
    label_Vlin = '{}: {:.6f}'.format(Vlin_label, q50_Vlin)
    label_pred = '{}: {:.6f}'.format(pred_label, q50_pred)

    # plot hist
    ax = fig.add_subplot(subplot_idx)
    plt.hist(dist_Vlin, bins= bins, label=label_Vlin, color=Vlin_c, alpha=alpha)
    plt.hist(dist_pred, bins= bins, label=label_pred, color=pred_c, alpha=alpha)

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
        loc_truth, loc_pred, loc_input, vel_input = mask_data(X_input, X_truth, X_pred)

        # Generate moving-along-velocity linear "model prediction"
        timestep = get_timestep(vel_input, loc_input, loc_truth)
        displacement = vel_input * timestep
        loc_Vlinear = loc_input + displacement

        # Get l2 distance to truth
        l2_dist_Vlinear = l2_dist(loc_truth, loc_Vlinear)
        l2_dist_pred    = l2_dist(loc_truth, loc_pred)

        # Plot hist
        plot_hist_ax(l2_dist_Vlinear, l2_dist_pred, pair, cur_splot_idx)

rs1 = [(10, 19), (12, 19), (15, 19), (16, 19), (17, 19), (18, 19), ]#(7, 19), (3, 19), (3,7), (0,1), (11,15)]
rs2 = [(3,7), (0,1), (11,15)]
'''
#cur_rs = rs1
cur_rs = rs1 + rs2

nr = 3#1
nc = 3#len(cur_rs)
#splot_idx = [int('1{}{}'.format(num_cols, i)) for i in range(1, num_cols+1)]
splot_idx = [int('{}{}{}'.format(nr, nc, i)) for i in range(1, len(cur_rs)+1)]

j = 5
#fsize = ((j+1)*len(cur_rs), j)
fsize = ((j+1)*nc, j*nr)
fig = plt.figure(figsize=fsize)
#plot_hist_ax(l2_dist_Vlinear, l2_dist_pred, (zx, zy), 111)

plot_multi(cur_rs, splot_idx)
#plt.suptitle('')
#fig.suptitle('Comparison of deep model (blue) against moving-along-velocity')
plt.tight_layout()



#plt.show()
save_plot(0, 19)
'''

zx = 11
zy = 15
rs_pair = (zx, zy)
rsx, rsy = REDSHIFTS[zx], REDSHIFTS[zy]
#loc_input, vel_input, loc_truth, loc_pred, loc_Vlinear, timestep = load_formatted_cubes(rs_pair)
cubes = load_formatted_cubes(rs_pair)
timestep = cubes[-1][0]
print('{:>2}, {:>2}: {:.4f} --> {:.4f}, TIMESTEP: {:.6f}'.format(zx, zy, rsx, rsy, timestep))
