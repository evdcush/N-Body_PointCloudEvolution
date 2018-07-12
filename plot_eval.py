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

trained_redshifts = [(15, 19), (16, 19), (17, 19), (18, 19), (12, 19), (10, 19), (7, 19), (3, 19), (3,7), (0,1), (11,15)]


# Data IO/Manipulation Functions
# ========================================
def load_cube(zx, zy, truth=False, model_name=None):
    if model_name is None:
        model_name = base_name
    ctag = 'true' if truth else 'prediction'
    cube_name = Cube_fname.format(zx, zy, ctag)
    m_name = model_name.format(zx, zy)
    path = Model_path + m_name + cube_name
    return np.load(path)

def save_plot(save_fname=None):
    if save_fname is None:
        save_fname = 'Hist_{}-{}'.format(zx, zy)
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    plt.savefig(SAVE_DIR + save_fname, dpi=700, bbox_inches='tight')
    print('{} plot saved!'.format(save_fname))

def get_mask(x, bound=0.1):
    xtmp = x[...,:3]
    lower, upper = bound, 1-bound
    mask1 = np.logical_and(xtmp[...,0] < upper, xtmp[...,0] > lower)
    mask2 = np.logical_and(xtmp[...,1] < upper, xtmp[...,1] > lower)
    mask3 = np.logical_and(xtmp[...,2] < upper, xtmp[...,2] > lower)
    mask = mask1 * mask2 * mask3
    mask_nz = np.nonzero(mask)[0]
    return mask_nz

# Alg/Plotting utils
# ========================================
def get_bins(dist):
    q95 = np.percentile(dist, 95)
    mu, std = np.mean(dist), np.std(dist)
    lower, mu_up = 0., mu + 2*std
    upper = min(q95, mu_up)
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

# Load Data
# ========================================
# Data vars
rs_tup_index = -2
zx, zy = trained_redshifts[rs_tup_index]
rsx, rsy = REDSHIFTS[zx], REDSHIFTS[zy]

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
median_Vlinear = np.median(l2_dist_Vlinear)
median_pred    = np.median(l2_dist_pred)
#
# Get hist bins from statistics on l2 distance
bins = get_bins(l2_dist_Vlinear)
#
# Plot hist
# ========================================
def plot_hist():
    plt.close('all')
    # format labels
    label_Vlin = '{}: {:.6f}'.format(Vlin_label, median_Vlinear)
    label_pred = '{}: {:.6f}'.format(pred_label, median_pred)
    # call hist
    plt.hist(l2_dist_Vlinear, bins= bins, label=label_Vlin, color=Vlin_c, alpha=alpha)
    plt.hist(l2_dist_pred,    bins= bins, label=label_pred, color=pred_c, alpha=alpha)
    # Add some graph details
    title = 'Error, single-step {:>2}-{:>2}: {:.4f} --> {:.4f}'.format(zx, zy, rsx, rsy)
    plt.title(title)
    plt.xlabel('Distance (L2)')
    #plt.legend()
    L = plt.legend()
    plt.setp(L.texts, family='monospace')
    plt.grid(True, alpha=0.5, ls='--')
    plt.tight_layout()


plot_hist()
plt.show()
#save_plot()


