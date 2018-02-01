import os
import sys
import time
import argparse
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.optimizers as optimizers
from chainer import cuda
import numpy as np
import cupy
import matplotlib.pyplot as plt

import models
import nn
import data_utils


# static vars
#RNG_SEEDS     = [98765, 12345, 319, 77743196] # takes too long
#RNG_SEEDS     = [98765, 12345, 77743196] 
RNG_SEED     = 77743196 # speed things up
BOUND         = (0.095, 1-0.095)
LEARNING_RATE = 0.01
GRAPH_CHANNELS = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 6] # for graph model
SET_CHANNELS   = [6, 32, 128, 256, 128, 32, 256, 16, 6]
RS_CHANNELS   = [6, 32, 128, 256, 64, 32, 16, 6]
CHANNELS     = {0:SET_CHANNELS, 1:GRAPH_CHANNELS, 2:None, 3:RS_CHANNELS}
NBODY_MODELS = {0:models.SetModel, 1:models.GraphModel, 2:models.VelocityScaled, 3:models.RSModel}
MTAGS        = {0:'S', 1:'G', 2:'V', 3:'RS'}
RS_IDX = {6.0:0, 4.0:1, 2.0:2, 1.5:3, 1.2:4, 1.0:5, 0.8:6, 0.6:7, 0.4:8, 0.2:9, 0.0:10} # for new data 

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-b', default=4,          type=int, help='batch size')
parser.add_argument('--num_iters', '-i', default=3000,       type=int, help='number of training iterations')
parser.add_argument('--model_dir', '-s', default='./Model/',           help='directory where you can find a file storing model information')
parser.add_argument('--model_name','-n', default='',         type=str, help='model name')
parser.add_argument('--model_type','-m', default=0,          type=int, help='model type, 0:set, 1:graph, 2:vel')
parser.add_argument('--use_theta', '-t', default=0,          type=int, help='if 1, use theta timestep coeff')
parser.add_argument('--pred_vel',  '-v', default=0,          type=int, help='if 1, predict velocity too')
parser.add_argument('--rotate_aug','-a', default=0,          type=int, help='if 1, randomly augment data by rotation')
parser.add_argument('--gpu_use',   '-g', default=1,          type=int, help='use gpu if 1, else cpu')
parser.add_argument('--particles', '-p', default=16,         type=int, help='number of particles, dataset')
parser.add_argument('--redshifts', '-r', nargs='+',          type=float, help='redshift tuple')
args = parser.parse_args()
start_time = time.time()

#=============================================================================
# Training and model params, from arg parser
#=============================================================================
use_gpu = True if args.gpu_use == 1 else False
xp = cupy if use_gpu == 1 else np
mb_size   = args.batchsize # 8
num_iters = args.num_iters # 3000
num_particles = args.particles
zX, zY   = args.redshifts
rs_start, rs_target = RS_IDX[zX], RS_IDX[zY]

mtype    = NBODY_MODELS[args.model_type]
channels = CHANNELS[args.model_type]
mname  = args.model_name
use_rotate = True if args.rotate_aug == 1 else False
if use_rotate:
    mname += 'R'
if args.pred_vel: 
    mname += 'VP_'
    channels[-1] = 6
theta = None
theta_tag = 'L' if args.use_theta == 1 else ''
save_label = '{}{}{}{}_{}_'.format(mname, MTAGS[3], MTAGS[args.model_type], theta_tag, num_particles)
if args.use_theta == 1:
    theta = np.load('./thetas_timesteps.npy').item()

model_dir = args.model_dir
loss_path = model_dir + 'Loss/'
if not os.path.exists(loss_path): os.makedirs(loss_path)
cube_path = model_dir + 'Cubes/'
if not os.path.exists(cube_path): os.makedirs(cube_path)

def seed_rng(s=12345):
    np.random.seed(s)
    xp.random.seed(s)

def split_data_validation(X, num_val_samples=200):
    """ split dataset into training and validation sets
    
    Args:        
        X, Y (ndarray): data arrays of shape (num_samples, num_particles, 6)
        num_val_samples (int): size of validation set
    """
    num_samples = X.shape[1]
    idx_list = np.random.permutation(num_samples)
    X = X[:,idx_list]
    X_train, X_val = X[:,:-num_val_samples], X[:,-num_val_samples:]
    return X_train, X_val
#=============================================================================
# Load data
#=============================================================================
data_all_path = '/home/evan/Data/nbody_simulations/ALL_{}.npy'.format(num_particles)
X = data_utils.normalize(np.load(data_all_path))
seed_rng()
X_train, X_val = split_data_validation(X, num_val_samples=200)
X = None # to reduce memory

# See if input and target redshift data has been saved
val_data_name = 'X_{}_data'
if not os.path.exists(cube_path + val_data_name.format(num_particles) + '.npy'):
    np.save(cube_path + val_data_name.format(num_particles), cuda.to_cpu(X_val))



#=============================================================================
# Loss history
#=============================================================================
train_loss_history = np.zeros((num_iters))
#num_val_batches = X_val.shape[0] // mb_size
validation_loss_history = np.zeros((X_val.shape[1]))

#=============================================================================
# Training
#=============================================================================
model_save_label = save_label + '{}_'.format(RNG_SEED)
#print('{} BEGIN'.format(model_save_label))
print('{}_{}:{}'.format(model_dir, model_save_label, num_iters))
seed_rng(RNG_SEED)

# setup model
model = models.RSModel(channels,layer=mtype)
if use_gpu:
    model.to_gpu()

# setup optimizer
optimizer = optimizers.Adam(alpha=LEARNING_RATE)
optimizer.setup(model)

    
# train loop
for cur_iter in range(num_iters):
    #print('train iter {}'.format(cur_iter))
    model.zerograds() # must always zero grads before another forward pass!

    # create mini-batches for training
    _x_in = data_utils.next_multi_minibatch(X_train, mb_size)
    if use_gpu:
        _x_in = cuda.to_gpu(_x_in)
    x_in = chainer.Variable(_x_in)

    # get prediction and loss
    x_hat, loss = model.fwd_pred_loss(x_in) # prediction
        
    # backprop and update
    loss.backward() # this calculates all the gradients (backprop)
    optimizer.update() # this updates the weights

    train_loss_history[cur_iter] = cuda.to_cpu(loss.data)
    if cur_iter % 10 == 0:
        np.save(loss_path + save_label + 'train_loss', train_loss_history)
    if cur_iter % 100 == 0 and cur_iter != 0:
        data_utils.save_model([model, optimizer], model_dir + model_save_label)

# save loss
np.save(loss_path + save_label + 'train_loss', train_loss_history)
print('{}: converged at {}'.format(model_save_label, np.median(train_loss_history[-150:])))

# save model, optimizer
data_utils.save_model([model, optimizer], model_dir + model_save_label)
X_train = None

# validation
rs_distance = rs_target - rs_start
val_predictions = np.zeros(((rs_distance,) + X_val.shape[1:])).astype(np.float32)
with chainer.using_config('train', False):
    for val_iter in range(X_val.shape[0]):
        _val_in   = X_val[rs_start,  val_iter: val_iter+1]
        _val_true = X_val[rs_target, val_iter: val_iter+1]
        if use_gpu:
            _val_in = cuda.to_gpu(_val_in)
            _val_true = cuda.to_gpu(_val_true)
        val_in, val_true = chainer.Variable(_val_in), chainer.Variable(_val_true)
        # predict
        val_hat, preds = model.fwd_pred(val_in, (rs_start, rs_target))
        val_predictions[:,val_iter] = cuda.to_cpu(preds[:,0])
        # loss
        val_loss = nn.mean_squared_error(val_hat, val_true, boundary=BOUND)
        validation_loss_history[val_iter] = cuda.to_cpu(val_loss.data)
    
    # save data
    pred_save_label ='{}X32_{}-{}_predictions'.format(mname, zX,zY)
    np.save(cube_path + pred_save_label, val_predictions)
    np.save(loss_path + save_label + 'val_loss', validation_loss_history)
    print('{}: validation avg {}'.format(model_save_label, np.mean(validation_loss_history)))
model = optimizer = None

with open('cube_tags.txt', 'a') as f:
    f.write(save_label + '\n')
#=============================================================================
# Plot
#=============================================================================
plt.clf()
plt.figure(figsize=(16,8))
plt.grid()
plot_title = save_label
lh = train_loss_history
plt.plot(lh[150:], c='b', label='train error, {}'.format(np.median(lh[-150:])))
#plt.plot(lh, c='b', label='train error, {}'.format(np.median(lh)))
plt.title(plot_title)
plt.legend()
plt.savefig(loss_path + save_label + 'train_plot', bbox_inches='tight')
plt.close('all')
total_time = (time.time() - start_time) // 60
print('{} END, elapsed time: {}'.format(save_label, total_time))