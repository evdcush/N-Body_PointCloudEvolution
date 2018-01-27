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
RNG_SEEDS     = [98765, 77743196] # speed things up
BOUND         = 0.095
LEARNING_RATE = 0.01
GRAPH_CHANNELS = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 3] # for graph model
SET_CHANNELS   = [6, 32, 128, 256, 128, 32, 256, 16, 3]
CHANNELS     = {0:SET_CHANNELS, 1:GRAPH_CHANNELS, 2:None}
NBODY_MODELS = {0:models.SetModel, 1:models.GraphModel, 2:models.VelocityScaled}
MTAGS        = {0:'S', 1:'G', 2:'V'}

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-b', default=8,          type=int, help='batch size')
parser.add_argument('--num_iters', '-i', default=3000,       type=int, help='number of training iterations')
parser.add_argument('--model_dir', '-s', default='./Model/',           help='directory where you can find a file storing model information')
parser.add_argument('--model_name','-n', default='',         type=str, help='model name')
parser.add_argument('--model_type','-m', default=0,          type=int, help='model type, 0:set, 1:graph, 2:vel')
parser.add_argument('--use_theta', '-t', default=0,          type=int, help='if 1, use theta timestep coeff')
parser.add_argument('--pred_vel',  '-v', default=0,          type=int, help='if 1, predict velocity too')
parser.add_argument('--gpu_use',   '-g', default=1,          type=int, help='use gpu if 1, else cpu')
parser.add_argument('--particles', '-p', default=16,         type=int, help='number of particles, dataset')
parser.add_argument('--redshifts', '-r', nargs='+',          type=float, help='redshift tuple')
args = parser.parse_args()

#=============================================================================
# Training and model params, from arg parser
#=============================================================================
use_gpu = True if args.gpu_use == 1 else False
xp = cupy if use_gpu == 1 else np
mb_size   = args.batchsize # 8
num_iters = args.num_iters # 3000
num_particles = args.particles
zX, zY   = args.redshifts

mtype    = NBODY_MODELS[args.model_type]
channels = CHANNELS[args.model_type]
mname  = args.model_name
if args.pred_vel: 
    mname += 'VP_'
    channels[-1] = 6
theta = None
save_label = data_utils.get_save_label(mname, MTAGS[args.model_type], args.use_theta, num_particles, zX, zY)
if args.use_theta == 1:
    thetas = np.load('./thetas_timesteps.npy').item()
    theta_val = thetas[(num_particles, zX, zY)]['W']
    theta = theta_val

model_dir = args.model_dir
loss_path = model_dir + 'Loss/'
if not os.path.exists(loss_path): os.makedirs(loss_path)



def seed_rng(s=12345):
    np.random.seed(s)
    xp.random.seed(s)

seed_rng()

#=============================================================================
# Load data
#=============================================================================
if (zX, zY) == (0.6, 0.0) and num_particles == 32:
    X = np.load('X32_0600.npy')
    Y = np.load('Y32_0600.npy')
elif (zX, zY) == (4.0, 2.0) and num_particles == 32:
    X = np.load('X32_4020.npy')
    Y = np.load('Y32_4020.npy')
else:
    X, Y = data_utils.load_data(num_particles, zX, zY, normalize_data=True)

if use_gpu:
    X = cuda.to_gpu(X)
    Y = cuda.to_gpu(Y)
X_tup, Y_tup = data_utils.split_data_validation(X,Y, num_val_samples=200)
X_train, X_val = X_tup
Y_train, Y_val = Y_tup
# memory overflow issues, try None-ing X,Y
X = Y = None

#=============================================================================
# Loss history
#=============================================================================
train_loss_history = np.zeros((len(RNG_SEEDS), num_iters))
num_val_batches = X_val.shape[0] // mb_size
validation_loss_history = np.zeros((len(RNG_SEEDS), num_val_batches))

#=============================================================================
# Training
#=============================================================================
for rng_idx, rseed in enumerate(RNG_SEEDS):
    model_save_label = save_label + '{}_'.format(rseed)
    print('{} BEGIN'.format(model_save_label))
    seed_rng(rseed)
    # setup model
    model = mtype(channels, theta=theta)
    if use_gpu:
        model.to_gpu()
    # setup optimizer
    optimizer = optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    # setup loss trackers
    lh_train = np.zeros((train_loss_history.shape[-1]))
    lh_val   = np.zeros((validation_loss_history.shape[-1]))
    
    
    # train loop
    for cur_iter in range(num_iters):
        #print('train iter {}'.format(cur_iter))
        model.zerograds() # must always zero grads before another forward pass!

        # create mini-batches for training
        _x_in, _x_true = data_utils.next_minibatch([X_train, Y_train], mb_size)
        x_in, x_true = chainer.Variable(_x_in), chainer.Variable(_x_true)

        # get prediction and loss
        x_hat = model(x_in, add=True) # prediction
        #loss = nn.mean_squared_error(x_hat, x_true, boundary=BOUND) # bound = 0.095
        if x_hat.shape[-1] == 6:
            loss, loc_loss, vel_loss = nn.get_bounded_MSE_vel(x_hat, x_true, boundary=BOUND)
            s_loss = cuda.to_cpu(loc_loss.data)
        else:
            loss = nn.mean_squared_error(x_hat, x_true, boundary=BOUND) # bound = 0.095
            s_loss = cuda.to_cpu(loss.data)
        
        # backprop and update
        loss.backward() # this calculates all the gradients (backprop)
        optimizer.update() # this updates the weights

        lh_train[cur_iter] = s_loss
    train_loss_history[rng_idx] = lh_train
    np.save(loss_path + save_label + 'train_loss', train_loss_history)
    print('{}: converged at {}'.format(model_save_label, np.median(lh_train[-150:])))
    # save model, optimizer
    data_utils.save_model([model, optimizer], model_dir + model_save_label)

    # validation
    with chainer.using_config('train', False):
        for val_iter in range(num_val_batches):
            j,k = val_iter * mb_size, (val_iter+1) * mb_size
            _val_in   = X_val[j:k]#xp.copy(X_val[j:k])
            _val_true = Y_val[j:k]#xp.copy(Y_val[j:k])
            val_in, val_true = chainer.Variable(_val_in), chainer.Variable(_val_true)

            val_hat  = model(val_in, add=True)
            #val_loss = nn.mean_squared_error(val_hat, val_true, boundary=BOUND)
            if x_hat.shape[-1] == 6:
                val_loss, loc_loss, vel_loss = nn.get_bounded_MSE_vel(val_hat, val_true, boundary=BOUND)
                s_loss = cuda.to_cpu(loc_loss.data)
            else:
                val_loss = nn.mean_squared_error(val_hat, val_true, boundary=BOUND) # bound = 0.095
                s_loss = cuda.to_cpu(val_loss.data)
            lh_val[val_iter] = s_loss
        validation_loss_history[rng_idx] = lh_val
        np.save(loss_path + save_label + 'val_loss', validation_loss_history)
        print('{}: validation avg {}'.format(model_save_label, np.mean(lh_val)))
    model = optimizer = lh_train = lh_val = None
#print('{}: averaged convergence at {}'.format(save_label, np.median(np.mean(train_loss_history, axis=0))[-150:]))

with open('model_tags.txt', 'a') as f:
    f.write(save_label + '\n')
#=============================================================================
# Plot
#=============================================================================
plt.clf()
plt.figure(figsize=(16,8))
plt.grid()
plot_title = save_label
avg_lh = np.mean(train_loss_history, axis=0)
plt.plot(avg_lh[100:], c='b', label='train error, {}'.format(np.median(avg_lh[-150:])))
plt.title(plot_title)
plt.legend()
plt.savefig(loss_path + save_label + 'train_plot', bbox_inches='tight')
plt.close('all')
print('{} END'.format(save_label))