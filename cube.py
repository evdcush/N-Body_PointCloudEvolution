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

from params import *
import models
import nn
import data_utils
import code

'''
Design notes:
- When training a new model, make sure you save save the py files to a model directory
'''

parser = argparse.ArgumentParser()
parser.add_argument('--particles', '-p', default=16,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--redshifts', '-z', default=[6.0, 0.0], nargs='+', type=float, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--model_type','-m', default=0,          type=int,  help='model type')
parser.add_argument('--multi_step','-r', default=False,      type=bool, help='use multi-step redshift model')
parser.add_argument('--num_iters', '-i', default=5000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=8,          type=int,  help='training batch size')
parser.add_argument('--model_dir', '-s', default='./Model/', type=str,  help='directory where model parameters are saved')
parser.add_argument('--model_name','-n', default='',         type=str,  help='model name')
parser.add_argument('--vel_coeff', '-c', default=False,      type=bool, help='use timestep coefficient on velocity')
parser.add_argument('--use_gpu',   '-g', default=True,       type=bool, help='use gpu')
parser.add_argument('--verbose',   '-v', default=False,      type=bool, help='verbose prints training progress')
args = vars(parser.parse_args())
start_time = time.time()
print('{}'.format(args))
#=============================================================================
# Training and model params, from arg parser
#=============================================================================
# backend
use_gpu = args['use_gpu']
xp      = cupy if use_gpu else np

# training vars
batch_size = args['batch_size']
num_iters  = args['num_iters']

# data vars
num_particles = args['particles']
zX, zY = args['redshifts']
rs_start  = REDSHIFTS.index(zX)
rs_target = REDSHIFTS.index(zY)

# Model vars
vel_coeffs = None
if args['vel_coeff']:
    vel_coeffs = data_utils.load_velocity_coefficients(num_particles)


if args['multi_step']:
    # if multi_step, then args['model_type'] is to RSModel's constituent layers
    model_class = models.RSModel
    child_model = NBODY_MODELS[args['model_type']]
    child_class = child_model['mclass']
    channels = child_model['channels'][:-1] + [6]
    tag   = '{}{}'.format('R', child_model['tag'])
    model = model_class(channels, layer=child_class, rng_seed=RNG_SEED)
mtype    = NBODY_MODELS[args.model_type]
print('mtype: {}'.format(mtype))
channels = CHANNELS[args.model_type]
mname    = args.model_name

# setup model
model = mtype(channels, theta=theta)
if use_gpu: model.to_gpu()

# setup optimizer
optimizer = optimizers.Adam(alpha=LEARNING_RATE)
optimizer.setup(model)


# timestep coefficient
theta = None
theta_tag  = 'L' if args.use_theta == 1 else ''
if args.use_theta == 1:
    theta_path = './velocity_coefficients_{}.npy'.format(num_particles)
    theta = np.load(theta_path).item()[(zX, zY)]

# pathing vars
save_label = '{}{}{}_{}_'.format(mname, MTAGS[args.model_type], theta_tag, num_particles)
model_dir  = args.model_dir
loss_path  = model_dir + 'Loss/'
cube_path  = model_dir + 'Cubes/'
if not os.path.exists(loss_path): os.makedirs(loss_path)
if not os.path.exists(cube_path): os.makedirs(cube_path)

def seed_rng(s=12345):
    np.random.seed(s)
    xp.random.seed(s)


#=============================================================================
# Load data
#=============================================================================
X, Y = data_utils.load_data(num_particles, zX, zY, normalize_data=True)

if use_gpu:
    X = cuda.to_gpu(X)
    Y = cuda.to_gpu(Y)
seed_rng()

X_tup, Y_tup = data_utils.split_data_validation(X,Y, num_val_samples=200)
X_train, X_val = X_tup
Y_train, Y_val = Y_tup

# memory overflow issues, try None-ing X,Y
X = Y = X_tup = Y_up = None

# See if input and target redshift data has been saved
val_data_name = 'X_{}_data'
if not os.path.exists(cube_path + val_data_name.format(num_particles) + '.npy'):
    np.save(cube_path + val_data_name.format(num_particles), cuda.to_cpu(X_val))


#=============================================================================
# Loss history
#=============================================================================
# to keep track of loss data
train_loss_history = np.zeros((num_iters))
validation_loss_history = np.zeros((X_val.shape[0]))

#=============================================================================
# Training
#=============================================================================
print('{}_{}:{}'.format(model_dir, save_label, num_iters))
seed_rng(RNG_SEED)

# setup model
model = mtype(channels, theta=theta)
if use_gpu: model.to_gpu()

# setup optimizer
optimizer = optimizers.Adam(alpha=LEARNING_RATE)
optimizer.setup(model)

    
# train loop
for cur_iter in range(num_iters):
    model.zerograds() # must always zero grads before another forward pass!

    # create mini-batches for training
    _x_in, _x_true = data_utils.next_minibatch([X_train, Y_train], batch_size)
    x_in   = chainer.Variable(_x_in)
    x_true = chainer.Variable(_x_true)

    # get prediction and loss
    x_hat = model(x_in, add=True) # prediction
    loss = nn.mean_squared_error_full(x_hat, x_true)
        
    # backprop and update
    loss.backward() # this calculates all the gradients (backprop)
    optimizer.update() # this updates the weights

    train_loss_history[cur_iter] = cuda.to_cpu(loss.data)
    if cur_iter % 10 == 0 and cur_iter != 0:
        np.save(loss_path + save_label + 'train_loss', train_loss_history)
        if cur_iter % 100 == 0:
            data_utils.save_model([model, optimizer], model_dir + save_label)

# save loss
np.save(loss_path + save_label + 'train_loss', train_loss_history)
print('{}: converged at {}'.format(save_label, np.median(train_loss_history[-150:])))

# save model, optimizer
data_utils.save_model([model, optimizer], model_dir + save_label)
X_train = None

# validation
rs_distance = rs_target - rs_start
val_pred_shape = X_val.shape if channels[-1] == X_val.shape[-1] else X_val.shape[:-1] + (3,)
val_predictions = np.zeros((val_pred_shape)).astype(np.float32)
with chainer.using_config('train', False):
    for val_iter in range(X_val.shape[0]):
        j,k = val_iter, val_iter+1
        val_in   = chainer.Variable(X_val[j:k])
        val_true = chainer.Variable(Y_val[j:k])

        # predict
        val_hat  = model(val_in, add=True)
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        val_predictions[val_iter] = cuda.to_cpu(val_hat.data)
        
        # loss
        val_loss = nn.mean_squared_error_full(val_hat, val_true)
        validation_loss_history[val_iter] = cuda.to_cpu(val_loss.data)
    
    # save data
    pred_save_label ='{}X32_{}-{}_predictions'.format(mname, zX,zY)
    np.save(cube_path + pred_save_label, val_predictions)
    np.save(loss_path + save_label + 'val_loss', validation_loss_history)
    print('{}: validation mode {}'.format(save_label, np.median(validation_loss_history)))
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