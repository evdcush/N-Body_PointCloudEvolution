import os, sys, code, time, argparse, shutil

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
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--particles', '-p', default=16,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--redshifts', '-z', default=[6.0, 0.0], nargs='+', type=float, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--model_type','-m', default=0,          type=int,  help='model type')
parser.add_argument('--multi_step','-r', default=False,      type=bool, help='use multi-step redshift model')
parser.add_argument('--num_iters', '-i', default=5000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=8,          type=int,  help='training batch size')
parser.add_argument('--model_dir', '-s', default='./Model/', type=str,  help='directory where model parameters are saved')
parser.add_argument('--save_name', '-n', default='',         type=str,  help='model name')
parser.add_argument('--vel_coeff', '-c', default=False,      type=bool, help='use timestep coefficient on velocity')
parser.add_argument('--use_gpu',   '-g', default=True,       type=bool, help='use gpu')
parser.add_argument('--verbose',   '-v', default=False,      type=bool, help='verbose prints training progress')
args = vars(parser.parse_args())
start_time = time.time()
print('{}'.format(args))

def seed_rng(s=DATASET_SEED):
    np.random.seed(s)
    xp.random.seed(s)
#=============================================================================
# Data and training parameters
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
vel_coeffs = vel_tag = None
if args['vel_coeff']:
    vel_coeffs = utils.load_velocity_coefficients(num_particles)
    vel_tag = 'L'

# multi-model
multi_step = args['multi_step']

#=============================================================================
# Model setup
#=============================================================================
# Initialize model
if multi_step:
    # if multi_step, then args['model_type'] corresponds to RSModel's constituent layers
    model_params = NBODY_MODELS[args['model_type']]
    child_class = model_params['mclass']
    channels    = model_params['channels'][:-1] + [6] # insure velocity also predicted
    child_tag   = model_params['tag']
    tag   = '{}{}'.format('R', child_tag)

    model_class = models.RSModel
    model = model_class(channels, layer=child_class, vel_coeff=vel_coeffs, rng_seed=PARAMS_SEED, loss)
else:
    model_params = NBODY_MODELS[args['model_type']]
    model_class = model_params['mclass']
    channels    = model_params['channels']
    tag         = model_params['tag']
    if vel_coeffs is not None:
        vel_coeffs = vel_coeffs[(zX, zY)]
    seed_rng(PARAMS_SEED)
    model = model_class(channels, vel_coeff=vel_coeffs)

if use_gpu:
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.Adam(alpha=LEARNING_RATE)
optimizer.setup(model)

# Loss function
loss_fun = model_params['loss']

#=============================================================================
# Session save parameters
#=============================================================================
# save names # eg newknn_GL_32
model_name = '{}{}_{}'.format(tag, vel_tag, num_particles)
if args['save_label'] != '':
    save_name = '{}_{}'.format(args['save_label'], model_name)
else:
    save_name = model_name

# path variables # eg ./Model/newknn_GL_32/
model_dir = '{}/{}/'.format(args['model_dir'], save_name)
loss_path = model_dir + 'Loss/'
cube_path = model_dir + 'Cubes/'
copy_path = model_dir + '.original_files/'
utils.make_dirs([loss_path, cube_path, copy_path])
utils.save_pyfiles(copy_path)

#=============================================================================
# Load data, normalize, split
#=============================================================================
X = utils.load_npy_data(num_particles) # shape (11, N, D, 6)
if not multi_step:
    X = X[[rs_start, rs_target]] # shape (2, N, D, 6)
X = utils.normalize_fullrs(X)

if use_gpu and not multi_step:
    # Since X is small when not multi-step, can just load to GPU here
    # otherwise, only batches should be loaded
    X = cuda.to_gpu(X)

# Split into train and validation sets
seed_rng()
X_train, X_val = utils.multi_split_data_validation(X, num_val_samples=200)

# Reduce memory overhead
X = None

# Save validation input data
utils.save_val_cube(X_val, cube_path, (zX, zY), prediction=False)

#=============================================================================
# Training
#=============================================================================
# to keep track of loss data
train_loss_history      = np.zeros((num_iters))
validation_loss_history = np.zeros((X_val.shape[1]))

# train loop
for cur_iter in range(num_iters):
    time_start = time.time()
    model.zerograds() # must always zero grads before another forward pass!
    _x_in = utils.next_minibatch(X_train, batch_size, data_aug=True)
    if use_gpu: _x_in = cuda.to_gpu(_x_in)

    # fwd pass
    if multi_step:
        x_in = chainer.Variable(_x_in)
        x_hat, loss = model.fwd_predictions(x_in, loss_fun=loss_fun)
    else:
        # need to split data if not multi
        _x_in, _x_true = _x_in[0], _x_true[1]
        x_in   = chainer.Variable(_x_in)
        x_true = chainer.Variable(_x_true)
        x_hat  = model(x_in)
        loss   = loss_fun(xhat, x_true)

    # backprop and update
    loss.backward() # this calculates all the gradients (backprop)
    optimizer.update() # this updates the weights

    train_loss_history[cur_iter] = cuda.to_cpu(loss.data)
    if args['verbose']:
        utils.print_status(cur_iter, train_loss_history[cur_iter], time_start)
    if cur_iter % 10 == 0 and cur_iter != 0:
        utils.save_loss(loss_path + save_name, train_loss_history)
        if cur_iter % 100 == 0:
            utils.save_model(model, optimizer, model_dir + save_name)

# save loss and hyperparameters
utils.save_loss(loss_path + save_name, train_loss_history)
utils.save_model(model, optimizer, model_dir + save_name)
print('{}: converged at {}'.format(save_name, np.median(train_loss_history[-100:])))

# reduce memory overhead
X_train = None

# validation
val_predictions = utils.init_val_predictions(X_val.shape, channels[-1])
with chainer.using_config('train', False):
    for val_iter in range(X_val.shape[1]):
        j,k = val_iter, val_iter+1
        _val_in = X_val[:,j:k]
        if use_gpu: cuda.to_gpu(_val_in)
        if multi_step:
            val_in = chainer.Variable(_val_in)
            val_hat, predictions = model.fwd_target(val_in, (rs_target, rs_start))
            val_predictions[:,val_iter] = cuda.to_cpu(predictions[:,0].data)
            val_loss = loss_fun(val_hat, val_in[-1])
        else:
            _val_in, _val_truth = _val_in[0], _val_in[1]
            val_in   = chainer.Variable(_val_in)
            val_true = chainer.Variable(_val_truth)
            val_hat  = model(val_in)
            val_predictions[val_iter] = cuda.to_cpu(val_hat[0].data)
            val_loss = loss_fun(val_hat, val_true)
        validation_loss_history[val_iter] = cuda.to_cpu(val_loss.data)

    # save data
    utils.save_val_cube(val_predictions, cube_path, (zX, zY), prediction=True)
    utils.save_loss(loss_path + save_name, validation_loss_history, validation=True)
    print('{}: validation mode {}'.format(save_name, np.median(validation_loss_history)))

#=============================================================================
# Plot and save
#=============================================================================
plt.save_loss_curves(loss_path + save_name, train_loss_history, save_name)
plt.save_loss_curves(loss_path + save_name, validation_loss_history, save_name, val=True)

print('{} Finished'.format(save_name))