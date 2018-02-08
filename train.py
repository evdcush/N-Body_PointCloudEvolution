import os, sys, code, time, argparse, shutil

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.optimizers as optimizers
from chainer.serializers import load_npz
from chainer import cuda

import numpy as np
import cupy
import matplotlib.pyplot as plt

import models
import nn
import utils
from utils import DATASET_SEED, REDSHIFTS, NBODY_MODELS, PARAMS_SEED, LEARNING_RATE, RS_TAGS

parser = argparse.ArgumentParser()
# argparse not handle bools well so 0,1 used instead
parser.add_argument('--particles', '-p', default=16,         type=int,  help='number of particles in dataset, either 16**3 or 32**3')
parser.add_argument('--redshifts', '-z', default=[6.0, 0.0], nargs='+', type=float, help='redshift tuple, predict z[1] from z[0]')
parser.add_argument('--model_type','-m', default=0,          type=int,  help='model type')
parser.add_argument('--resume',    '-r', default=0,          type=int,  help='resume training from serialized params')
parser.add_argument('--multi_step','-s', default=0,          type=int, help='use multi-step redshift model')
parser.add_argument('--num_iters', '-i', default=5000,       type=int,  help='number of training iterations')
parser.add_argument('--batch_size','-b', default=8,          type=int,  help='training batch size')
parser.add_argument('--model_dir', '-d', default='./Model/', type=str,  help='directory where model parameters are saved')
parser.add_argument('--save_prefix','-n', default='',         type=str,  help='model name prefix')
parser.add_argument('--vel_coeff', '-c', default=0,          type=int, help='use timestep coefficient on velocity')
parser.add_argument('--use_gpu',   '-g', default=1,          type=int, help='use gpu')
parser.add_argument('--verbose',   '-v', default=0,          type=int, help='verbose prints training progress')
sess_args = vars(parser.parse_args())
start_time = time.time()
print('{}'.format(sess_args))

def seed_rng(s=DATASET_SEED):
    np.random.seed(s)
    xp.random.seed(s)
#=============================================================================
# Training, data, and model vars
#=============================================================================
# backend
use_gpu = sess_args['use_gpu']
xp      = cupy if use_gpu else np

# training vars
batch_size = sess_args['batch_size']
num_iters  = sess_args['num_iters']
verbose    = sess_args['verbose']

# data vars
num_particles = sess_args['particles']
zX, zY = sess_args['redshifts']
rs_start  = REDSHIFTS.index(zX)
rs_target = REDSHIFTS.index(zY)
vel_coeffs = None
if sess_args['vel_coeff']:
    vel_coeffs = utils.load_velocity_coefficients(num_particles)

# model vars
model_params = NBODY_MODELS[sess_args['model_type']]
model_class  = model_params['mclass']
channels     = model_params['channels']
loss_fun     = model_params['loss']

#=============================================================================
# Session save parameters
#=============================================================================
# model name
model_name = utils.get_model_name((num_particles, zX, zY), model_params['tag'],
                                  sess_args['vel_coeff'], sess_args['save_prefix'])

# save paths
model_dir = '{}{}/'.format(sess_args['model_dir'], model_name)
loss_path = model_dir + 'Loss/'
cube_path = model_dir + 'Cubes/'
utils.make_dirs([loss_path, cube_path])
utils.save_pyfiles(model_dir)

#=============================================================================
# Model setup
#=============================================================================
if vel_coeffs is not None:
    vel_coeffs = vel_coeffs[(zX, zY)]

# Setup model
seed_rng(PARAMS_SEED)
model = model_class(channels, vel_coeff=vel_coeffs)
if sess_args['resume']:
    load_npz('{}{}.model'.format(model_dir, model_name), model)

if use_gpu:
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.Adam(alpha=LEARNING_RATE)
optimizer.setup(model)
if sess_args['resume']:
    load_npz('{}{}.optimizer'.format(model_dir, model_name), optimizer)

#=============================================================================
# Load data, normalize, split
#=============================================================================
X = utils.load_npy_data(num_particles) # shape (11, N, D, 6)
X = X[[rs_start, rs_target]] # shape (2, N, D, 6), # Need to overhaul data loads
X = utils.normalize_fullrs(X)

#if use_gpu: X = cuda.to_gpu(X)

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
#train_loss_history = np.load(loss_path + model_name + '_loss_train.npy')
validation_loss_history = np.zeros((X_val.shape[1]))

# train loop
for cur_iter in range(num_iters):
    time_start = time.time()
    model.zerograds() # must always zero grads before another forward pass!
    _x_in = utils.next_minibatch(X_train, batch_size, data_aug=True)
    if use_gpu: _x_in = cuda.to_gpu(_x_in)
    _x_in, _x_true = _x_in[0], _x_in[1]
    x_in   = chainer.Variable(_x_in)
    x_true = chainer.Variable(_x_true)
    x_hat  = model(x_in)
    loss   = loss_fun(x_hat, x_true)

    # backprop and update
    loss.backward() # this calculates all the gradients (backprop)
    optimizer.update() # this updates the weights

    train_loss_history[cur_iter] = cuda.to_cpu(loss.data)
    loss = x_hat = None
    if verbose:
        utils.print_status(cur_iter, train_loss_history[cur_iter], time_start)
    if cur_iter % 10 == 0 and cur_iter != 0:
        utils.save_loss(loss_path + model_name, train_loss_history)
        if cur_iter % 100 == 0:
            utils.save_model(model, optimizer, model_dir + model_name)

# save loss and hyperparameters
utils.save_loss(loss_path + model_name, train_loss_history)
utils.save_model(model, optimizer, model_dir + model_name)
print('{}: converged at {}'.format(model_name, np.median(train_loss_history[-100:])))

# reduce memory overhead
X_train = None

# validation
val_predictions = np.zeros((X_val.shape[1:-1] + (channels[-1],)))
with chainer.using_config('train', False):
    for val_iter in range(X_val.shape[1]):
        start_time = time.time()
        j,k = val_iter, val_iter+1
        _x_val = X_val[:,j:k]
        if use_gpu: x_val = cuda.to_gpu(_x_val)
        val_in    = chainer.Variable(_x_val[0])
        val_truth = chainer.Variable(_x_val[1])
        val_hat   = model(val_in)
        #val_predictions[val_iter] = cuda.to_cpu(val_hat[0].data)
        val_loss = loss_fun(val_hat, val_truth)
        val_predictions[val_iter] = cuda.to_cpu(nn.get_readout(val_hat)[0].data)
        validation_loss_history[val_iter] = cuda.to_cpu(val_loss.data)
        if verbose:
            utils.print_status(val_iter, validation_loss_history[val_iter], start_time)

    # save data
    utils.save_val_cube(val_predictions, cube_path, (zX, zY), prediction=True)
    utils.save_loss(loss_path + model_name, validation_loss_history, validation=True)
    print('{}: validation mode {}'.format(model_name, np.median(validation_loss_history)))

#=============================================================================
# Plot and save
#=============================================================================
utils.save_loss_curves(loss_path + model_name, train_loss_history, model_name)
utils.save_loss_curves(loss_path + model_name, validation_loss_history, model_name, val=True)
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
print('{} Finished'.format(model_name))