import os, code, sys, time, argparse

import numpy as np
import sklearn.neighbors as skn
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import tensorflow as tf

import utils
import nn
from utils import REDSHIFTS, PARAMS_SEED, LEARNING_RATE, RS_TAGS


# data specs
num_particles = 32
redshift_steps = [11,19]
num_rs = len(redshift_steps)
num_rs_layers = num_rs - 1

# load data
X = utils.load_zuni_npy_data(redshifts=redshift_steps, norm_coo=True)[...,:-1]
Y = X[1]
X = X[0]
print('X.shape: {}'.format(X.shape))


def rad_fn(h_in, rad=0.08):
    return radius_neighbors_graph(h_in[...,:3], rad, include_self=True).astype(np.float32)

#
R = 0.08
j = 375
x = X[j]

N = 32**3
sess = tf.InteractiveSession()

x_r = rad_fn(np.copy(x), R)
r_indptr = x_r.indptr
r_indices = x_r.indices


def convert_sparse_matrix_to_sparse_tensor(X):
    X = X.astype(np.float32)
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def convert_batch(lst):
    x = lst[0]
    b = len(lst)
    N = x.shape[0]
    rshape = (b, N, N)
    coo = x.tocoo()
    idx = np.mat([coo.row, coo.col]).transpose()[None,...]
    data = coo.data

def tf_sparse_matmul(sparse_mat, x, diffs):
    return tf.sparse_tensor_dense_matmul(sparse_mat, x) / diffs

r1 = rad_fn(np.copy(X[1]), R)
r2 = rad_fn(np.copy(X[2]), R)
r3 = rad_fn(np.copy(X[3]), R)
r4 = rad_fn(np.copy(X[4]), R)


r1coo = r1.tocoo()
r1row = r1coo.row
r1col = r1coo.col

r2coo = r2.tocoo()
r2row = r2coo.row
r2col = r2coo.col

print('r1row.shape: {}, r2row.shape: {}'.format(r1row.shape, r2row.shape))

idx1 = np.mat([r1row, r1col]).transpose()
idx2 = np.mat([r2row, r2col]).transpose()

spt1 = tf.SparseTensor(idx1, r1coo.data, r1coo.shape)
spt2 = tf.SparseTensor(idx2, r2coo.data, r2coo.shape)
#indices = np.concatenate((idx1, idx2), axis=0)

