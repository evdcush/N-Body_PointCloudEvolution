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
RAD = 0.08

def rad_fn(h_in, rad=RAD):
    return radius_neighbors_graph(h_in[...,:3], rad, include_self=True).astype(np.float32)

#
R = RAD
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


def get_rad_sparse_tensor(x, R=RAD):
    # just easier to diff indptr for now
    # get csr
    rad_csr = rad_fn(x, R)
    rad_coo = rad_csr.tocoo()

    # diff data for matmul op select
    div_diff = np.diff(rad_csr.indptr)
    coo_data_divisor = np.repeat(div_diff, div_diff).astype(np.float32)
    coo_data = rad_coo.data / coo_data_divisor

    # construct sparse tensor
    idx = np.mat([rad_coo.row, rad_coo.col]).transpose()
    rad_sparse_tensor = tf.SparseTensor(idx, coo_data, rad_coo.shape)
    return rad_sparse_tensor



def tf_sparse_matmul(sparse_mat, x):
    return tf.sparse_tensor_dense_matmul(sparse_mat, x)

r1 = rad_fn(np.copy(X[1]), R)
r2 = rad_fn(np.copy(X[2]), R)
r3 = rad_fn(np.copy(X[3]), R)
r4 = rad_fn(np.copy(X[4]), R)


r1coo = r1.tocoo()
r1row = r1coo.row
r1col = r1coo.col

#r2coo = r2.tocoo()
#r2row = r2coo.row
#r2col = r2coo.col

#print('r1row.shape: {}, r2row.shape: {}'.format(r1row.shape, r2row.shape))

idx1 = np.mat([r1row, r1col]).transpose()
#idx2 = np.mat([r2row, r2col]).transpose()

spt1 = tf.SparseTensor(idx1, r1coo.data, r1coo.shape)
#spt2 = tf.SparseTensor(idx2, r2coo.data, r2coo.shape)
diffs = np.diff(r1.indptr).astype(np.float32)[:,None]
y1 = tf.sparse_tensor_dense_matmul(spt1, np.copy(X[1])) / diffs


spt1_control = get_rad_sparse_tensor(np.copy(X[1]))
y1_control = tf_sparse_matmul(spt1_control, np.copy(X[1]))

# np.allclose(y1.eval(), y1_control.eval(), atol=1e-06) == True # atol default 1e-08
# np.sum(np.abs(y1.eval() - y1_control.eval())) == 0.01489

"""
Notes:

CSR: (N, N)
 - indptr: (N+1,) index pointers
 - indices: (M) sample indices
     indices for sample i found at indices[indptr[i]:indptr[i+1]]
 - data: (M) data values (in the case of adjecency matrix, just 1 or 0)
     data for sample i found at data[indptr[i]:indptr[i+1]]

COO: (N, N)
 - row: (M) row indices
    for each data sample index i (of N samples), i gets broadcasted k times,
    where k is the number of neighbors i has.
        eg: for kneighbor graph with K=5, row look like
         row: [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3 ...]
 - col: (M) column indices
    This one has the actual indices of the neighbors for sample i
        eg: if K=5, col: [0,13,49,6,1123,| 1,29,7,4985,12,| ...]
 - data: (M), all ones, for indices


use `_, counts = np.unique(row, return_counts=True)` for your matmul div
 preprocess coo.data as such:
   _, counts = np.unique(row, return_counts=True)
   counts_broadcasted = np.repeat(counts, counts).astype(np.float32)
   coo_data = coo_data / counts_broadcasted





"""
