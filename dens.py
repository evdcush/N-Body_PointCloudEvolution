import os, code, sys, time, argparse

import numpy as np
import sklearn.neighbors as skn
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import tensorflow as tf
from scipy.sparse import coo_matrix, csr_matrix

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

N = 32**3
sess = tf.InteractiveSession()


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

def get_rad_sparse_mat(x, R=RAD):
    N = x.shape[0]
    # just easier to diff indptr for now
    # get csr
    rad_csr = rad_fn(x, R)
    rad_coo = rad_csr.tocoo()

    # diff data for matmul op select
    div_diff = np.diff(rad_csr.indptr)
    coo_data_divisor = np.repeat(div_diff, div_diff).astype(np.float32)
    coo_data = rad_coo.data / coo_data_divisor

    new_coo = coo_matrix((coo_data, (rad_coo.row, rad_coo.col)), shape=(N,N)).astype(np.float32)
    return new_coo


def get_rad_sparse_batch_mat(x, R=RAD):
    b, N = x.shape[:2]
    # just easier to diff indptr for now
    coo = get_rad_sparse_mat(x[0], R)
    rows = coo.row
    cols = coo.col
    data = coo.data

    for i in range(1, b):
        # get coo, offset indices
        coo = get_rad_sparse_mat(x[i], R)
        row = coo.row + (N * i)
        col = coo.col + (N * i)
        datum = coo.data

        # concat to what we have
        rows = np.concatenate((rows, row))
        cols = np.concatenate((cols, col))
        data = np.concatenate((data, datum))

    coo = coo_matrix((data, (rows, cols)), shape=(N*b, N*b)).astype(np.float32)
    return coo


def get_rad_sparse_tensor(x, R=RAD):
    # get rad sparse coo
    coo = get_rad_sparse_mat(x, R)

    # construct sparse tensor
    idx = np.mat([coo.row, coo.col]).transpose()
    rad_sparse_tensor = tf.SparseTensor(idx, coo.data, coo.shape)
    return rad_sparse_tensor

def get_rad_sparse_batch_tensor(x, R=RAD):
    # get rad sparse coo
    coo = get_rad_sparse_batch_mat(x, R)

    # construct sparse tensor
    idx = np.mat([coo.row, coo.col]).transpose()
    rad_sparse_tensor = tf.SparseTensor(idx, coo.data, coo.shape)
    return rad_sparse_tensor

def tf_sparse_matmul(sparse_mat, x):
    return tf.sparse_tensor_dense_matmul(sparse_mat, x)

j = 1
spt1 = get_rad_sparse_tensor(np.copy(X[j]))
y1 = tf_sparse_matmul(spt1, np.copy(X[j])).eval()

mb_size = 3
batch_spt = get_rad_sparse_batch_tensor(np.copy(X[:mb_size]))
batch_x = np.copy(X[:mb_size]).reshape(-1,6)
batch_y = tf.sparse_tensor_dense_matmul(batch_spt, batch_x).eval()
y2 = batch_y.reshape(mb_size, N, 6)

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
