import os, code, sys, time
import numpy as np
import tensorflow as tf
#import chainer.functions as F
from sklearn.neighbors import kneighbors_graph
import utils
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use

''' TF nuggets:
 - tf.get_variable only returns an existing variable with name if it was
   created earlier by get_variable. It won't return a variable created using
   tf.Variable()
'''

#=============================================================================
# layer ops
#=============================================================================
def left_mult(h, W):
    return tf.einsum('ijl,lq->ijq', h, W)

def linear_fwd(h_in, W, b):
    """ permutation equivariant linear transformation
    Args:
        h_in: external input, of shape (mb_size, n_P, k_in)
        W: layer weight, of shape (k_in, k_out)
        b: bias, of shape (k_out,)
    """
    mu = tf.reduce_mean(h_in, axis=1, keepdims=True)
    h = h_in - mu
    h_out = left_mult(h, W) + b
    return h_out

def linear_layer(h, layer_idx):
    """ layer gets weights and returns linear transformation
    """
    W, B = utils.get_layer_vars(layer_idx)
    return linear_fwd(h, W, B)

#=============================================================================
# periodic boundary conditions, loss
#=============================================================================
def get_readout(h_out):
    gt_one  = (tf.sign(h_out - 1) + 1) / 2
    ls_zero = -(tf.sign(h_out) - 1) / 2
    rest = 1 - gt_one - ls_zero
    readout = rest*h_out + gt_one*(h_out - 1) + ls_zero*(1 + h_out)
    return readout

def periodic_boundary_dist(readout, x_truth):
    x_truth_coo = x_truth[...,:3]
    dist = tf.minimum(tf.square(readout - x_truth_coo), tf.square(readout - (1 + x_truth_coo)))
    dist = tf.minimum(dist, tf.square((1 + readout) - x_truth_coo))
    return dist

def pbc_loss(readout, x_truth):
    pbc_dist = periodic_boundary_dist(readout, x_truth)
    pbc_error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1), name='loss')
    return pbc_error