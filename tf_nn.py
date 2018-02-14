import os, code, sys, time
import numpy as np
import tensorflow as tf
#import chainer.functions as F
from sklearn.neighbors import kneighbors_graph
import utils
#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
WEIGHT_TAG = 'W_{}'
BIAS_TAG   = 'B_{}'

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

def linear_layer(h, W, b):
    """ permutation equivariant linear transformation
    Args:
        h: external input, of shape (mb_size, n_P, k_in)
        W: layer weight, of shape (k_in, k_out)
        b: bias, of shape (k_out,)
    """
    mu = tf.reduce_mean(h, axis=1, keepdims=True)
    h_out = h - mu
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    h_out = left_mult(h, W) + b
    return h_out

def linear_layer_get(h, layer_idx):
    """ permutation equivariant linear transformation
    Args:
        h: external input, of shape (mb_size, n_P, k_in)
        W: layer weight, of shape (k_in, k_out)
        b: bias, of shape (k_out,)
    """
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    with tf.variable_scope("params", reuse=True):
        W = tf.get_variable(WEIGHT_TAG.format(layer_idx))
        B = tf.get_variable(  BIAS_TAG.format(layer_idx))
        # tf.get_tensor_by_name apparently can get variables too
        #W = tf.get_tensor_by_name(WEIGHT_TAG.format(layer_idx))
        #B = tf.get_tensor_by_name(  BIAS_TAG.format(layer_idx))
        return linear_layer(h, W, B)

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