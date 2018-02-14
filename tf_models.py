import os, code, sys, time
import numpy as np
import tensorflow as tf
#import chainer.functions as F
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

from utils import init_weight, init_bias
from tf_nn import linear_layer, get_readout, pbc_loss

'''
I don't know if train has to be defined here with all the placeholders,
or whether it can be defined in train script
'''
#=============================================================================
# model setup
#=============================================================================
# var naming
WEIGHT_TAG = 'W_{}'
BIAS_TAG   = 'B_{}' # eg 'B_6'

# model params
channels = [6, 8, 16, 32, 16, 8, 3, 8, 16, 32, 16, 8, 3]
kdims = [(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
lr = 0.01
activation = tf.nn.relu

#=============================================================================
# hardcoded tensorflow boilerplate
# I mean "graph"
#=============================================================================
# external data
X_input = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_input')
X_truth = tf.placeholder(tf.float32, shape=[None, num_particles**3, 6], name='X_truth')

# network layers and hyper params, 12 layers
#============================================================================= H0
W0 = init_weight(*kdims[0], WEIGHT_TAG.format(0))
B0 = init_bias(  *kdims[0],   BIAS_TAG.format(0))
H0 = activation(linear_layer(X_input, W0, B0))

#============================================================================= H1
W1 = init_weight(*kdims[1], WEIGHT_TAG.format(1))
B1 = init_bias(  *kdims[1],   BIAS_TAG.format(1))
H1 = activation(linear_layer(H0, W1, B1))

#============================================================================= H2
W2 = init_weight(*kdims[2], WEIGHT_TAG.format(2))
B2 = init_bias(  *kdims[2],   BIAS_TAG.format(2))
H2 = activation(linear_layer(H1, W2, B2))

#============================================================================= H3
W3 = init_weight(*kdims[3], WEIGHT_TAG.format(3))
B3 = init_bias(  *kdims[3],   BIAS_TAG.format(3))
H3 = activation(linear_layer(H2, W3, B3))

#============================================================================= H4
W4 = init_weight(*kdims[4], WEIGHT_TAG.format(4))
B4 = init_bias(  *kdims[4],   BIAS_TAG.format(4))
H4 = activation(linear_layer(H3, W4, B4))

#============================================================================= H5
W5 = init_weight(*kdims[5], WEIGHT_TAG.format(5))
B5 = init_bias(  *kdims[5],   BIAS_TAG.format(5))
H5 = activation(linear_layer(H4, W5, B5))

#============================================================================= H6
W6 = init_weight(*kdims[6], WEIGHT_TAG.format(6))
B6 = init_bias(  *kdims[6],   BIAS_TAG.format(6))
H6 = activation(linear_layer(H5, W6, B6))

#============================================================================= H7
W7 = init_weight(*kdims[7], WEIGHT_TAG.format(7))
B7 = init_bias(  *kdims[7],   BIAS_TAG.format(7))
H7 = activation(linear_layer(H6, W7, B7))

#============================================================================= H8
W8 = init_weight(*kdims[8], WEIGHT_TAG.format(8))
B8 = init_bias(  *kdims[8],   BIAS_TAG.format(8))
H8 = activation(linear_layer(H7, W8, B8))

#============================================================================= H9
W9 = init_weight(*kdims[9], WEIGHT_TAG.format(9))
B9 = init_bias(  *kdims[9],   BIAS_TAG.format(9))
H9 = activation(linear_layer(H8, W9, B9))

#============================================================================= H10
W10 = init_weight(*kdims[10], WEIGHT_TAG.format(10))
B10 = init_bias(  *kdims[10],   BIAS_TAG.format(10))
H10 = activation(linear_layer(H9, W10, B10))

#============================================================================= H11
W11 = init_weight(*kdims[11], WEIGHT_TAG.format(11))
B11 = init_bias(  *kdims[11],   BIAS_TAG.format(11))
H11 = linear_layer(H10, W11, B11)

#============================================================================= output
readout = get_readout(H11)
loss  = pbc_loss(readout, X_truth)
train = tf.train.AdamOptimizer(lr).minimize(loss)