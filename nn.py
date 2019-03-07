import code, sys
import numpy as np
import tensorflow as tf
import utils

#-----------------------------------------------------------------------------#
#                                  set model                                  #
#-----------------------------------------------------------------------------#

def set_layer(h_in, layer_vars):
    """
    Params
    ------
    h_in : tf.tensor(float32); (b, N, D)
        point cloud input data

    layer_vars : tuple(tensor)
        tuple with this layer's weights W, and bias B
    """
    #=== get vars
    W, B = layer_vars
    W = W[0]  # only one weight for set layer

    #=== layer fwd, # W.(X - X_mu) + B
    h_mu  = tf.reduce_mean(h_in, axis=1, keepdims=True) # (b, 1, D)
    h     = h_in - h_mu
    h_out = tf.einsum('bnk,kq->bnq', h, W) + B
    return h_out


def network_func_set(X_in, model_vars):
    """ network funcs are composed of their constituent layer funcs

    Params
    ------
    X_in : tf.tensor(float32); (b, N, D)
        input batch

    model_vars : collections.namedtuple
        collection of model vars:
            num_layers : int, the number of layers in network
            var_scope : str, namespace for tf vars
            get_layer_vars : <function: utils.get_params>
                the scoped getter for the model vars
            activation : tf.nn function

    Returns
    -------
    H : tf.tensor(float32)
        network output
    """
    #=== Unpack some model variables
    num_layers = model_vars.num_layers
    activation = model_vars.activation
    get_layer_vars = model_vars.get_layer_vars
    is_last = lambda i: i >= num_layers - 1

    #=== Input layer
    H = activation(set_layer(X_in, get_layer_vars(0)))

    #=== Hidden layers
    for layer_idx in range(1, num_layers):
        layer_vars = get_layer_vars(layer_idx)
        H = set_layer(H, layer_vars)
        if not is_last(layer_idx): # no activation on output
            H = activation(H)
    return H


def model_func_set(za_in, model_vars):
    """ Model functions are the interface to network functions
    The network function is purely the neural network forward ops,
    while the model function manages any pre/post processing

    Params
    ------
    za_in : tf.tensor(float32); (b, N, 6)
        za input data, where X_in[...,:3] is init pos, X_in[...,3:] is displacement

    model_vars : Initializer
        Initializer instance that has model config and variable utils
    """
    #var_scope = model_vars.var_scope
    #num_layers = len(model_vars.channels) - 1

    # Network forward
    # ===============
    #with tf.variable_scope(var_scope, reuse=True): # so layers can get variables
    #    # ==== Network output
    #    X_in = get_init_pos_tf(za_disp)
    #    pred_error = network_func_set(X_in, num_layers, activation, model_vars)
    #    return pred_error

    init_pos = get_init_pos(za_in)
    X_in = tf.concat([init_pos, za_in], axis=-1)

    X_out = network_func_set(X_in, model_vars)
    return X_out






#=============================================================================
# periodic boundary conditions, loss
#=============================================================================
def get_readout(h_out):
    M = h_out.get_shape().as_list()[-1]

    # bounding coo
    h_out_coo = h_out[...,:3]
    gt_one  = (tf.sign(h_out_coo - 1) + 1) / 2
    ls_zero = -(tf.sign(h_out_coo) - 1) / 2
    rest = 1 - gt_one - ls_zero
    readout = rest*h_out_coo + gt_one*(h_out_coo - 1) + ls_zero*(1 + h_out_coo)

    if M > 3: # then vel was predicted as well, concat
        readout = tf.concat([readout, h_out[...,3:]], axis=-1)
    return readout



def periodic_boundary_dist(readout_full, x_truth):
    """ minimum distances between particles given periodic boundary conditions
    Normal squared distance would penalize for large difference between particles
    on opposite sides of cube
    """
    readout = readout_full[...,:3]
    x_truth_coo = x_truth[...,:3]
    d1 = tf.squared_difference(readout, x_truth_coo)
    d2 = tf.squared_difference(readout, (1 + x_truth_coo))
    d3 = tf.squared_difference((1 + readout),  x_truth_coo)
    dist = tf.minimum(tf.minimum(d1, d2), d3)
    return dist

# diff for ZA, temp try somethin else
def pbc_loss(x_pred, x_truth, scale_error=True):
    """ MSE over full dims with periodic boundary conditions
    Args:
        readout (tensor): model prediction which has been remapped to inner cube
        x_truth (tensor): ground truth (mb_size, N, 6)
        vel: if vel, then include vel error in loss
    """
    pbc_dist  = periodic_boundary_dist(x_pred, x_truth)
    error = tf.reduce_mean(tf.reduce_sum(pbc_dist, axis=-1))
    if scale_error:
        error = error * 1e5
    return error


def loss_ZA(predicted_error, true_error):
    """
    We want net to predict the error from ZA approx to truth (FPM)
    thus:
        loss = predicted_error - true_error
    where
        predicted_error = network output
        # We want this AS close to possible as actual_error

        true_error == FPM_disp - ZA_disp

    predicted_error.shape == true_error.shape == (b, N, 3)
    """
    err_diff = tf.squared_difference(predicted_error, true_error) # (b, N, 3)
    error = tf.reduce_mean(tf.reduce_sum(err_diff, axis=-1))
    return error


#=============================================================================#
#                                                                             #
#      _ __    _   _   _ __ ___    _ __    _   _      ___    _ __    ___      #
#     | '_ \  | | | | | '_ ` _ \  | '_ \  | | | |    / _ \  | '_ \  / __|     #
#     | | | | | |_| | | | | | | | | |_) | | |_| |   | (_) | | |_) | \__ \     #
#     |_| |_|  \__,_| |_| |_| |_| | .__/   \__, |    \___/  | .__/  |___/     #
#                                 |_|      |___/            |_|               #

def mse_za(fpm_displacement, za_displacement):
    err_diff = np.square(fpm_displacement - za_displacement)
    error = np.mean(np.sum(err_diff, axis=-1))
    return error


def get_init_pos(za_disp):
    b, N, k = za_disp.shape
    mg = range(2, 130, 4)
    q = np.einsum('ijkl->kjli', np.array(np.meshgrid(mg, mg, mg)))
    qr = q.reshape(-1, 3)
    init_pos = za_disp + qr
    return init_pos


def get_init_pos_tf(disp):
    mg = tf.range(2,130,4)
    q = tf.cast(tf.einsum('ijkl->kjli', tf.meshgrid(mg, mg, mg)), tf.float32)
    qr = tf.reshape(q, (-1, 3))
    init_pos = disp + qr
    return init_pos





# https://arxiv.org/abs/1506.02025 # spatial trans
# https://arxiv.org/abs/1706.03762 # attn all u need (nlp)
# >>-----> https://arxiv.org/pdf/1710.10903.pdf  # graph attention nets GATs

def attn_layer(foo):
    """ see p.3,4 of GATs, eqns 1,2,6, fig1
        auth's TF code: https://github.com/PetarV-/GAT
    """
    pass
