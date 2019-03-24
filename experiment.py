import os
import code
import argparse
from functools import partial

import numpy as np
import tensorflow as tf

import utils

####  data  ####
didx = 0
num_test = 200
dataset = utils.Dataset(didx, num_test)
X_train = np.copy(dataset.X_train)
X_val   = np.copy(dataset.X_val)
X_test  = np.copy(dataset.X_test)
del dataset


####  vars  ####
lr = 0.006
#lr = 0.01
#channels = [6, 64, 128, 256, 32, 3]
channels = [6,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,3]
kdims = list(zip(channels[:-1], channels[1:]))
rdims = [(6, k) for _, k in kdims]
num_layers = len(kdims)

####  train  ####
num_iters  = 100000
batch_size = 10
num_particles = 32**3

####  loss  ####
def loss(yhat, y):
    err_diff = tf.squared_difference(yhat, y) # (b, N, 3)
    error = tf.reduce_mean(tf.reduce_sum(err_diff, axis=-1))
    return error

####  init  ####
def glorot_normal(kdims, scale=1.0):
    fan = sum(kdims)
    dv = scale * np.sqrt(2 / fan)
    arr = np.random.normal(scale=dv, size=kdims).astype(np.float32)
    return arr

def init_weight(kdims):
    w = glorot_normal(kdims)
    wvar = tf.Variable(w)
    return wvar

def init_bias(kdims):
    b = np.ones((kdims[-1],), dtype=np.float32) * 1e-6
    bvar = tf.Variable(b)
    return bvar

# Initialization
# ==============
# rng seeds
s1 = 77743196
s2 = 1052
s3 = 918273
np.random.seed(s1)

# init vars
#Wset = [init_weight(k) for k in kdims]
Wf = [init_weight(k) for k in kdims]
Wg = [init_weight(k) for k in kdims]
Wh = [init_weight(k) for k in kdims]

Rset = [init_weight(r) for r in rdims]
Bset = [init_bias(k) for k in kdims]


#-----------------------------------------------------------------------------#
#                                   network                                   #
#-----------------------------------------------------------------------------#

# Layers
# ======

def set_transform(x_in, w, b=None):
    xmu = tf.reduce_mean(x_in, axis=1, keepdims=True)
    x = x_in - xmu
    x_out = tf.einsum('bnk,kq->bnq', x, w)# + b
    if b is not None:
        x_out += b
    return x_out


def set_layer(h_in, idx):
    w, b = Wset[idx], Bset[idx]
    return set_transform(h_in, w, b)


def res_layer(idx):
    """ skip connections
    All skips are from input to some hidden layer,
    which means all res_layer weights are of shape (6, S),
    where 6 is the num of input chans ([grid_pos, za_disp])
    and   S is the num of chans for its corresponding hidden layer
    """
    w = Rset[idx]
    return set_transform(X_in, w)


def attn_layer(x_in, idx):
    # Layer vars
    wf = Wf[idx] # (k_in, k_out)
    wg = Wg[idx] # (k_in, k_out)
    wh = Wh[idx] # (k_in, k_out)
    b  = Bset[idx]

    # Layer transformations
    xf = set_transform(x_in, wf) # (B, N, k_out)
    xg = set_transform(x_in, wg) # (B, N, k_out)
    xh = set_transform(x_in, wh) # (B, N, k_out)

    # reshape for matmuls
    k = kdims[idx][-1]
    xfr = tf.reshape(xf, (-1, k)) # (BN, k_out)
    xgr = tf.reshape(xg, (-1, k)) # (BN, k_out)
    xhr = tf.reshape(xh, (-1, k)) # (BN, k_out)

    # gate fwd
    fg = tf.nn.softmax(tf.matmul(tf.transpose(xfr), xgr)) # (k_out, k_out)
    o = tf.matmul(xhr, fg) # (BN, k_out) * (k_out, k_out) --> (BN, k_out)
    in_shape = tf.shape(x_in)
    bdim = in_shape[0]; n = in_shape[1]
    x_out = tf.reshape(o, (bdim, n, k)) + b
    return x_out


#-----------------------------------------------------------------------------#
#                                    model                                    #
#-----------------------------------------------------------------------------#

def net_fwd(x_in):
    #=== misc func
    norm = tf.layers.batch_normalization

    #=== Activations
    act_set = tf.nn.leaky_relu
    act_res = tf.nn.tanh

    #=== net vars
    #H = norm(act_set(set_layer(x_in, 0)))
    H = norm(act_set(attn_layer(x_in, 0, )))
    R = act_res(res_layer(0))
    #R = res_layer(0)
    for i in range(1, num_layers - 1):
        #H = norm(act_set(set_layer(H + R, i)))
        H = norm(act_set(attn_layer(H, i, )))
        R = act_res(res_layer(i))
        #R = res_layer(i)
    return attn_layer(H + R, num_layers - 1, )

# Setup computational graph
# =========================
X_in  = tf.placeholder(tf.float32, shape=(None, num_particles, 6))
Y     = tf.placeholder(tf.float32, shape=(None, num_particles, 3))
Y_hat = net_fwd(X_in)

opt   = tf.train.AdamOptimizer(lr)
error = loss(Y_hat, Y)
train = opt.minimize(error)

# Session init
# ============
gpu_op   = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
gpu_conf = tf.ConfigProto(gpu_options=gpu_op)
sess = tf.InteractiveSession(config=gpu_conf)
sess.run(tf.global_variables_initializer())


#-----------------------------------------------------------------------------#
#                                    utils                                    #
#-----------------------------------------------------------------------------#

# Model info & results
# ====================
def savestuff(name, err, data, dpath='/home/evan/.Data/Nbody/za_misc'):
    # path dir
    spath = f'{dpath}/{name}'
    if not os.path.exists(spath):
        os.makedirs(spath)
    # save
    np.save(f'{spath}/test_cubes', data)
    np.save(f'{spath}/test_error', err)
    print('saved to ' + spath)

def print_evaluation_results(err, label='Test', retstring=False):
    #==== Statistics
    err_avg = np.mean(err)
    err_std = np.std(err)
    err_median = np.median(err)
    #==== Text
    tbody = [f'\n# {label} Error\n# {"="*17}',
             f'  median : {err_median : .5f}',
             f'    mean : {err_avg : .5f} +- {err_std : .4f} stdv\n',]
    eval_results = '\n'.join(tbody)
    print(eval_results)
    if retstring:
        return eval_results

# Data batching
# =============
def get_feed_dict(x, b, i=None):
    """ get a batch of data to feed model
    x : input data
    b : int (batch_size)
    i : current index or iter
        i only used  for indexing algebra on val and test sets
    """
    if i is None:
        idx = np.random.choice(x.shape[0], b, replace=False)
    else: # non-training set
        idx = np.arange(i*b, (i+1)*b)
    _x_batch = np.copy(x[idx])
    x_za, x_fpm = _x_batch[...,:6], _x_batch[...,6:]
    return {X_in: x_za, Y: x_fpm}

# modeel feeding partials
get_train_feed = partial(get_feed_dict, X_train, i=None)
get_val_feed   = partial(get_feed_dict, X_val)
get_test_feed  = partial(get_feed_dict, X_test)


#-----------------------------------------------------------------------------#
#                                    Model                                    #
#-----------------------------------------------------------------------------#

# VALIDATION
def model_validation(bsize):
    nval = X_val.shape[0] // bsize
    val_hist = np.zeros((nval), dtype=np.float32)
    for i in range(nval):
        err = sess.run(error, feed_dict=get_val_feed(bsize, i))
        val_hist[i] = err
    return val_hist

# TEST
def model_test(bsize):
    ntest = X_test.shape[0] // bsize
    preds_shape = (2,) + X_test.shape[:-1] + (channels[-1],)
    test_hist  = np.zeros((ntest), dtype=np.float32)
    test_preds = np.zeros(preds_shape, dtype=np.float32)
    for i in range(ntest):
        j,k = i*bsize, (i+1)*bsize
        err, pred = sess.run([error, Y_hat], feed_dict=get_test_feed(bsize, i))
        test_hist[i] = err
        test_preds[1, j:k] = pred
    test_preds[0] = X_test[...,6:]
    print_evaluation_results(test_hist, 'Test')
    return test_hist, test_preds

# TRAIN
def model_train(num_iters, batch_size, chkpt):
    # Checkpoints (saving, info)
    num_checkpoints = num_iters // chkpt
    is_checkpoint = lambda i: (i+1) % chkpt == 0
    train_hist = np.zeros((num_checkpoints), dtype=np.float32) # val error during train

    # training loop
    for step in range(num_iters):
        fdict = get_train_feed(batch_size)
        train.run(feed_dict=fdict)

        if is_checkpoint(step):
            # check model performance on val set
            val_error = model_validation(batch_size)
            vmu = val_error.mean()
            print(f"{step+1:>6}: Validation Error = {vmu:.6f}")
            train_hist[(step+1) // chkpt - 1] = vmu
    return train_hist


#-----------------------------------------------------------------------------#
#                                     RUN                                     #
#-----------------------------------------------------------------------------#
cli = argparse.ArgumentParser()
cli.add_argument('-i', '--num_iters', type=int, default=num_iters)
cli.add_argument('-b', '--batch_size', type=int, default=batch_size)
cli.add_argument('-n', '--name', type=str, default='TEST')

def main():
    # training conf
    args = cli.parse_args()
    n_iters = args.num_iters
    bsize = args.batch_size
    sname = args.name

    # train and test
    train_hist = model_train(n_iters, bsize, 100)
    test_hist, test_preds  = model_test(bsize)

    # save results
    savestuff(sname, test_hist, test_preds)
    return 0

if __name__ == '__main__':
    main()


"""
####  NETWORK  ####
# explicit layer vars
A = tf.nn.relu
C = tf.nn.tanh

H0 = tf.layers.batch_normalization(A(set_layer(X_in, 0)))
R0 = C(res_layer(0))

H1 = tf.layers.batch_normalization(A(set_layer(H0 + R0, 1)))
R1 = C(res_layer(1))

H2 = tf.layers.batch_normalization(A(set_layer(H1 + R1, 2)))
R2 = C(res_layer(2))

H3 = tf.layers.batch_normalization(A(set_layer(H2 + R2, 3)))
R3 = C(res_layer(3))

Y_hat = set_layer(H3 + R3, 4)  # output
"""
