import os, code, sys, time
import numpy as np
import tensorflow as tf
#import chainer.functions as F
import matplotlib.pyplot as plt
import tf_utils
import tf_models
from tf_models import train

# training params
batch_size = 32
num_iters = 5000


# Sess
saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

loss_history = np.zeros((num_iters))
verbose = True

for i in range(num_iters):
    _x_batch = utils.next_minibatch(X_train, batch_size, data_aug=True)
    x_in   = _x_batch[0]
    x_true = _x_batch[1]

    if verbose:
        error = sess.run(loss, feed_dict={X_input: x_in, X_truth: x_true})
        loss_history[i] = error
        if i % 10 == 0:
            print('{}: {:.6f}'.format(i, error))
    train.run(feed_dict={X_input: x_in, X_truth: x_true})