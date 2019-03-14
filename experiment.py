import os
import code
import utils
import numpy as np
import tensorflow as tf

####  data  ####
didx = 0
num_test = 200
dataset = utils.Dataset(didx, num_test)
#code.interact(local=dict(globals(), **locals()))

####  vars  ####
lr = 0.01
channels = [6, 64, 128, 256, 32, 3]
#channels = [6, 64, 128, 256, 8, 3]
kdims = list(zip(channels[:-1], channels[1:]))
rdims = [(6, k) for _, k in kdims]
num_layers = len(kdims)

####  train  ####
num_iters  = 100000
batch_size = 10
N = 32**3

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
Wset = [init_weight(k) for k in kdims]
Rset = [init_weight(r) for r in rdims]
Bset = [init_bias(k) for k in kdims]


####  layers  ####
def res_layer(idx):
    """ skip connections
    All skips are from input to some hidden layer,
    which means all res_layer weights are of shape (6, S),
    where 6 is the num of input chans ([grid_pos, za_disp])
    and   S is the num of chans for its corresponding hidden layer
    """
    w = Rset[idx]
    xmu = tf.reduce_mean(X_in, axis=1, keepdims=True)
    h = X_in - xmu
    h_out = tf.einsum('bnk,kq->bnq', h, w)
    return h_out

def set_layer(h_in, idx):
    w, b = Wset[idx], Bset[idx]
    hmu = tf.reduce_mean(h_in, axis=1, keepdims=True)
    h = h_in - hmu
    h_out = tf.einsum('bnk,kq->bnq', h, w) + b
    return h_out

# Placeholders
X_in = tf.placeholder(tf.float32, shape=(None, N, 6))
Y = tf.placeholder(tf.float32, shape=(None, N, 3))
A = tf.nn.relu
C = tf.nn.tanh

####  NETWORK  ####
H0 = tf.layers.batch_normalization(A(set_layer(X_in, 0)))
R0 = C(res_layer(0))

H1 = tf.layers.batch_normalization(A(set_layer(H0 + R0, 1)))
R1 = C(res_layer(1))

H2 = tf.layers.batch_normalization(A(set_layer(H1 + R1, 2)))
R2 = C(res_layer(2))

H3 = tf.layers.batch_normalization(A(set_layer(H2 + R2, 3)))
R3 = C(res_layer(3))

H4 = set_layer(H3 + R3, 4)  # output

####  MODEL OPT  ####
opt = tf.train.AdamOptimizer(lr)
error = loss(H4, Y)
train = opt.minimize(error)

### Session setup ###
gpu_op = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
gpu_conf = tf.ConfigProto(gpu_options=gpu_op)
sess = tf.InteractiveSession(config=gpu_conf)
sess.run(tf.global_variables_initializer())


# Utils
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


def validation(pstats=False):
    X_val = np.copy(dataset.X_val) # (100, N, 9)
    num_val = X_val.shape[0] // batch_size
    val_hist = np.zeros((num_val), dtype=np.float32)
    for j in range(num_val):
        #_x_batch = dataset.get_minibatch(batch_size)
        p,q = j*batch_size, (j+1)*batch_size
        _x_batch = X_val[p:q]
        x_za = _x_batch[...,:6]
        x_fpm = _x_batch[...,6:]
        fdict = {X_in : x_za, Y : x_fpm}
        #train.run(feed_dict=fdict)

        err = sess.run(error, feed_dict=fdict)
        val_hist[j] = err
        #print(f"Validation batch {j+1 : >3} : {err:.6f}")
    if pstats:
        print_evaluation_results(val_hist, 'Validation')
    return val_hist


#### TRAIN ####
J = 100
chk = lambda i: (i+1) % J == 0
num_chks = num_iters // J

def train_model(num_iters, peval=True):
    train_hist = np.zeros((num_chks,), dtype=np.float32)
    for step in range(num_iters):
        _x_batch = dataset.get_minibatch(batch_size)
        x_za = _x_batch[...,:6]
        x_fpm = _x_batch[...,6:]
        fdict = {X_in : x_za, Y : x_fpm}
        train.run(feed_dict=fdict)

        if chk(step):
            #err = sess.run(error, feed_dict=fdict)
            #print(f"Checkpoint {step + 1 :>5} : {err:.6f}")
            # VALIDATION
            sp1 = step + 1
            vhist = validation()
            vmu = vhist.mean()
            if peval:
                print(f"{sp1: >5}\n")
                print_evaluation_results(vhist, label='Validation')
            else:
                print(f"{sp1 :>5}: Validation Error = {vmu:.6f}")
            train_hist[(sp1 // J)-1] = vmu
    return train_hist


####  TEST  ####
def test():
    X_test = np.copy(dataset.X_test) # (100, N, 9)
    M = X_test.shape[0]
    num_test = M // batch_size
    test_hist = np.zeros((num_test), dtype=np.float32)
    test_data = np.zeros((2,M, N, 3), dtype=np.float32)
    for j in range(num_test):
        #_x_batch = dataset.get_minibatch(batch_size)
        p,q = j*batch_size, (j+1)*batch_size
        _x_batch = X_test[p:q]
        x_za = _x_batch[...,:6]
        x_fpm = _x_batch[...,6:]
        fdict = {X_in : x_za, Y : x_fpm}
        #train.run(feed_dict=fdict)

        err, pred = sess.run([error, H4], feed_dict=fdict)
        test_hist[j] = err
        #code.interact(local=dict(globals(), **locals()))
        test_data[0,p:q] = x_fpm
        test_data[1,p:q] = pred
        #print(f"Test batch {j+1 : >3} : {err:.6f}")

    print_evaluation_results(test_hist, 'Test')
    return test_hist, test_data


def mkd(p):
    if not os.path.exists(p):
        os.makedirs(p)

dpath = '/home/evan/.Data/Nbody/za_misc'

def savestuff(name, err, data):
    spath = f'{dpath}/{name}'
    mkd(spath)
    np.save(f'{spath}/test_cubes', data)
    np.save(f'{spath}/test_error', err)
    print('saved to ' + spath)



if __name__ == '__main__':
    train_hist = train_model(num_iters, False)
    test_hist, test_data  = test()
    #sname = sys.argv[1]
    sname = 'resnet_bnorm_relu_r0_raw_100k'
    savestuff(sname, test_hist, test_data)


"""
# with selu activations (raw out)
  median :  1.49732
    mean :  1.51526 +-  0.0793 stdv


# ALL 10k
# Test Error  # batchnorm, relu, tanh, selu out
# =================
  median :  1.44911
    mean :  1.47064 +-  0.0719 stdv

# Test Error  # bnorm, selu, tanh, raw
# =================
  median :  1.48984
    mean :  1.50676 +-  0.0777 stdv

# Test Error  # bnorm, relu, tanh, raw
# =================
  median :  1.45148
    mean :  1.47033 +-  0.0725 stdv

# Test Error # 100k iters, batch_size 10, bnorm, relu, tanh
# =================
  median :  1.42991
    mean :  1.45038 +-  0.0726 stdv

"""
