import numpy as np
import cupy
import chainer
from chainer import cuda
import code
import chainer.serializers as serializers
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_utils import *
'''
Utils
'''
BOUND = 0.095
LR = 0.01
class ModelTrainer():# should really look into chainer's trainer class
    def __init__(self, model, opt, save_path='./Evaluation/weights/', n_iters=1000, use_graph=False):
        self.model = model
        self.opt = opt
        self.save_path = save_path
        self.n_iters = 1000
        self.loss_history = xp.zeros((n_iters))
        self.rng_seed = 98765
        self.use_graph = use_graph
    
    def seed_rng(self):
        xp.random.seed(self.rng_seed)
        np.random.seed(self.rng_seed)
    
    def extract_data(self, data):
        X, Y = data
        return xp.copy(X), xp.copy(Y)
    
    def train(self, X, Y, mb_size):
        self.seed_rng()
        for cur_iter in range(self.n_iters):
            self.model.zerograds()
            _x_in, _x_true = gpunext_minibatch([X, Y], mb_size)
            x_in, x_true = to_var_xp([_x_in, _x_true])
            graphNN = None
            if self.use_graph:
                graphNN = graph_ops.GraphNN(cuda.to_cpu(x_in), 'knn', 14)
            x_hat = self.model(x_in, graphNN=graphNN)
            loss = nn.get_MSE(x_hat, x_true, boundary=BOUND)
            loss.backward()
            self.opt.update()
            self.loss_history[cur_iter] = loss.data
        x_in   = cuda.to_cpu(x_in.data)
        x_hat  = cuda.to_cpu(x_hat.data)
        x_true = cuda.to_cpu(x_true.data)
        save_model((self.model, self.opt), self.save_path)
        return (x_in, x_hat, x_true), cuda.to_cpu(self.loss_history)
    
    def validation(self, X, Y, mb_size):
        M, N, D = X.shape
        assert M % mb_size == 0
        epoch_len = M // mb_size
        validation_history = xp.zeros((epoch_len))
        with chainer.using_config('train', False):
            for i in range(epoch_len):
                start_idx, end_idx = (i*mb_size, (i+1)*mb_size)
                _x_in = X[start_idx:end_idx]
                _x_true = Y[start_idx:end_idx]
                x_in, x_true = to_var_xp([_x_in, _x_true])
                graphNN = None
                if self.use_graph:
                    graphNN = graph_ops.GraphNN(cuda.to_cpu(x_in), 'knn', 14)
                x_hat = self.model(x_in, graphNN=graphNN)
                loss = nn.get_MSE(x_hat, x_true, boundary=bound)
                validation_history[i] = loss.data
            x_in   = cuda.to_cpu(x_in.data)
            x_hat  = cuda.to_cpu(x_hat.data)
            x_true = cuda.to_cpu(x_true.data)
            return (x_in, x_hat, x_true), cuda.to_cpu(validation_history)
        
    def __call__(self, data, is_validation=False, mb_size=8):
        X, Y = self.extract_data(data)
        if is_validation:
            return self.validation(X, Y, mb_size)
        else:
            return self.train(X, Y, mb_size)