import sys, os
sys.path.append(os.pardir)
import numpy as np
from DLB.deeplearningFunctions import *
from DLB.util import im2col_JY
from DLB.Layer import Affine, Sigmoid, Relu
from collections import OrderedDict

class XORnet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):
        self.params = {}
        #초깃값 설정 - ppt자료(25p) 참고
        self.params['W1'] = np.array([[1.0, -1.0], [1.0, -1.0]])
        self.params['b1'] = np.array([-0.5, 1.5])
        self.params['W2'] = np.array([[1.0], [1.0]])
        self.params['b2'] = np.array([-1.0])

        self.Layers = OrderedDict()
        
        # Sigmoid : Gradient vanishing problem 발생

        self.Layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.Layers['Sigmoid1'] = Sigmoid()
        self.Layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.Layers['Sigmoid2'] = Sigmoid()

        # self.Layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # self.Layers['ReLU1'] = Relu()
        # self.Layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # self.Layers['ReLU2'] = Relu()

    def predict(self, x):
        for l in self.Layers.values():
            x = l.forward(x)

        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        self.loss_func = mean_squared_error(y, t)

        return self.loss_func
    
    def gradient(self, x, t):
        dout = self.loss(x, t)
        dout = np.array([[dout]])
        layers = list(self.Layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.Layers['Affine1'].dW
        grads['b1'] = self.Layers['Affine1'].db
        grads['W2'] = self.Layers['Affine2'].dW
        grads['b2'] = self.Layers['Affine2'].db

        return grads
    
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = np.array([[0], [1], [1], [0]])

iters_num = 100
learning_rate = 0.1

network = XORnet(2, 2, 1, learning_rate)
for i in range(iters_num):
    bmask = np.random.choice(x.shape[0], 1)
    x_train = x[bmask]
    t_train = t[bmask]

    g = network.gradient(x_train, t_train)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * g[key]
        #network.params[key] = np.round(network.params[key], 3)

print(network.params)

# bmask = np.random.choice(x.shape[0], 2)
# x_test = x[bmask]
# print(x_test)
print(x)
print("predict : ", network.predict(x))
# print("loss_function : ", network.loss_func)