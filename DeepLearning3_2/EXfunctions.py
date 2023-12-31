import numpy as np
from collections import OrderedDict
'''
수업시간에 배운 개념들을 직접 구현
'''

#Affine : weighted sum을 계산하기 위함.
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

    #forward : 순전파, weighted sum 계산
    def forward(self, x):
        # self.original_x_shape = x.shape
        # x = x.reshape(x.shape[0], -1)
        # self.x = x
        # out = np.dot(self.x, self.W) + self.b
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    #backward : 오차역전파를 이용하여 W와 b에 대한 gradient 계산
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        #dx = dx.reshape(*self.original_x_shape)
        return dx

#Sigmoid : 활성화 함수 중 sigmoid function
class Sigmoid:
    def __init__(self):
        self.out = None

    #forward : Sigmoid 통과시킴    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    #backward : y(1-y) 수행, 연쇄법칙에 따라 dout에 곱하기
    def backward(self, dout):
        dx = dout*self.out*(1-self.out)

        return dx

class MSE:
    def __init__(self):
        self.y = None
        self.t = None
    
    def forward(self, y, t):
        self.y = y
        self.t = t

        out = mean_squared_error(y, t)
        return out

    def backward(self, dout):
        dx = dout*(self.y - self.t)

        return dx

def mean_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)

class XORnet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):
        self.params = {}        
        self.params['W1'] = np.array([[1.0, -1.0], [1.0, -1.0]])
        self.params['b1'] = np.array([-0.5, 1.5])
        self.params['W2'] = np.array([[1.0], [1.0]])
        self.params['b2'] = np.array([1.0])

        self.Layers = OrderedDict()
        self.Layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.Layers['Sigmoid1'] = Sigmoid()
        self.Layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.Layers['Sigmoid2'] = Sigmoid()
        
        self.LastLayers = MSE()

    def predict(self, x):
        for layer in self.Layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        self.loss_func = self.LastLayers.forward(y, t)

        return self.loss_func
    
    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.LastLayers.backward(dout)

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
    
#BatchNormalization 구현
'''
gamma, beta 값을 초기에 받을 수 있어야 하며, 이를 갱신할 때 필요한 moving average 정도를 위해 momentum 매개변수 추가
'''
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum

        #test 시에 필요한 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var
    
    def forward(self, x):
        mean = np.mean(x, axis=1) # 행에 대해서 평균 계산
        var = np.var(x, axis=1) # 행에 대해서 분산 계싼


    def backward(self):
        pass
