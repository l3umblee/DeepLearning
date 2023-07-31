import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from DLB.Layer import SoftmaxWithLoss, Affine, Relu, Pooling, Convolution
from collections import OrderedDict
'''
합성곱 신경망 구현

초기화 인수
input_dim : 입력 데이터 (채널 수, 높이, 너비)
conv_param : 합성곱 계층의 하이퍼 파라미터 (딕셔너리 형태로 되어 있음)
    -filter_num : 필터수 / filter_size : 필터 크기 / stride : 스트라이드 / pad : 패딩 
    / hidden_size : 은닉층의 뉴런 수 / output_size : 출력층의 뉴런 수 / weight_init_std : 초기화 때의 가중치 표준편차

계층 순서 : Conv - ReLU - Pooling - Affine - ReLU - Affine - Softmax
'''

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num':30, 'filter_size' : 5, 'stride':1, 'pad':0}, hidden_size=100, output_size=10, weight_init_std=0.01):
        self.input_dim = input_dim
        self.filter_num = conv_param['filter_num']
        self.filter_size = conv_param['filter_size']
        self.stride = conv_param['stride']
        self.pad = conv_param['pad']
        self.input_size = input_dim[1] #어차피 정사각형

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.convout_size = (self.input_size + 2*self.pad - self.filter_size)/self.stride + 1
        self.poolout_size = int(self.filter_num*(self.convout_size/2)*(self.convout_size/2))
        #poolout_size를 이런 식으로 계산한 이유는 4등분해서 하겠다는 이야기

        self.params = {}
        #Conv
        self.params['W1'] = weight_init_std*np.random.randn(self.filter_num, self.input_dim[0], self.filter_size, self.filter_size)
        self.params['b1'] = np.zeros(self.filter_num)
        #Affine1
        self.params['W2'] = weight_init_std*np.random.randn(self.poolout_size, self.hidden_size)
        self.params['b2'] = np.zeros(self.hidden_size)
        #Affine2
        self.params['W3'] = weight_init_std*np.random.randn(self.hidden_size, self.output_size)
        self.params['b3'] = np.zeros(self.output_size)

        self.layers = OrderedDict()
        self.layers['Conv'] = Convolution(self.params['W1'], self.params['b1'], self.stride, self.pad)
        self.layers['ReLU'] = Relu()
        self.layers['Pooling'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.last_layer.forward(y, t)

    #accuracy : 코드 발췌
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv'].dW
        grads['b1'] = self.layers['Conv'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        
        return grads
    
    def save_params(self, file_name='params.pkl'):
        params={}
        for key, val in self.params.items():
            self.params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        
        for i, key in enumerate(['Conv', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
            