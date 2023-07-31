import sys, os
sys.path.append(os.pardir)
import numpy as np
from DLB.deeplearningFunctions import _numerical_gradient
from DLB.Layer import SoftmaxWithLoss, Affine, Relu
from collections import OrderedDict

#2층 신경망 구현 -> 오차 역전파법 이용, OrderDict 이용
'''
구현해야 하는 메서드
predict, loss, accuracy, numerical_gradient, gradient
'''
class TwoLayerNet_BP:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        np.random.seed(23)
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        self.Layers = OrderedDict()
        self.Layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.Layers['Relu1'] = Relu()
        self.Layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.OutputLayer = SoftmaxWithLoss()

    #x의 값은 데이터 N개에 대한 입력.
    def predict(self, x): 
        for l in self.Layers.values():
            x = l.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.OutputLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W:self.loss(x, t)

        grad = {}
        grad['W1'] = _numerical_gradient(loss_W, self.params['W1'])
        grad['b1'] = _numerical_gradient(loss_W, self.params['b1'])
        grad['W2'] = _numerical_gradient(loss_W, self.params['W2'])
        grad['b2'] = _numerical_gradient(loss_W, self.params['b2'])

        return grad

    def gradient(self, x, t):
        self.loss(x, t) #predict-> SoftmaxWithLoss.forward / forward 메서드를 통해 역전파에 필요한 값 설정됨

        dout = 1
        dout = self.OutputLayer.backward(dout)

        layers = list(self.Layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grad = {}
        grad['W1'] = self.Layers['Affine1'].dW
        grad['b1'] = self.Layers['Affine1'].db
        grad['W2'] = self.Layers['Affine2'].dW
        grad['b2'] = self.Layers['Affine2'].db

        return grad