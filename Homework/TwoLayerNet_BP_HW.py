import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from DLB.DeepLearningLB import softmax, sigmoid, cross_entropy_error, _numerical_gradient
from DLB.BackPropagationLayer import SoftmaxWithLoss
from collections import OrderedDict

#2층 신경망 구현 -> 오차 역전파법 이용, OrderDict 이용
'''
구현해야 하는 메서드
predict, loss, accuracy, numerical_gradient, gradient
'''
class TwoLayerNet_BP:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    #x의 값은 데이터 N개에 대한 입력.
    def predict(self, x): 
        pass

    def loss(self, x, t):
        pass

    def accuracy(self, x, t):
        pass

    def numerical_gradient(self, x, t):
        pass

    def gradient(self, x, t):
        pass