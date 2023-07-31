import numpy as np
from DLB.util import im2col, col2im
from DLB.Layer import Convolution, Pooling
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

        self.convout_size = int((self.filter_size + 2*self.pad - self.filter_size)/self.stride + 1)


        self.params = {}
        #Conv
        self.params['W1'] = weight_init_std*np.random.randn(self.filter_num, self.input_dim[0], self.filter_size, self.filter_size)
        self.params['b1'] = np.zeros(self.filter_num)
        #Affine1
        self.params['W2'] = weight_init_std*np.random.randn(self.filter_num, self.input_dim[0], self.hidden_size, self.hidden_size)
        self.params['b2'] = np.zeros(self.filter_num)
        #Affine2
        self.params['W3'] = weight_init_std*np.random.randn(self.filter_num, self.input_dim[0], self.output_size, self.output_size)



    def predict(self, x):
        pass

    def loss(self, x):
        pass

    def accuracy(self, x, t):
        pass

    def gradient(self, x, t):
        pass