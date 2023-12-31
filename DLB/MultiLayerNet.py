import numpy as np
from collections import OrderedDict
from DLB.deeplearningFunctions import _numerical_gradient
from DLB.Layer import *
#MultiLayerNet : 완전 연결 다층 신경망
'''
구현해야 할 것
Parameters
-------------------------------------------------------
input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    weight_decay_lambda : 가중치 감소(L2 법칙)의 세기'

Method
-------------------------------------------------------
init_weight(weight_init_std) : 가중치 초기화 메서드
predict(x) : 입력값에 대한 예측값 반환
loss(x, t) : 입력값과 정답 레이블에 대한 손실함수의 값 반환
accuracy(x, t) : 정확도 계산
numerical_gradient(x, t) : 수치 미분을 이용한 기울기 반환
gradient(x, t) : 오차역전파법을 이용한 기울기 반환
'''
class MultiLayerNet:
    #생성자 정의
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.hidden_size_list_num = len(hidden_size_list)
        self.output_size = output_size
        self.weight_init_std = weight_init_std
        self.weight_decay_lambda = weight_decay_lambda #오버피팅 억제를 위한 가중치 감소
        self.params = {}

        self.init_weight(weight_init_std)
        activation_layer = {'relu' : Relu(), 'sigmoid' : Sigmoid()}
        self.layers = OrderedDict()
        for i in range(1, self.hidden_size_list_num+1):
            self.layers['Affine' + str(i)] = Affine(self.params['W'+str(i)],
                                                     self.params['b'+str(i)]) #Affine의 매개변수는 W, b
            self.layers['Activation Func' + str(i)] = activation_layer[activation]
        
        #SoftmaxwithLoss를 위하여 마지막 층은 for문과 따로 정의
        idx = self.hidden_size_list_num+1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], 
                                                  self.params['b' + str(idx)])
        #self.layers에는 마지막의 내적 층 까지만 저장, SoftmaxWithLoss를 위한 last_layer는 별개이므로 주의
        self.last_layer = SoftmaxWithLoss()
    
    #가중치 초기화
    def init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for i in range(1, len(all_size_list)):
            self.params['W' + str(i)] = np.random.randn(all_size_list[i-1], all_size_list[i]) #W1은 input_size x hidden_size1, W마지막은 hidden_size마지막 x output_size
            if weight_init_std == 'relu' or weight_init_std == 'he':
                self.params['W' + str(i)] *= np.sqrt(2.0 / all_size_list[i])
            elif weight_init_std == 'sigmoid' or weight_init_std == 'xavier':
                self.params['W' + str(i)] *= np.sqrt(1.0 / all_size_list[i])
            else:
                self.params['W' + str(i)] *= self.weight_init_std
            
            self.params['b' + str(i)] = np.zeros_like(all_size_list[i])
    
    #예측값 계산
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    #손실함수 계산
    def loss(self, x, t):
        y = self.predict(x)
        
        weight_decay = 0
        for idx in range(1, self.hidden_size_list_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2) 
            #"가중치 감소는 모든 가중치 각각의 손실함수에 1/2 * 람다 * W^2를 더합니다."
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    #수치 미분
    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)

        grads = {}
        for i in range(1, self.hidden_size_list_num+2):
            grads['W'+str(i)] = _numerical_gradient(loss_W, self.params['W'+str(i)])
            grads['b'+str(i)] = _numerical_gradient(loss_W, self.params['b'+str(i)])
        
        return grads

    #오차 역전파법을 이용한 기울기
    def gradient(self, x, t):
        self.loss(x, t) #forward를 통해 오차역전파를 위한 모든 값 세팅

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for i in range(1, self.hidden_size_list_num+2):
            #가중치 감소 적용
            grads['W'+str(i)] = self.layers['Affine'+str(i)].dW + self.weight_decay_lambda * self.layers['Affine' + str(i)].W
            grads['b'+str(i)] = self.layers['Affine'+str(i)].db
        
        return grads