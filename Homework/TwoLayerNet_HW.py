import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from DLB.deeplearningFunctions import softmax, sigmoid, cross_entropy_error, _numerical_gradient

#2층 신경망 클래스 구현 -> 손글씨 인식을 위한 신경망
'''
구현해야 하는 함수
predict , loss, accuracy, numerical_gradient
'''
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        #내가 구현한 코드 -> 개선할 점은 정답은 1개이기 때문에 결과값 y와 정답값 t의 인덱스가 같은지만 확인하면 됨
        # accuracy_cnt = 0
        # for i in range(len(x)):        
        #     y = self.predict(x)

        #     p = np.argmax(y)
        #     if p == t[i]:
        #         accuracy_cnt += 1

        # return str(float(accuracy_cnt) / len(x))


    #library에 있는 _numerical_gradient와는 조금 다름
    def numerical_gradient(self, x, t):
        loss_W = lambda W:self.loss(x, t)
        grad = {}
        grad['W1'] = _numerical_gradient(loss_W, self.params['W1'])
        grad['b1'] = _numerical_gradient(loss_W, self.params['b1'])
        grad['W2'] = _numerical_gradient(loss_W, self.params['W2'])
        grad['b2'] = _numerical_gradient(loss_W, self.params['b2'])
        #**|중요!|** = 가중치 매개변수와 '편향'들까지도 각각의 기울기를 구해줘야 한다.
        return grad