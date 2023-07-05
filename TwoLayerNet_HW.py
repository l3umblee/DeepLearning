import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from DLB.DeepLearningLB import softmax

#2층 신경망 클래스 구현 -> 손글씨 인식을 위한 신경망

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def accuracy():
        pass