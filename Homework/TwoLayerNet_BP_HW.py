import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from DLB.DeepLearningLB import softmax, sigmoid, cross_entropy_error, _numerical_gradient

#2층 신경망 구현 -> 오차 역전파법 이용
class TwoLayerNet_BP:
    def __init__(self):
        pass