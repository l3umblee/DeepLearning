import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
import DeepLearningLB as DL
from dataset.mnist import load_mnist

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

'''
-유의-
pickle 파일인 'sample_weight.pkl'에는 학습된 가중치 매개변수와 편향이 저장되어 있음
keras 모델을 만들 때 필요한 것이 아니라 신경망 사이의 계산에서 필요한, 더 low level을 생각했을 때 필요한 것
--> 궁금한 점 : keras 모델에서는 이러한 매개변수들이 없이 어떻게 작동하는 것일까?
'''
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x): 
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = DL.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = DL.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = DL.softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])

    p= np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
