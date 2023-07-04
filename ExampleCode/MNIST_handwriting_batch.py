import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from DLB.DeepLearningLB import sigmoid, softmax
from dataset.mnist import load_mnist

'''
[배치 (batch)]
이미지를 여러 장 묶어서 predict() 함수에 넘길 경우를 생각해보는 것
각 픽셀들에 가중치가 곱해자고 편향이 더해지는 것이므로 상관이 없다.

다만 이처럼 여러 개의 데이터를 하나로 묶은 입력 데이터를 '배치'라고 함.
'''

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x): 
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch) #y_batch에는 각 이미지에 대한 확률이 들어 있음, 하지만 2차원 배열로 batch_size 만큼 겹쳐져 있음
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt)/len(x)))