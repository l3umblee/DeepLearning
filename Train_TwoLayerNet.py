import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from TwoLayerNet_HW import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#시간이 너무 오래 걸리므로 원래는 10000번이지만, 줄였음
#-> DeepLearningLB에 구현된 _numerical_gradient가 while문으로 수치미분을 진행하므로 오래 걸림

iters_num = 10
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

progress_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    progress_[i] = '-'
    print("\r{0}".format(progress_), end='')
    #미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x = x_train[batch_mask]
    t = t_train[batch_mask]

    #기울기 계산
    g = network.numerical_gradient(x, t)

    #매개변수 갱신
    network.params['W1'] -= learning_rate*g['W1']
    network.params['b1'] -= learning_rate*g['b1']
    network.params['W2'] -= learning_rate*g['W2']
    network.params['b2'] -= learning_rate*g['b2']

    #학습 경과 기록
    loss = network.loss(x, t)
    train_loss_list.append(loss)

    if network.accuracy(x, t) >= 0.95:
        break

x_list = list(range(0, iters_num))
plt.plot(x_list, train_loss_list)
plt.show()