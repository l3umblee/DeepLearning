import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet_HW import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    pass
    #미니배치 획득

    #기울기 계산

    #매개변수 갱신

    #학습 경과 기록

    #1에폭당 정확도 계산