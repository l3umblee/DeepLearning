import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import progressbar
from dataset.mnist import load_mnist
from TwoLayerNet_BP_HW import TwoLayerNet_BP

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet_BP(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1)

bar = progressbar.ProgressBar(maxval=iters_num).start()

for i in range(iters_num):
    bar.update()
    batch_mask = np.random.choice(train_size, batch_size)
    x = x_train[batch_mask]
    t = t_train[batch_mask]
    
    #기울기 구하기
    g = network.gradient(x, t)

    #갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * g[key]

    loss = network.loss(x, t)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

bar.finish()    
x_list = np.arange(len(train_acc_list))
plt.plot(x_list, test_acc_list, label="test_acc")
plt.plot(x_list, train_acc_list, label='train_acc')
plt.ylim(0, 1.0)
plt.show()