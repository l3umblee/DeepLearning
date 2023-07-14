import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from DLB.util import smooth_curve
from dataset.mnist import load_mnist
from Homework.TwoLayerNet_BP_HW import TwoLayerNet_BP
from DLB.Optimizer import SGD, Momentum, AdaGrad, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet_BP(input_size=784, hidden_size=50, output_size=10)

iters_num = 2000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size/batch_size, 1)

optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

LayerNets = {}
train_acc = {}
for key in optimizers.keys():
    LayerNets[key] = TwoLayerNet_BP(input_size=784, hidden_size=100, output_size=10)
    train_acc[key] = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x = x_train[batch_mask]
    t = t_train[batch_mask]
    
    #갱신
    for key in optimizers.keys():
        grads = LayerNets[key].gradient(x, t)
        optimizers[key].update(LayerNets[key].params, grads)
        acc = LayerNets[key].accuracy(x, t)
        train_acc[key].append(acc)

x_list = np.arange(iters_num)
for key in optimizers.keys():
    plt.plot(x_list, smooth_curve(train_acc[key]), label=key)
plt.legend()
plt.ylim(0, 1.0)
plt.show()