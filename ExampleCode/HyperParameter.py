import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from DLB.MultiLayerNet import MultiLayerNet
from DLB.Trainer import Trainer
from dataset.mnist import load_mnist
from DLB.util import shuffle_dataset
#스탠퍼드 대학교의 CS231n 수업
'''
학습률과 가중치 감소의 세기를 조절하는 계수를 탐색 (가중치 감소 계수 - lambda로 알고 있는 것)
'''
(x_train, t_train), (x_test, t_test) = load_mnist()
x_train, t_train =  shuffle_dataset(x_train, t_train)

validation_rate = 0.20
validation_num = int(x_train.shape[0]*validation_rate)

#훈련 데이터와 검증 데이터를 나눔 -> 이전에도 배웠듯이 검증 데이터는 오버 피팅을 막기 위한 수단
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# 하이퍼파라미터 무작위 탐색======================================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 탐색한 하이퍼파라미터의 범위 지정===============
    weight_decay = 10 ** np.random.uniform(-8, -4)
    #uniform은 numpy에서 제공하는 균등 분포 함수 (최소 -8 ~ 최대 -4 중에서 한가지를 뽑음 -> 이는 가중치 감소의 값이 될 것임.)
    lr = 10 ** np.random.uniform(-6, -2)
    #learing rate, 학습률 또한 무작위로 선정
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay) #선정한 파라미터들을 이용해서 훈련 시킴
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 그래프 그리기========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()