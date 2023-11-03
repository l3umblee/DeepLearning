from EXfunctions import *

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = np.array([[0], [1], [1], [0]])

network = XORnet(2, 2, 1)

iters_num = 20000
learning_rate = 0.1

for i in range(iters_num):
    bmask = np.random.choice(x.shape[0], 4)
    x_train = x[bmask]
    t_train = t[bmask]

    g = network.gradient(x_train, t_train)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * g[key]

print("parameters : ", network.params)
print("predict : ", network.predict(x))