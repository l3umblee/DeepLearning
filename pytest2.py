import numpy as np

list1 = [0, 1, 2]
x = np.reshape(np.array(range(30)), (1, 10, 3))
print(x.shape)
print(x[:,list1,:])

print(np.mean(x[:,list1,:], axis=1))