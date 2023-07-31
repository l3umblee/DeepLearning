import sys, os
sys.path.append(os.pardir)
import numpy as np
from DLB.util import im2col_JY

x = np.array([[1, 2, 0, 1], [3, 0, 2, 4], [1, 0, 3, 2], [4, 2, 0, 1]])
arg_max = np.argmax(x, axis=1)
print(arg_max)
print(x)
print(x.shape)
print(x.shape + (4, ))

arg_max = np.argmax(x, axis=1)
y = np.zeros_like(x)
y[np.arange(arg_max.size), arg_max.flatten()] = arg_max.flatten()
print(y)