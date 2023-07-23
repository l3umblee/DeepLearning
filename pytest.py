import sys, os
sys.path.append(os.pardir)
import numpy as np
from DLB.util import im2col_JY

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col_JY(x1, 5, 5, stride=1, pad=0)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col_JY(x2, 5, 5, stride=1, pad=0)
print(col2.shape)