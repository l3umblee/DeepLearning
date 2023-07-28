import sys, os
sys.path.append(os.pardir)
import numpy as np
from DLB.util import im2col_JY

input_data = np.arange(147).reshape(1, 3, 7, 7)
input_data2 = input_data
col1 = im2col_JY(input_data, 5, 5, stride=1, pad=0)

N, C, H, W = input_data.shape
filter_h = 5
filter_w = 5
stride = 1
pad = 0

oh = (H + 2*pad - filter_h) // stride + 1
ow = (W + 2*pad - filter_w) // stride + 1

col2 = np.zeros((N, C, filter_h, filter_w, oh, ow))

for y in range(filter_h):
    y_max = y + stride*oh
    for x in range(filter_w):
        x_max = x + stride*ow
        col2[:,:,y, x, :, :] =  input_data2[:, :, y:y_max:stride, x:x_max:stride]

print("im2col_JY의 결과")
print(col1)
print("-"*30)
print("im2col의 원래 버전, transpose 하기 이전, 6차원의 결과")
print(col2)
print("-"*30)
col2 = col2.transpose(0, 4, 5, 1, 2, 3).reshape(N*oh*ow, -1)
print("transpose 한 버전")
print(col2)