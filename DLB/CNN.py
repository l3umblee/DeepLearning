import numpy as np
from DLB.util import im2col_JY
#합성곱 계층 구현
'''
backward의 경우 affine 계층과 흡사하지만, im2col이 아닌 col2im을 사용해야하므로 col2im 구현 요망
'''
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad #pad는 역시 padding의 두께를 의미

    def forward(self, x): #x는 input 데이터, 필터는 self.W에 저장이 되어 있음
        FN, C, FH, FW = self.W.shape
        
        N, C, H, W = x.shape #N, C, H, W는 input 데이터의 shape를 따온 것인데, C는 어차피 필터의 것과 input의 데이터가 모두 같음
        oh = (H + 2*self.pad - FH) // self.stride + 1
        ow = (W + 2*self.pad - FW) // self.stride + 1

        col = im2col_JY(x) #input 데이터 x를 im2col를 통해 2차원 형상으로 바꿈
        c_filter = self.W.reshape(FN, -1).T #filter 또한 np.dot을 위해 2차원으로 변경, 두번째 인수가 -1인 이유는 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 묶어줌
        out = np.dot(col, c_filter) + self.b

        out = out.reshape(N, oh, ow, -1).transpose(0, 3, 1, 2) #N개의 데이터
        return out
    
    def backward(self, dout):
        pass