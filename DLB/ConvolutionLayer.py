import numpy as np
from DLB.util import im2col_JY, col2im
#합성곱 계층 구현
'''
backward의 경우 affine 계층과 흡사하지만, im2col이 아닌 col2im을 사용해야하므로 col2im 구현 요망

forward 중, N차원 텐서의 전치 관련
- 다차원 텐서의 전치는 numpy.transpose함수로 처리할 수 있음.
- 먼저 나온 결과 out을 일단, out.reshape(N, oh, ow, -1)로 reshape 해주는데, 이는 channel last 포맷이다.
  (channel  last 포맷 : NHWC는 픽셀단위로 채널의 값이 연속적으로 저장됨.
  tensor 연산은 효율적인 연산을 위해 병렬적으로 수행되는데, 채널 단위로 벡터화하여 연산이 수행되기 때문에 NHWC로 메모리에 저장하는 것이 메모리 엑세스 관점에서 효율적)
  -> 쉽게 말하면 NCHW 포맷은 채널별로 나뉘어져 있으니까 비효율적으로 값을 읽어와야 하지만, NHWC는 그렇지 않음

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

        col = im2col_JY(x, FH, FW) #input 데이터 x를 im2col를 통해 2차원 형상으로 바꿈
        c_filter = self.W.reshape(FN, -1).T #filter 또한 np.dot을 위해 2차원으로 변경, 두번째 인수가 -1인 이유는 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 묶어줌
        out = np.dot(col, c_filter) + self.b

        out = out.reshape(N, oh, ow, -1).transpose(0, 3, 1, 2) #N개의 데이터
        
        #오차 역전파 법을 위해 입력 데이터, 입력 데이터와 가중치를 affine 시킬 때의 버전을 저장해 둠.
        self.x = x
        self.col = col
        self.c_filter = c_filter
        
        return out
    #col2im 필요
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN) #channel last format으로 바꾼 다음, 열이 FN이 되도록 형 변환

        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.trasnpose(1, 0).reshape(FN, C, FH, FW)
        self.db = np.sum(dout, axis=0)

        dcol = np.dot(dout, self.c_filter.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx