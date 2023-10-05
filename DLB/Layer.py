import numpy as np
from DLB.deeplearningFunctions import *
from DLB.util import *
#곱셈 계층
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y

        return out

    def backward(self, dout):
        dx = dout*self.y
        dy = dout*self.x

        return dx, dy

#덧셈 계층
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y
        
        return out

    def backward(self, dout):
        dx = dout*1
        dy = dout*1

        return dx, dy
    
#ReLU 계층
'''
Q. 역전파에서의 mask를 왜 순전파의 mask 그대로 사용할까? 그 뒤의 연산에서 0보다 작았던 원소가 0보다 더 크게 바뀐다면?
A. 현재로서 드는 생각은 ReLU 계층을 통과시키면 그 부분은 0이 됨 -> 필요하지 않았던 부분...
'''
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
#sigmoid 계층
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout*self.out*(1-self.out)

        return dx
    
#Affine 계층 (텐서 고려 x)
class _Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        #dW를 계산할 때 x의 입력값이 필요하기 때문에 변수로 가지고 있어야 함.
        self.x = None
        #가중치와 편향의 편미분
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        '''
        (편향의 편미분)

        - 순전파의 편향 덧셈은 각각의 데이터에 더해지기 때문에 역전파 때는 각 데이터의 역전파 값이 편향의 원소에 모여야 함.
        - dout의 첫 번째 축(0축, 열방향)의 합 (아래로 쭉 더하는 것)
        '''

        return dx

#Affine 계층 (텐서 고려 O)
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx

#SoftmaxWithLoss (My ver.)
class SoftmaxWithLoss_JY:
    def __init__(self):
        self.loss = None
        self.S = None
        self.Y = None

    def forward(self, a, t):
        self.a = a
        self.t = t
        #Softmax
        if a.ndim == 1:
            self.Y = softmax(a)
        else:
            c = np.max(self.a, axis=1)
            self.a = np.exp(self.a - c)
            self.S = np.sum(self.a, axis=1) #Sum 행렬 -> 1행당 데이터 1개에 대한 예측값을 모두 더함. 열은 당연히 1개
            self.Y = self.a / self.S
        
        #cross_entropy_error
        L = cross_entropy_error(self.Y, t)

        return L

    def backward(self):
        out = self.Y - self.t

        return out

#SoftmaxWithLoss (Standard Ver.)
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: #정답 레이블이 원-핫 인코딩일 경우
            dx = (self.y - self.t) / batch_size
        else: #정답레이블이 원-핫 인코딩이 아니라면 (여기서 원-핫 인코딩으로 바꿔준다고 생각)
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1 #정답에 해당하는 요소들은 모두 1로 바꿔줌.
            dx = dx / batch_size
        #전파하는 값을 배치의 수로 나눠 데이터 1개당 오차를 앞 계층으로 전파
        return dx
    
#Dropout
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        #일반적인 경우 dropout_ratio는 0.5로 지정
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            #dropout_ratio보다 큰 원소만 True로 설정한 것 -> random에서 뽑아서 dropout_ratio보다 큰 확률의 위치만 True로 표시한 것
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        #역전파 시에는 순전파 때 통과시킨 건 통과시키고, 통과시키지 않은 것은 차단 -> 순전파의 과정을 그대로 반영
        return dout * self.mask
    
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

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x): #x는 input 데이터, 필터는 self.W에 저장이 되어 있음
        FN, C, FH, FW = self.W.shape
        
        N, C, H, W = x.shape #N, C, H, W는 input 데이터의 shape를 따온 것인데, C는 어차피 필터의 것과 input의 데이터가 모두 같음
        # oh = (H + 2*self.pad - FH) // self.stride + 1
        # ow = (W + 2*self.pad - FW) // self.stride + 1
        oh = 1 + int((H + 2*self.pad - FH) / self.stride)
        ow = 1 + int((W + 2*self.pad - FW) / self.stride)

        #col = im2col_JY(x, FH, FW) #input 데이터 x를 im2col를 통해 2차원 형상으로 바꿈
        col = im2col(x, FH, FW)
        c_filter = self.W.reshape(FN, -1).T #filter 또한 np.dot을 위해 2차원으로 변경, 두번째 인수가 -1인 이유는 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 묶어줌
        out = np.dot(col, c_filter) + self.b

        out = out.reshape(N, oh, ow, -1).transpose(0, 3, 1, 2) #N개의 데이터
        
        #오차 역전파 법을 위해 입력 데이터, 입력 데이터와 가중치를 affine 시킬 때의 버전을 저장해 둠.
        self.x = x
        self.col = col
        self.col_W = c_filter
        
        return out
    
    #col2im 필요
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN) #channel last format으로 바꾼 다음, 열이 FN이 되도록 형 변환

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

#Pooling 계층이며, max pooling(최대 풀링 기법을 사용)
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        oh = int(1 + (H - self.pool_h) / self.stride)
        ow = int(1 + (W - self.pool_w) / self.stride)
    
        #col = im2col_JY(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1) #한 행 중 가장 큰 요소
        out = out.reshape(N, oh, ow, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1) #channel last 포맷으로 변환

        pool_size = self.pool_h*self.pool_w #pool_size는 일단, forward 상에서 im2col 바꿨을 당시의 열의 개수임. 즉 Pooling 계층을 통과했을 때 풀 1개에 대한 후보들의 개수
        dmax = np.zeros((dout.size, pool_size)) #dout의 사이즈는 원래의 데이터가 pool로 묶었을 때 총 몇 개 있었는지 나타냄 + 열을 pool_size로 둠으로써 forward 당시의 행렬을 영행렬로 구현
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        #arg_max는 처음 forward의 input_data에서 구한 것, arg_max의 사이즈는 결국 원래 데이터에서 풀링한 것의 묶음 개수 / 평탄화한 다음은 dmax 상에서 원래 max 원소들이 있었던 곳을 열로 나타냄
        dmax = dmax.reshape(dout.shape + (pool_size,)) #차원을 dout.shape에서 pool_size만큼 늘림.
        #pool_size 만큼 차원을 늘려준 이유는 현재 dout은 forward 한 상태인데, 여기서 차원을 세로로 추가한다고 생각, 원본의 크기로 되돌리는 의미
        dcol = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2], -1) #다시 한 행으로 만듦 (col2im을 위해)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        #pooling에서 stride는 pooling window 크기와 같은 값으로 설정하는 것이 보통!
        return dx