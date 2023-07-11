import numpy as np
from DLB.DeepLearningLB import softmax, cross_entropy_error
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