import numpy as np
class Dropout:
    def __init__(self, dropout_ratio=0.5):
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
        #역전파 시에는 순전파 때 통과시킨 건 통과시키고, 통과시키지 않은 것은 차단
        return dout * self.mask