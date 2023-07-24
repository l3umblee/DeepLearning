#합성곱 계층 구현
'''
backward의 경우 affine 계층과 흡사하지만, im2col이 아닌 col2im을 사용해야하므로 col2im 구현 요망
'''
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        pass

    def forward(self, x):
        pass
    
    def backward(self, dout):
        pass