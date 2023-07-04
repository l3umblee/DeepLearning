import sys, os
sys.path.append(os.pardir)
import numpy as np
from DLB.DeepLearningLB import softmax, cross_entropy_error, numerical_gradient

'''
simpleNet (단순한 신경망)

멤버 변수
self.W -> 가중치 매개변수

메서드
preidct(self, x) : 가중치 매개변수와 입력 데이터를 내적 -> 단순한 1층 신경망
loss(self, x, t) : z는 내적의 결과값, y는 z를 softmax에 통과시킨 예측값 -> loss는 y와 t를 교차 엔트로피 오차 구한 것


'''
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) #->가우시안 표준 정규 분포에서 난수 matrix array 생성
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))

t = np.array([0, 0, 0])
for idx in range(0, len(t)):
    if idx == np.argmax(p):
        t[idx] = 1

print(t)
print(net.loss(x, t))

f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)