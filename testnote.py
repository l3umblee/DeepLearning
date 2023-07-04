import sys, os
sys.path.append(os.pardir)
import numpy as np
import DeepLearningLB as DL

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
        self.W = np.array([[0.47355232, 0.9977393, 0.84668094], 
                           [0.8555741, 0.03563661, 0.69422093]]) #->가우시안 표준 정규 분포에서 난수 matrix array 생성
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = DL.softmax(z)
        loss = DL.cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))

t = np.array([0, 0, 1])
l = net.loss(x, t)
print(l)