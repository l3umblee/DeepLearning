import numpy as np

#AND 게이트 (단일 퍼셉트론) (weight와 bias 이용)
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#OR 게이트 (단일 퍼셉트론)
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.9, 0.9])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#NAND 게이트 (단일 퍼셉트론)
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#XOR 게이트 (배타적 논리합 -> 다층 퍼셉트론)
def XOR(x1, x2):
    z1 = OR(x1, x2)
    z2 = NAND(x1, x2)
    return AND(z1, z2)

#시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#ReLU 함수
def relu(x):
    return np.maximum(0, x)

#소프트맥스 함수
'''
e 제곱의 값이 커질 경우, inf를 반환

오버플로우를 막기위해 다음과 같은 성질을 사용
yk = exp(ak) / sum(exp(ai)) = C*exp(ak) / C*(sum(exp(ai)))
 = exp(ak + logC) / sum(exp(ai + logC)) = exp(ak + C') / sum(exp(ai + C'))
(C' = logC)

-> 입력값에 어떤 수를 더하거나 빼도 결과는 같다.
'''
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y