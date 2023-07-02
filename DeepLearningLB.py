import numpy as np

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