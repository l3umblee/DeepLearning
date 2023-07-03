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

#평균 제곱 오차 (손실함수) -> t에 해당하는 값은 one-hot-encoding된 데이터
def mean_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)

#교차 엔트로피 오차 (손실함수)
'''
(해석)
t는 정답 (타깃).
log x의 그래프는 x가 1일 때 0이 되고, x가 0에 가까워질수록 y의 값이 점점 작아짐 (음수)
여기에 -를 취했으니, 결과적으로는 정답에 가까울수록 값이 작아짐.

-> 여러 개의 데이터는 tk*logyk를 모두 더한 것의 평균을 구하는 방식을 적용 (평균 손실 함수)

-> y가 1차원이 아니라는 것은 데이터 N개에 대해서 판별하겠다는 뜻

-> 원-핫-인코딩이 아니라 정답 라벨이 0과 1뿐이 아닌 여러 개의 경우, 
y([np.arange(batch_size), t]를 통해서 몇 번째 데이터에 몇 번째 인덱스의 확률이 정답인지 2차원 배열로 짝지어주는 것)

-> 정답이 아님을 뜻하는 0은 어차피 더할 때 0이므로 생략하겠다는 뜻

-> 결국 t의 라벨링을 통해 y의 데이터에서 정답 데이터만을 찾고 그 데이터의 로그값을 이용하겠다는 의미
'''
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size