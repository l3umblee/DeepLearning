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

(np.log(y[np.arange(batch_size), t])에 대해서...
-> y가 1차원이 아니라는 것은 데이터 N개에 대해서 판별하겠다는 뜻

-> 원-핫-인코딩이 아니라 정답 라벨이 0과 1뿐이 아닌 여러 개의 경우에도 통하도록, 
y([np.arange(batch_size), t]를 통해서 몇 번째 데이터에 몇 번째 인덱스의 확률이 정답인지 2차원 배열로 짝지어주는 것)
결국 t의 라벨링을 통해 y의 데이터에서 정답 데이터만을 찾고 그 데이터의 로그값을 이용하겠다는 의미
'''
def cross_entropy_error(y, t):
    #y의 차원이 1차원인 경우, (데이터 1개에 대한 것)
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    #원-핫 벡터일 경우, 정답 레이블의 인덱스만 가져와도 됨. -> 원-핫 벡터가 아니라면 각각의 값이 모두 중요할 것임
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    #원-핫 벡터일 경우, 해당하는 인덱스의 값만 살리면 됨 / 아닐 경우에는 t가 인덱스가 아닌 그대로 numpy 배열일 것이므로 이 경우에도 성립
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

#수치 미분 -> 아주 작은 차분으로 미분
'''
1) h -> 반올림 오차(rounding error) 해소
2) 미분으로 구하는 것과 엄밀히 일치하지는 않지만, x를 중심으로 그 전후의 차분을 계산 -> 중앙차분
'''
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

#gradient(기울기) -> 모든 변수의 편미분을 벡터로 정리한 것
'''
f에 대한 편미분은 편미분하고 싶은 변수는 살리고, 다른 변수는 원래의 값을 대입해주면 됨
-> '엄밀한' 편미분을 생각했을 때 해당하는 변수의 항을 제외하고 나머지는 상수 취급하므로
원래의 값을 대입해주는 것으로 생각하면, 상수 부분은 f(x+h) - f(x-h)의 과정에서 없어짐

결과적으로는 해당하는 변수의 항에 대해서만 중심차분을 계산하는 셈
'''
def numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def numerical_gradient_2d(f, x):
    if f.ndim == 1:
        return numerical_gradient_1d(f, x)
    else:    
        h = 1e-4
        grad = np.zeros_like(x)

        for idx, xn in enumerate(x):
            grad[idx] = numerical_gradient_1d(f, xn)
    
        return grad
    
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(x[idx]) + h
        fxh1 = f(x)

        x[idx] = float(x[idx]) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()
    
    return grad

#경사 하강법
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x