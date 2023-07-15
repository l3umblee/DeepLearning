#MultiLayerNet : 완전 연결 다층 신경망
'''
구현해야 할 것
Parameters
-------------------------------------------------------
input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    weight_decay_lambda : 가중치 감소(L2 법칙)의 세기'

Method
-------------------------------------------------------
init_weight(weight_init_std) : 가중치 초기화 메서드
predict(x) : 입력값에 대한 예측값 반환
loss(x, t) : 입력값과 정답 레이블에 대한 손실함수의 값 반환
accuracy(x, t) : 정확도 계산
numerical_gradient(x, t) : 수치 미분을 이용한 기울기 반환
gradient(x, t) : 오차역전파법을 이용한 기울기 반환
'''
