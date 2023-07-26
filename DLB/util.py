import numpy as np

def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    
    kaiser window를 이용
    -> 입력값 x와 kaiser window를 covolution 시킨 것 (kaiser window의 자세한 형태는 위키피디아 참조)
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다.

    Parameters
    ----------
    x : 훈련 데이터
    t : 정답 레이블
    
    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

#im2col : image to column이라는 뜻으로, 입력 데이터를 필터링하기 좋게 펼치는 함수, 쉽게 말해 일렬로 나열하는 함수
'''
사실 im2col을 사용해 합성곱 계층을 구현하면 메모리를 더 많이 소비하는 단점이 있지만
컴퓨터는 큰 행렬을 묶어서 계산하는데 탁월하기 때문에 큰 행렬의 곱셈을 빠르게 계산할 수 있다.

-> 행렬 계산은 선형 대수 라이브러리를 활용해 효율을 높일 수 있다.

im2col의 결과는 2차원 행렬이다.
------------------------------------------
input_data - (데이터 수, 채널 수, 높이, 너비)의 4차원 배열로 이루어진 입력 데이터 (N, C, H, W)
filter_h - 필터의 높이
filter_w - 필터의 너비
stride - 스트라이드
pad - 패딩 (두께)
'''

#내가 구현한 버전 -> 패딩이 고려가 안됨
def im2col_JY(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape

    #이 두 변수는 출력 크기의 높이와 너비를 나타내기도 하지만, 필터에 의해 입력 데이터를 출력하는 동작을 몇번할지도 결정할 수 있다.
    oh = (H + 2*pad - filter_h) // stride + 1 #출력 크기의 높이 (정수 취급을 위해 // 연산자 사용)
    ow = (W + 2*pad - filter_w) // stride + 1 #출력 크기의 너비
    #oh와 ow는 최종적으로 output 데이터의 가로와 세로의 크기를 나타내기도 하지만, 그렇다는 것은 필터가 '도장'을 가로로 몇번, 세로로 몇번 찍느냐도 나타냄

    out = np.zeros((N*(oh*ow), filter_h*filter_w*C))
    cnt = 0
    for hidx in range(oh):
        ymax = filter_h + hidx #ymax : image 한장의 y축 (세로) 좌표를 표시
        for widx in range(ow):
            xmax = filter_w + widx #xmax : image 한장의 x축 (가로) 좌표를 표시
            out[cnt:cnt+N] = input_data[:, :, hidx:ymax, widx:xmax].flatten().reshape(N, filter_h*filter_w*C)
            cnt += 1

    return out

#col2im : im2col의 반대 버전
'''
주된 사용은 convolution의 backward (오차역전파법)을 위함

parameters
-----------
col : 2차원 배열 (입력 데이터)
input_shape : 원래 이미지 데이터의 형상
filter_h : 필터의 높이
filter_w : 필터의 너비
stride: 스트라이드
pad : 패딩
'''
def col2im_JY(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    oh = (H + 2*pad - filter_h) // stride + 1
    ow = (W + 2*pad - filter_w) // stride + 1
    col = col.reshape(N, oh, ow, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    #이렇게 형변환을 하는 이유는 input_data (im2col을 통과한 버전)의 가로와 세로를 결정지었던, 모든 요소대로 분해한 것!
    #세로 : N x oh x ow / 가로 : C x filter_h x filter_w