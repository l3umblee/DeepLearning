import numpy as np

x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
x1 = np.array([[[11, 12, 13], [14, 15, 16], [17, 18, 19]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])
x_final = np.array([x, x1])
print(x_final)
print(x_final.shape)
print(x_final[0][:1, :2, :2]) #4차원 배열 형태를 다루는 법
x_final = x_final.flatten()