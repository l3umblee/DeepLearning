import numpy as np
dict = {'K1' : np.zeros((2, 3)), 'K2' : np.zeros((3, 4))}

it = np.nditer(dict['K1'], flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    idx = it.multi_index
    print(dict['K1'][idx])

    it.iternext()