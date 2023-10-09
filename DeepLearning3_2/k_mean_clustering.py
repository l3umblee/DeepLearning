import numpy as np
import cv2
import tqdm

#k_mean_clustering : k는 임의로 정해줘도 되고, 다른 방법을 사용해서 구한 다음 해도 됨.
def k_mean_clustering(img, k):
    img_col = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
    
    #group_list는 0, 1, 2... group에 대한 위치값을 가지고 있음. (img를 1차원으로 펴서, 위치값을 저장)
    group_list = list()
    for i in range(k):
        group_list.append(list())  
    
    numbers = np.random.choice(range(0, img_col.shape[1]), k, replace=False) #초기 centroid는 난수! (중복없이)
    #numbers = list([2842, 13770, 297, 27233, 2163])
    centroid_list = img_col[:, numbers, :]

    #초기 centroid에 대해서 clustering 진행
    #먼저 임의로 선정된 centroid와 픽셀값들의 L2 norm을 구하고, centorid와 차이가 가장 작은 인덱스, group에 해당 픽셀 정보를 저장
    for idx in range(img_col.shape[1]):
        temp_np = (np.tile(img_col[:,idx,:], (1, k, 1)) - centroid_list)**2
        temp_np = np.sum(temp_np, axis=2)

        min_dist_idx = np.argmin(temp_np)
        group_list[min_dist_idx].append(idx)
        
    #centroid를 갱신해줄 것임 -> k-mean으로
    for idx in range(k):
        temp_np = img_col[:,group_list[idx],:]
        temp_np = np.mean(temp_np, axis=1)
        centroid_list[:,idx,:] = np.asarray(temp_np, dtype=int)

    iter_num = 10
    for i in tqdm.tqdm(range(iter_num)):
        for idx in range(k):
            group_list[idx].clear()

        for idx in range(img.shape[0]*img.shape[1]):
            temp_np = (img_col[:,idx,:] - centroid_list)**2
            temp_np = np.sum(temp_np, axis=1)

            min_dist_idx = np.argmin(temp_np)
            group_list[min_dist_idx].append(idx)

        for idx in range(k):
            temp_np = img_col[:,group_list[idx],:]
            temp_np = np.mean(temp_np, axis=1)
            centroid_list[:,idx,:] = np.asarray(temp_np, dtype=int)

    new_img = np.zeros_like(img_col)

    #color_list 지정
    color_list = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0], [139, 0, 255]]

    for idx in range(k):
        new_img[:,group_list[idx],:] = np.array(color_list[idx])
            
    # for idx in range(k):
    #     new_img[:,group_list[idx],:] = centroid_list[:,idx,:]
    
    new_img = np.reshape(new_img, (img.shape[0], img.shape[1], 3))
    return new_img

input_img = cv2.imread('test.jpg')
a = k_mean_clustering(input_img, 5)

cv2.imshow('Result', a)
cv2.waitKey(0)