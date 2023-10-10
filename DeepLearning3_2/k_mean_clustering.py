import numpy as np
import cv2
import tqdm

#generate_color : 동적으로 k개의 color list 생성
def generate_color(k):
    color_list = list()
    idx = 0
    while idx < k:
        temp = np.random.randint(0, 255, size=3)
        temp = list(temp)
        if (temp in color_list) == False:
            color_list.append(temp)
            idx += 1
    return color_list

#k_mean_clustering : k는 임의로 정해줘도 되고, 다른 방법을 사용해서 구한 다음 해도 됨.
def k_mean_clustering(img, k, iter_num):
    img_col = np.reshape(img, (1, img.shape[0]*img.shape[1], 3))
    
    #group_list는 0, 1, 2... group에 대한 위치값을 가지고 있음. (img를 펴서, 위치값을 저장)
    group_list = list()
    for i in range(k):
        group_list.append(list())  
    
    numbers = np.random.choice(range(0, img_col.shape[1]), k, replace=False) #초기 centroid는 난수! (중복없이)
    centroid_list = img_col[:, numbers, :]

    #초기 centroid에 대해서 clustering 진행
    #먼저 임의로 선정된 centroid와 픽셀값들의 L2 norm을 구하고, centorid와 차이가 가장 작은 인덱스, group에 해당 픽셀 정보를 저장
    for idx in range(img_col.shape[1]):
        temp_np = (centroid_list - img_col[:,idx,:].astype(int))**2
        temp_np = np.sum(temp_np, axis=2)

        min_dist_idx = np.argmin(temp_np)
        group_list[min_dist_idx].append(idx)
        
    #centroid를 갱신해줄 것임 -> k-mean으로
    for idx in range(k):
        temp_np = img_col[:,group_list[idx],:]
        temp_np = np.mean(temp_np, axis=1)
        centroid_list[:,idx,:] = np.asarray(temp_np, dtype=int)

    for i in tqdm.tqdm(range(iter_num)):
        for idx in range(k):
            group_list[idx].clear()

        for idx in range(img.shape[0]*img.shape[1]):
            temp_np = (centroid_list - img_col[:,idx,:].astype(int))**2
            temp_np = np.sum(temp_np, axis=2)

            min_dist_idx = np.argmin(temp_np)
            group_list[min_dist_idx].append(idx)

        for idx in range(k):
            temp_np = img_col[:,group_list[idx],:]
            temp_np = np.mean(temp_np, axis=1)
            centroid_list[:,idx,:] = np.asarray(temp_np, dtype=int)

    new_img = np.zeros_like(img_col)

    #color_list 지정
    color_list = generate_color(k)

    for idx in range(k):
        new_img[:,group_list[idx],:] = np.array(color_list[idx])
    
    new_img = np.reshape(new_img, (img.shape[0], img.shape[1], 3))
    return new_img

k = 5
iter_num = 10

input_img = cv2.imread('test2.jpg')
output_img = k_mean_clustering(input_img, k, iter_num)

cv2.imshow('Result', output_img)
cv2.waitKey(0)