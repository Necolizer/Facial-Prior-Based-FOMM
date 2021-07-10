# Written by tzx
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
import matplotlib.pyplot as plt
def _gaussian(dis, sigma):
    return np.exp(-np.sqrt(dis) / (2 * sigma ** 2))
import time
def get_distance_map(points, size, sigma, kpnums):
    gmap1 = np.array(list(itertools.product(range(1, size[0] + 1), (range(1, size[1] + 1)))))
    gmap1 = np.expand_dims(gmap1, 0)
    gmap1 = np.repeat(gmap1, kpnums, 0)
    np_points = points.detach().numpy()
    gmap1 = gmap1 - np.expand_dims(np_points, 1)
    gmap1 = abs(gmap1)
    gmap1 = gmap1[:,:,0]+gmap1[:,:,1]
    gmap1 = _gaussian(gmap1,sigma)
    gmap1_max = np.max(gmap1, axis=1)
    gmap1_max = np.expand_dims(gmap1_max, 1)
    gmap1 = gmap1/gmap1_max
    mean = np.mean(gmap1, axis=1)
    mean = np.expand_dims(mean,1)
    thresh = (gmap1 >= mean)
    gmap1 = gmap1*thresh
    gmap1 = gmap1.reshape((kpnums,size[0],size[1]))
    out = np.zeros(size)

    for i in range(kpnums):
        out+=gmap1[i]
    out = out/out.max()
    # mean = np.mean(out)
    # mean = np.expand_dims(mean, 1)
    # thresh = (out >= mean)
    # out = out*thresh
    return out

#keypoints 是在原人脸图片上的检测点， img_size是原图片的大小， feature_size是feature_map的大小, sigma是正态分布的标准差
def make_masks(key_points, img_size, feature_size, sigma):
    key_points = key_points*64/img_size[0]
    point_nums = 18
    mask = np.zeros((point_nums, *(feature_size[0], feature_size[1])))
    key_points_new = torch.zeros([point_nums,2])
    key_points_new[:, 0] = key_points[:, 1]
    key_points_new[:, 1] = key_points[:, 0]
    mask = get_distance_map(key_points_new, feature_size, sigma, point_nums)
    out = torch.tensor(mask).unsqueeze(dim=0)

    return out

if __name__ == '__main__':
    a = torch.load("kp.pt")
    print(a.shape)
    b = a[0]
    make_masks(b, (256,256), (64,64), 1.2)
    # print(np.array([(64,64), (30,23)]).shape)
# (456,354)