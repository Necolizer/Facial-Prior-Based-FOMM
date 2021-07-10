import os
from glob import glob
import re
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import argparse
import numpy as np
import torch
import itertools
from skimage import io

'''
    根据路径读取数据
    文件的结构与老师给定的一致
    只需要将 os.getcwd() 改成相应的前缀即可
    :)
'''
'''
    配置相应的参数
'''
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=1000,
                    help='epoch number')

parser.add_argument('--num_classes', type=int, default=11,
                    help='num_classes')

parser.add_argument('--predict_size', type=tuple, default=(32, 32),
                    help='predict_size')

parser.add_argument('--stride', type=int, default=16,
                    help='stride')

parser.add_argument('--sigma', type=float, default=4,
                    help='sigma')

parser.add_argument('--threshold', type=float, default=0.4,
                    help='threshold')

config = parser.parse_args(args=[])

'''
    根据高斯核函数定义的函数
    检查是否符合其定义
    :)
'''


def _gaussian(dis, sigma):
    return np.exp(-np.sqrt(dis) / (2 * sigma ** 2))


'''
    核心代码输出推理标签
'''


def c1_encoder(config, gt_coords, gt_classes, tensor=True):
    if not isinstance(gt_coords, np.ndarray):
        gt_coords = gt_coords.detach().cpu().numpy
    heatmap = np.zeros((config.num_classes, *config.predict_size))
    offymap = np.zeros((config.num_classes, *config.predict_size))
    offxmap = np.zeros((config.num_classes, *config.predict_size))
    maskmap = np.zeros((config.num_classes, *config.predict_size))
    clssmap = np.zeros((config.num_classes + 6, *config.predict_size))

    gridmap = np.array(
        list(itertools.product(range(1, config.predict_size[0] + 1), range(1, config.predict_size[1] + 1))))
    gridmap = (gridmap - 0.5) * config.stride
    for i, (gt_coord, gt_class) in enumerate(zip(gt_coords, gt_classes[: gt_coords.shape[0]])):
        distance = np.square((gt_coord[::-1] - gridmap)).sum(axis=-1)
        heatmap[i] = (_gaussian(distance, config.sigma).reshape(*config.predict_size))
        heatmap[i] = heatmap[i] / heatmap[i].max()
        offset = ((gt_coord[::-1] - gridmap) / config.stride).reshape(*config.predict_size, -1)
        offymap[i] = offset[:, :, 0]
        offxmap[i] = offset[:, :, 1]
        clssmap[i] = gt_class
        maskmap[i] = (heatmap[i] >= 0.2).astype(np.int32)

    for i, gt_class in enumerate(gt_classes[gt_coords.shape[0]:]):
        clssmap[gt_coords.shape[0] + i] = int(gt_class > 0)
    if tensor:
        heatmap = torch.from_numpy(heatmap)
        offymap = torch.from_numpy(offymap)
        offxmap = torch.from_numpy(offxmap)
        clssmap = torch.from_numpy(clssmap)

    return heatmap, offymap, offxmap, clssmap, maskmap





def load_label(txt_files):
    img_files = []
    gt_coordes = []
    gt_classes = []
    is_zhuitis = []
    '''
        从 txt 文件中导入类别
        该函数返回四个列表，分别对应存储 图片路径 路径图片对应的坐标 路径图片坐标点对应的类别 以及对应位是否为椎体(vertebra)
        返回的列表方便核心代码 c1_encoder 使用
        :)
    '''

    disc_dict = {'': 0, 'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3, 'v5': 16, 'v1,v5': 11, 'v2,v5': 12, 'v3,v5': 13,
                 'v4,v5': 14,
                 'v5,v1': 11, 'v5,v2': 12, 'v5,v3': 13, 'v5,v4': 14}
    vertebra_dict = {'': 0, 'v1': 0, 'v2': 1}
    identification_dict = {'L1': 0, 'L2': 1, 'L3': 2, 'L4': 3, 'L5': 4,
                           'T12-L1': 5, 'L1-L2': 6, 'L2-L3': 7, 'L3-L4': 8, 'L4-L5': 9, 'L5-S1': 10,
                           'T11-T12': 5}
    '''
        此处针对txt中的一些小bug做了一些适应性调整
            a) txt 中的标签存在既没有 disc 又没有 vertebra 标签的样本，这种情况将其判定为正常 v1 情况
            b) txt 中出现了一个 T11-T12 的样本，经过比对，发现是第一个椎间盘 T12-L1
    '''

    '''
        读取文件夹下所有 txt 文件并读取其中的类别标签
        ### 建议这一部分大家可以把里层 for循环 抠出来，既可以看看输出了什么，也可以一起查一查哪里出错了
    '''
    for txt in txt_files:
        img_files.append(txt[:-4] + '.jpg')
        gt_coords = [np.zeros([1, 2])] * 11
        gt_clss = [0] * 17
        is_zhuiti = [0] * 11
        for line in open(txt, 'r'):
            # print(line)
            info = re.findall('(.*),{(.*)', line)[0]
            # print(info)
            origin_size = io.imread(txt[:-4] + '.jpg').shape
            coord_ = np.array(list(eval(info[0])))
            coord = ((coord_ / origin_size) * 512).astype(np.int32)
            # 将原本在 (346, 384) 大小的图片下表示的坐标通过线性变换转换成在 (512, 512) 大小的图片的位置
            clss = eval('{' + info[1])
            # print(gt_coords)
            gt_coords[identification_dict[clss['identification']]] = coord
            # print(gt_clss)
            if identification_dict[clss['identification']] <= 4:
                # print(identification_dict[clss['identification']])
                gt_clss[identification_dict[clss['identification']]] = vertebra_dict[clss['vertebra']]
                is_zhuiti[identification_dict[clss['identification']]] = 1



            else:
                t = disc_dict[clss['disc']]
                if t > 5:
                    if t == 11:
                        gt_clss[identification_dict[clss['identification']]] = disc_dict['v1']
                        gt_clss[identification_dict[clss['identification']] + 6] = 1

                    if t == 12:
                        gt_clss[identification_dict[clss['identification']]] = disc_dict['v2']
                        gt_clss[identification_dict[clss['identification']] + 6] = 1

                    if t == 13:
                        gt_clss[identification_dict[clss['identification']]] = disc_dict['v3']
                        gt_clss[identification_dict[clss['identification']] + 6] = 1

                    if t == 14:
                        gt_clss[identification_dict[clss['identification']]] = disc_dict['v4']
                        gt_clss[identification_dict[clss['identification']] + 6] = 1
                    if t == 16:
                        gt_clss[identification_dict[clss['identification']]] = disc_dict['v1']
                        gt_clss[identification_dict[clss['identification']] + 6] = 1

                else:
                    gt_clss[identification_dict[clss['identification']]] = (disc_dict[clss['disc']])

            # print('\n\n')
        gt_coordes.append(np.array(gt_coords))
        gt_classes.append(gt_clss)
        is_zhuitis.append(is_zhuiti)

    return img_files, gt_coordes, gt_classes, is_zhuitis


def load_class(txt_labels):
    _heatmap = []
    _offymap = []
    _offxmap = []
    _clssmap = []
    _maskmap = []

    img_files, gt_coordes, gt_classes, is_zhuitis = txt_labels
    for i, (img_file, gt_coords, gt_clsses, is_zhuiti) in enumerate(zip(img_files, gt_coordes, gt_classes, is_zhuitis)):
        try_heatmap, try_offymap, try_offxmap, try_clssmap, try_maskmap = c1_encoder(config, gt_coords, gt_clsses)
        try_offymap = try_offymap * try_maskmap
        try_offxmap = try_offxmap * try_maskmap
        _heatmap.append(try_heatmap)
        _offymap.append(try_offymap)
        _offxmap.append(try_offxmap)
        # t = np.zeros([17,32,32])
        # k = try_clssmap.numpy()
        # t[0:5] = k[0:5]
        # t[0:5] *= try_maskmap[0:5]
        # t[int(k[5][0][0]+5)] = 1
        # t[int(k[5][0][0]+5)] *= try_maskmap[5]
        # t[int(k[6][0][0]+9)] = 1
        # t[int(k[6][0][0]+9)] *= try_maskmap[6]
        # t[int(k[7][0][0]+13)] = 1
        # t[int(k[7][0][0]+13)] *= try_maskmap[7]
        # t[int(k[8][0][0]+17)] = 1
        # t[int(k[8][0][0]+17)] *= try_maskmap[8]
        # t[int(k[9][0][0]+21)] = 1
        # t[int(k[9][0][0]+21)] *= try_maskmap[9]
        # t[int(k[10][0][0]+25)] = 1
        # t[int(k[10][0][0]+25)] *= try_maskmap[10]
        # t[-6:]=k[-6:]
        try_clssmap[:11] *= try_maskmap
        try_clssmap[11:] *= try_maskmap[5:]
        _clssmap.append(try_clssmap)
        _maskmap.append(try_maskmap)
        # print(img_file, i)
        # print(_clssmap)
        # print(_clssmap[0].shape)

    return img_files, _heatmap, _offymap, _offxmap, _clssmap, _maskmap, is_zhuitis


# %%
# 这里保存了所有map，还有把classmap从17深度转化为35深度的classmap，但还没有save下来
if __name__ == '__main__':
    train_pth = os.path.join(os.getcwd(), 'train/data')
    test_pth = os.path.join(os.getcwd(), 'test/data')
    train_txts = glob(os.path.join(train_pth, '*.txt'))
    test_txts = glob(os.path.join(test_pth, '*.txt'))
    train_txt_labels = load_label(train_txts)
    a = load_class(train_txt_labels)
    test_txt_labels = load_label(test_txts)
    b = load_class(test_txt_labels)

# # %%
# import torch
# from loss import Lossfuction4 as L4
# cm_true = torch.ones((1, 35, 32, 32))
# cm_pred = torch.ones((1, 35, 32, 32))
# loss_test = L4(cm_true, cm_pred, is_zhuiti=True)

# %%
