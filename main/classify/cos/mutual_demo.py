
import cv2
import numpy as np
import math
import os

from sklearn.metrics import mutual_info_score

'''
使用互信息测试图片相似度
参考地址：https://blog.csdn.net/qq_39923466/article/details/118809611

该方法未跑通，因为sklearn总是报错：“ImportError: DLL load failed: 找不到指定的程序。”
'''

# def binary_mutula_information(label, sample):
#     # 用字典来计数
#     d = dict()
#     # 统计其中00,01,10,11各自的个数
#     binary_mi_score = 0.0
#     label = np.asarray(label)
#     sample = np.asarray(sample)
#     if label.size != sample.size:
#         print('error！input array length is not equal.')
#         exit()
#
#     # np.sum(label)/label.size表示1在label中的概率,
#     # 前者就是0在label中的概率
#     # 这里需要用总的数目减去1的数目再除以总的数目，提高精度
#     x = [(label.size - np.sum(label)) / label.size, np.sum(label) / label.size]
#
#     y = [(sample.size - np.sum(sample)) / sample.size, np.sum(sample) / sample.size]
#
#     for i in range(label.size):
#         if (label[i], sample[i]) in d:
#             d[label[i], sample[i]] += 1
#         else:
#             d[label[i], sample[i]] = 1
#
#     # 遍历字典，得到各自的px,py,pxy，并求和
#     for key in d.keys():
#         px = x[key[0]]
#         py = y[key[1]]
#         pxy = d[key] / label.size
#         binary_mi_score = binary_mi_score + pxy * math.log(pxy / (px * py))
#
#     return binary_mi_score