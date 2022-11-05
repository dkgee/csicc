from PIL import Image
from numpy import average, dot, linalg
import os

'''
测试直方图相似度算法
'''


# 将图片转化为RGB
def make_regalur_image(img, size=(64, 64)):
    gray_image = img.resize(size).convert('RGB')
    return gray_image


# 计算直方图
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
    return hist


# 计算相似度
def calc_similar(li, ri):
    st = li.histogram()
    calc_sim = hist_similar(li.histogram(), ri.histogram())
    return calc_sim

