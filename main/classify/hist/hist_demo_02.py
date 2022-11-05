from PIL import Image
from numpy import average, dot, linalg
import os

'''
测试直方图相似度算法
来源：https://zhuanlan.zhihu.com/p/93893211

利用直方图计算图片的相似度时，是按照颜色的全局分布情况来看待的，无法对局部的色彩进行分析，
同一张图片如果转化成为灰度图时，在计算其直方图时差距就更大了。对于灰度图可以将图片进行等分，然后在计算图片的相似度。

参考地址：https://zhuanlan.zhihu.com/p/274429582

结论：如果是绝对匹配的话，可以发现，如果是图片出现伸缩，难以发现（51%-58%之间）

不适合有网页发现，如果应用

对于伸缩类图片：CNN识别准确率是100%，
卷积神经网络训练的过程中，选择样本图像内容基本要一样，尺寸可以不一样。
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
    calc_sim = hist_similar(li.histogram(), ri.histogram())
    return calc_sim


