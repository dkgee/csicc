
import cv2
import numpy as np


# hash distance
# 均值哈希算法
def aHash(img, resize):
    # 缩放为8*8
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(resize):
        for j in range(resize):
            s = s + gray[i, j]
    print("=================")
    print(gray)
    print("=================")
    # 求平均灰度
    avg = s / (resize*resize)
    print("灰度图像均值：", avg)
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(resize):
        for j in range(resize):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    print("哈希值", hash_str)
    return hash_str


# 图像灰度处理
def pic_gray(img, resize, path):
    # 缩放
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 转化为RGB格式
    # BGR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 二值化
    # ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(path, gray)  # 保存

    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(resize):
        for j in range(resize):
            s = s + gray[i, j]
    print("=================")
    print(gray)
    print("=================")
    # 求平均灰度
    avg = s / (resize * resize)
    print("灰度图像均值：", avg)
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(resize):
        for j in range(resize):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    print("哈希值", hash_str)



# # 均值哈希算法
# def aHash(img, resize):
#     # 缩放为8*8
#     img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
#     # # 转换为灰度图
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # s为像素和初值为0，hash_str为hash值初值为''
#     s = 0
#     hash_str = ''
#     # 遍历累加求像素和
#     for i in range(resize):
#         for j in range(resize):
#             s = s + img[i, j]
#     # 求平均颜色点
#     avg = s / (resize*resize)
#
#     # 计算各个颜色点与平均颜色点的欧式距离（距离大小阀值为多少）
#
#
#     # 灰度大于平均值为1相反为0生成图片的hash值
#     for i in range(resize):
#         for j in range(resize):
#             if img[i, j] > avg:
#                 hash_str = hash_str + '1'
#             else:
#                 hash_str = hash_str + '0'
#     return hash_str


# 差值感知算法
def dHash(img, resize):
    # 缩放8*8  （此处根据实际情况进行调整）
    img = cv2.resize(img, (resize+1, resize), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(resize):
        for j in range(resize):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str


def flatten(x):
    result = []
    for el in x:
        result.extend(el)
    return result


# 感知哈希
def pHash(img, resize):
    """get image pHash value"""
    # 加载并调整图片为8*8灰度图片
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img       # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(resize, resize)

    # 把二维list变成一维list
    img_list = flatten(vis1.tolist())

    # 计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]
    hash_str = ''
    for i in avg_list:
        hash_str += str(i)
    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]), 2) for x in range(0, resize*resize)])


# 汉明距离
def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    s = sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])
    return s


# Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n+1
    return n


# 感知哈希
def p_middle_Hash(imgfile):
    """get image pHash value"""
    # 加载并调整图片为32x32灰度图片
    # img = cv2.imread(imgfile, cv2.INTER_CUBIC)
    # 加载并调整图片为8*8灰度图片

    img = cv2.resize(imgfile, (32, 32), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img  # 填充数据

    # 二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    # 通过人为设定侧重区域来实现相似度算法注意点
    vis1 = vis1[0:8, 12:20]  # 取矩阵中间8*8数据
    # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    # vis1.resize(8, 8)

    # 把二维list变成一维list
    img_list = flatten(vis1.tolist())

    # 计算均值
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i < avg else '1' for i in img_list]

    # 得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 64)])


# 计算哈希相似度
def hash_similarity(img1, img2, hash_type, resize):
    resize2 = resize * resize
    if hash_type == "aHash":
        # 均值哈希
        hash1 = aHash(img1, resize)
        hash2 = aHash(img2, resize)
        n = cmpHash(hash1, hash2)
        percent = (resize2 - n) / resize2
    elif hash_type == "dHash":
        # 差值哈希
        hash1 = dHash(img1, resize)
        hash2 = dHash(img2, resize)
        m = cmpHash(hash1, hash2)
        percent = (resize2 - m) / resize2
    elif hash_type == "pHash":
        # 感知哈希
        hash1 = pHash(img1, resize)
        hash2 = pHash(img2, resize)
        p = hammingDist(hash1, hash2)
        percent = (resize2-p)/resize2
    elif hash_type == "pmHash":
        # 感知哈希
        hash1 = p_middle_Hash(img1)
        hash2 = p_middle_Hash(img2)
        p = hammingDist(hash1, hash2)
        percent = (64 - p) / 64
    else:  # 差值和平均的平均
        # 均值哈希
        hash1 = aHash(img1, resize)
        hash2 = aHash(img2, resize)
        n = cmpHash(hash1, hash2)
        precent_a = (resize2 - n) / resize2
        # 差值哈希
        hash1 = dHash(img1, resize)
        hash2 = dHash(img2, resize)
        m = cmpHash(hash1, hash2)
        precent_d = (resize2 - m) / resize2
        percent = (precent_d+precent_a)/2

    return percent