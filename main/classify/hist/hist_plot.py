
# from skimage import data
# import matplotlib.pyplot as plt


# img = data.camera()
# arr = img.flatten()
#
# plt.figure("hist")
# n, bins, patches = plt.hist(arr, bins=256, normed=1, edgecolor='None', facecolor='red')
# plt.show()

'''
将图片转换为直方图展示
https://www.cnblogs.com/april0315/p/13721229.html
'''


import cv2
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import os
import numpy as np

plt.rcParams["font.sans-serif"] = ['SimHei']  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

root_dir = os.path.dirname(__file__)
hist_save_path = os.path.join(root_dir, '../../../data/dataset/hist/')


def random_colormap(N: int, cmaps_='gist_ncar', show=False):
    # 从颜色图（梯度多）中取N个
    # test_cmaps = ['gist_rainbow', 'nipy_spectral', 'gist_ncar']
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, N)))
    if show:
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        fig, ax = plt.subplots(1, 1, figsize=(5, 1))
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        plt.show()
    return cmap


def image_hist_demo(filepath):
    # for filename in os.listdir(path):
    #     img = cv2.imdecode(np.fromfile(path + "/" + filename, dtype=np.uint8), -1)
    #
    #     for i in range(1, 665):  # 设置读取数量
    #         plt.hist(img[i].ravel(), 256)
    #         plt.show()  # 显示直方图
    #         cv2.imshow("original_image", img)  # 显示原图
    #
    #         cv2.waitKey()
    #         cv2.destroyAllWindows()

    # 绘制单个图的直方图
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    for i in range(1, 665):  # 设置读取数量
        plt.hist(img[i].ravel(), 256)
        plt.show()  # 显示直方图
        cv2.imshow("original_image", img)  # 显示原图

        cv2.waitKey()
        cv2.destroyAllWindows()


def image_hist(image):
    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color) #传入直方图数据，设置显示颜色
        plt.xlim([0, 256])  #设定图标的上下限，默认是全选，可不用设置
    # plt.show()
    plt.savefig(hist_save_path, format="png")


def plot_demo(image):
    # print(image.ravel())
    plt.hist(image.ravel(), 256, [0, 256])  # ravel将图像3维转一维数组，便于统计频率
    # 统计为256个bin,显示0-255bin,意思是全部显示，我们可以设置只显示一部分
    # plt.show()
    plt.savefig(hist_save_path, format="png")


def plot_demo02(image):
    color = random_colormap(256, cmaps_='gist_ncar', show=False)
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)  # 传入直方图数据，设置显示颜色
    plt.xlim([0, 256])  # 设定图标X轴的上下限，默认是全选，可不用设置
    plt.show()
    # plt.savefig(hist_save_path, format="png")


def plot_demo03(image_path, hist_image_name):
    """
    能够绘制完成hist,色彩默认是蓝色
    https://blog.csdn.net/weixin_48615832/article/details/108188652
    :param image_path:
    :return:
    """
    # image = Image.open(image_path).convert('L')  # 此处转换为灰阶图像了
    image = Image.open(image_path)
    image_arr = np.array(image)

    # 每个像素点由 (R, G，B)三种颜色构成，对其进行分组，将其按每种颜色单独分一组，
    # 例如 (15, 219, 254), (145, 189, 210), (215, 156, 210)三个像素点，分组后为
    # R颜色组为(15, 145, 215), G颜色组为(219， 189， 156)，B颜色组为(254, 210, 210), 将其绘制于统计图上。
    arr = image_arr.flatten()
    plt.figure("hist")
    # density 将频数(整数)转换成频率（小数）
    n, bins, patches = plt.hist(arr, bins=256, density=True, orientation='horizontal', stacked=True)
    plt.xlabel('Frequency')  # 频率
    plt.ylabel('Color Distribution')  # 颜色分布
    plt.grid(axis="both", ls="dotted", lw=1, c="gray", alpha=0.6)  # 绘制x、y轴网格
    plt.savefig(hist_save_path + hist_image_name, format="png")


def plot_demo04(image_path, hist_image_name):
    """
    能够绘制完成hist,色彩是蓝色的
    plt.hist() 函数参数说明参考地址：https://blog.csdn.net/weixin_45520028/article/details/113924866
    :param image_path:
    :return:
    """
    image = Image.open(image_path)
    # 网页截图的颜色模式为RGBA，其中的A为透明度(0-1之间)，就是在RGB的基础上加了一个透明度通道Alpha。
    print("im1的色彩模式为{}".format(image.mode))
    img_mode = image.mode
    if img_mode == 'RGBA':
        r, g, b, a = image.split()
    else:
        r, g, b = image.split()
    # r, g, b, a = image.split()  # 将图像分成单独的波段
    plt.figure("hist")
    red_ar = np.array(r).flatten()
    plt.hist(red_ar, bins=256, density=True, color="red", histtype='barstacked', # orientation='horizontal',
             stacked=True)
    green_ar = np.array(g).flatten()
    plt.hist(green_ar, bins=256, density=True, color="green", histtype='barstacked',# orientation='horizontal',
             stacked=True)
    blue_ar = np.array(b).flatten()
    plt.hist(blue_ar, bins=256, density=True, color="blue", histtype='barstacked',# orientation='horizontal',
             stacked=True)
    # plt.xlabel('Frequency')
    # plt.ylabel('Color Distribution')
    plt.xlabel('色彩分布')
    plt.ylabel('密度频率')
    plt.savefig(hist_save_path + hist_image_name, format="png")


def plot_demo05(image1, image2, image3):
    img1 = Image.open(image1)
    img_mode = img1.mode
    if img_mode == 'RGBA':
        r, g, b, a = img1.split()
    else:
        r, g, b = img1.split()

    plt.figure("hist")
    plt.subplot(3, 1, 1)
    red_ar = np.array(r).flatten()
    plt.hist(red_ar, bins=256, density=True, color="red", histtype='barstacked',  # orientation='horizontal',
             label="Red",
             stacked=True)
    green_ar = np.array(g).flatten()
    plt.hist(green_ar, bins=256, density=True, color="green", histtype='barstacked',  # orientation='horizontal',
             label="Green",
             stacked=True)
    blue_ar = np.array(b).flatten()
    plt.hist(blue_ar, bins=256, density=True, color="blue", histtype='barstacked',  # orientation='horizontal',
             label="Blue",
             stacked=True)
    plt.ylabel('图片一')
    plt.legend(fontsize=10)

    plt.subplot(3, 1, 2)
    img2 = Image.open(image2)
    img_mode = img2.mode
    if img_mode == 'RGBA':
        r, g, b, a = img2.split()
    else:
        r, g, b = img2.split()
    red_ar = np.array(r).flatten()
    plt.hist(red_ar, bins=256, density=True, color="red", histtype='barstacked',  # orientation='horizontal',
             label="Red",
             stacked=True)
    green_ar = np.array(g).flatten()
    plt.hist(green_ar, bins=256, density=True, color="green", histtype='barstacked',  # orientation='horizontal',
             label="Green",
             stacked=True)
    blue_ar = np.array(b).flatten()
    plt.hist(blue_ar, bins=256, density=True, color="blue", histtype='barstacked',  # orientation='horizontal',
             label="Blue",
             stacked=True)
    plt.ylabel('图片二')
    plt.legend(fontsize=10)

    plt.subplot(3, 1, 3)
    img3 = Image.open(image3)
    img_mode = img3.mode
    if img_mode == 'RGBA':
        r, g, b, a = img3.split()
    else:
        r, g, b = img3.split()
    red_ar = np.array(r).flatten()
    plt.hist(red_ar, bins=256, density=True, color="red", histtype='barstacked', # orientation='horizontal',
             label="Red",
             stacked=True)
    green_ar = np.array(g).flatten()
    plt.hist(green_ar, bins=256, density=True, color="green", histtype='barstacked', # orientation='horizontal',
             label="Green",
             stacked=True)
    blue_ar = np.array(b).flatten()
    plt.hist(blue_ar, bins=256, density=True, color="blue", histtype='barstacked', # orientation='horizontal',
             label="Blue",
             stacked=True )
    plt.ylabel('图片三')
    plt.legend(fontsize=10)

    plt.savefig(hist_save_path + "hist-stat-03.png", format="png")



