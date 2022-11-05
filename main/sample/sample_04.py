import matplotlib.pyplot as plt
import numpy as np
import os

'''
【数据可视化】
展示Python绘制激活函数图像
参考地址：https://blog.csdn.net/Hankerchen/article/details/123436597
'''
root_dir = os.path.dirname(__file__)
sigmoid_pic_save_path = os.path.join(root_dir, '../../data/model/network_img/sigmoid-03.png')
relu_pic_save_path = os.path.join(root_dir, '../../data/model/network_img/relu-03.png')
prelu_pic_save_path = os.path.join(root_dir, '../../data/model/network_img/prelu-01.png')
tan_pic_save_path = os.path.join(root_dir, '../../data/model/network_img/tan-01.png')


# 定义函数
def style_01(x):
    # 返回1，绘制辅助虚线
    return x * 0 + 1


def sigmoid(x):
    # 直接返回sigmoid函数
    return 1. / (1. + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def relu_2(x):
    return np.where(x < 0, 0, x)


def prelu(x):
    return np.where(x < 0, x * 0.5, x)


def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)
    plt.plot(x, y)
    # plt.show()
    plt.savefig(sigmoid_pic_save_path, format="png")


def plot_relu():
    # param:起点，终点，间距
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    # plt.show()
    plt.savefig(relu_pic_save_path, format="png")


def plot_sigmoid_02():
    x = np.arange(-8, 8, 0.1)
    y = sigmoid(x)
    # 绘制虚线
    y_2 = style_01(x)
    fig = plt.figure()  # 如果使用plt.figure(1)表示定位（创建）第一个画板，如果没有参数默认创建一个新的画板，如果plt.figure(figsize = (2,2)) ，表示figure 的大小为宽、长
    ax = fig.add_subplot(111)  # 表示前面两个1表示1*1大小，最后面一个1表示第1个
    ax.spines['top'].set_color('none')  # ax.spines设置坐标轴位置，set_color设置坐标轴边的颜色
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.plot(x, y, color="b", lw=1)  # 设置曲线颜色，线宽
    ax.plot(x, y_2, color="k", lw=1, linestyle='--')  # 设置曲线颜色，线宽
    plt.xticks(fontsize=14)  # 设置坐标轴的刻度子字体大小
    plt.yticks(fontsize=14)
    plt.xlim([-8.05, 8.05])  # 设置坐标轴范围
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()  # 自动调整子图参数
    # plt.grid(axis="y", ls="-", lw=1, c="c", alpha=0.8) # 只绘制y轴
    # 线轴风格 '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    plt.grid(axis="both", ls="dotted", lw=1, c="gray", alpha=0.6)    # 绘制x、y轴
    # plt.show()  # 显示绘图
    plt.savefig(sigmoid_pic_save_path, format="png")


def plot_relu_02():
    x = np.arange(-8, 8, 0.1)
    y = relu_2(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.plot(x, y, color="b", lw=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([-8.05, 8.05])
    plt.ylim([-0.02, 1.02])
    ax.set_yticks([2, 4, 6, 8])
    plt.tight_layout()
    plt.grid(axis="both", ls="dotted", lw=1, c="gray", alpha=0.6)
    # plt.show()
    plt.savefig(relu_pic_save_path, format="png")


def plot_tanh():
    x = np.arange(-8, 8, 0.1)
    y = tanh(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color="b", lw=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([-8.05, 8.05])
    plt.ylim([-0.02, 1.02])
    ax.set_yticks([-1.1, -0.5, 0.5, 1.1])
    ax.set_xticks([-8, -5, 5, 8])
    plt.tight_layout()
    # plt.show()
    plt.savefig(tan_pic_save_path, format="png")


def plot_prelu():
    x = np.arange(-8, 8, 0.1)
    y = prelu(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color="b", lw=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig(prelu_pic_save_path, format="png")


if __name__ == '__main__':
    # plot_sigmoid()
    # plot_relu()
    # plot_sigmoid_02()
    plot_relu_02()
    # plot_tanh()
    # plot_prelu()
