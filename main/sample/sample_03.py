
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.utils.vis_utils import plot_model

import os
# os.environ["PATH"] += os.pathsep + ' C:/Program Files/Graphviz/bin'


'''
【CNN网络结构打印】
展示CNN网络结构图像
'''
root_dir = os.path.dirname(__file__)
# model_pic_save_path = os.path.join(root_dir, '../../data/model/network_img/flatten-demo.png')
model_pic_save_path = os.path.join(root_dir, '../../data/model/network_img/self-lenet-demo02.png')

# demo01、测试使用
# mode = Sequential()
# mode.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
# mode.add(Flatten())
# # plot_model(mode, to_file='Flatten.png', show_shapes=True)
# plot_model(mode, to_file=model_pic_save_path, show_shapes=True)

# demo02、训练的模型
def LeNet():
    classes = 14
    resize = 224
    model = Sequential()    # 顺序模型指引
    # 卷积层01：5个滤波器，3×3的卷积核，步长为1×1，周围空白填充空白，将输入的图像统一归一化为224×224×3（宽、高，颜色通道），数据通道为channels_last模式，
    # 激活函数为Relu（矫正线性单元），内核初始化为 uniform 格式
    model.add(Conv2D(filters=5, kernel_size=(3, 3), strides=(1, 1), input_shape=(resize, resize, 3), padding='same',
                     data_format='channels_last', activation='relu', kernel_initializer='uniform'))  # [None,224,224,5]
    model.add(Dropout(0.2))     # Dropout正则化，2%数据随机丢弃
    model.add(MaxPooling2D((2, 2)))  # 池化核大小[None,112,112,5]  池化层01：最大值池化，2×2的特征图

    # 卷积层02：16个滤波器，3×3的卷积核，步长为1×1，周围空白填充空白，数据通道为channels_last模式，
    # 激活函数为Relu（矫正线性单元），内核初始化为 uniform 格式
    model.add(Conv2D(16, (3, 3), strides=(1, 1), data_format='channels_last', padding='same', activation='relu',
                     kernel_initializer='uniform'))  # [None,112,112,16]
    model.add(Dropout(0.2))      # Dropout正则化，2%数据随机丢弃
    model.add(MaxPooling2D(2, 2))  # output_shape=[None,56,56,16]  池化层02：最大值池化，2×2的特征图

    # 卷积层03：32个滤波器，3×3的卷积核，步长为1×1，周围空白填充空白，数据通道为channels_last模式，
    # 激活函数为Relu（矫正线性单元），内核初始化为 uniform 格式
    model.add(Conv2D(32, (3, 3), strides=(1, 1), data_format='channels_last', padding='same', activation='relu',
                     kernel_initializer='uniform'))   # [None,56,56,32]

    model.add(Dropout(0.2))      # Dropout正则化，2%数据随机丢弃

    model.add(MaxPooling2D(2, 2))       # 池化层03：最大值池化，2×2的特征图 # [None,28,28,32]

    # 卷积层04：100个滤波器，3×3的卷积核，步长为1×1，数据通道为channels_last模式，
    # 激活函数为Relu（矫正线性单元），内核初始化为 uniform 格式
    model.add(Conv2D(100, (3, 3), strides=(1, 1), data_format='channels_last', activation='relu',
                     kernel_initializer='uniform'))  # [None,26,26,100]

    # 拼接层：拉平，将各个特征图拼接，数据通道为channels_last模式，keras默认数据格式即为channels_last
    # model.add(Flatten(data_format='channels_last'))  # [None,67600]
    model.add(Flatten())

    # 全连接层01：168个神经元，激活函数为Relu（矫正线性单元）
    model.add(Dense(168, activation='relu'))   # [None,168] # [None,168]
    # 全连接层02：84个神经元，激活函数为Relu（矫正线性单元）
    model.add(Dense(84, activation='relu'))    # [None,84]  # [None,84]
    # 全连接层02：2个神经元，输出函数为softmax

    model.add(Dense(classes, activation='softmax'))  # [None,2]
    # 打印概要日志
    # model.summary()
    return model


model = LeNet()
plot_model(model, to_file=model_pic_save_path, show_shapes=True)
