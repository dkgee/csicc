
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers


def le_net(resize, classes):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5),
                     strides=(1, 1),
                     input_shape=(resize, resize, 3),
                     padding='valid', activation='tanh',
                     kernel_initializer='uniform',
                     name='C1'))     # [None,220,220,6]
    model.add(MaxPooling2D(pool_size=(2, 2), name='S2'))    # [None,110,110,6]
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     strides=(1, 1),
                     padding='valid',
                     activation='tanh',
                     kernel_initializer='uniform',
                     name='C3'))    # [None,106,106,16]
    model.add(MaxPooling2D(pool_size=(2, 2), name='S4'))    # [None,53,53,16]
    model.add(Flatten())    # [None,44944]
    model.add(Dense(120, activation='tanh', name='F5'))     # [None,120]
    model.add(Dense(84, activation='tanh', name='F6'))      # [None,84]
    model.add(Dense(classes, activation='softmax', name='Pre'))     # [None,2]
    model.summary()
    return model


# LeNet卷积神经网络（需要技术分享）：
#   原始图像输入为 224×224×3 通道
#   特征提取部分：4个卷积层（2%丢弃、最大池化）；分类识别部分：1个拼接层，3个全连接层
#   共使用4个卷积层，1个拼接层，3个全连接层，对图像进行处理。
#
def LeNet(resize, classes):
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
    model.summary()
    return model


def alex_net(resize, classes):
    model = Sequential()
    model.add(Conv2D(filters=96,
                     kernel_size=(11, 11),
                     strides=(4, 4),
                     input_shape=(resize, resize, 3),
                     padding='same', activation='relu',
                     kernel_initializer='uniform',
                     name='C1'))    # [None, 56, 56, 96]
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='M2'))    # [None, 27, 27, 96]
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
                     padding='same', activation='relu', kernel_initializer='uniform',
                     name='C3'))    # [None, 27, 27, 256]
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='M4'))    # [None, 13, 13, 256]
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu', kernel_initializer='uniform',
                     name='C5'))    # [None, 13, 13, 384]
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu', kernel_initializer='uniform',
                     name='C6'))    # [None, 13, 13, 384]
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu', kernel_initializer='uniform',
                     name='C7'))    # # [None, 13, 13, 256]
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='M8'))    # [None, 6, 6, 256]
    model.add(Flatten(name='F9'))   # [None, 9216]
    model.add(Dense(4096, activation='relu', name='F10'))   # [None, 4096]
    model.add(Dropout(0.5))     # [None, 4096]
    model.add(Dense(4096, activation='relu', name='F11'))   # [None, 4096]
    model.add(Dropout(0.5))     # [None, 4096]
    model.add(Dense(classes, activation='softmax', name='Pre'))     # [None, 2]
    model.summary()
    return model


def sampel_model(resize, classes):
    model = Sequential()
    model.add(Conv2D(filters=5, kernel_size=(4, 4), strides=(1, 1), input_shape=(resize, resize, 3), padding='same',
                     data_format='channels_last', activation='relu',
                     kernel_initializer='uniform'))     # [None, 224, 224, 5]
    model.add(MaxPooling2D(2, 2))   # [None, 112, 112, 5]
    model.add(Conv2D(filters=8, kernel_size=(4, 4), padding="same", activation="relu"))     # [None, 112, 112, 8]
    model.add(MaxPooling2D(2, 2))   # [None, 56, 56, 8]
    model.add(Flatten())    # [None, 25088]
    model.add(Dense(64, activation="relu"))     # [None, 64]
    model.add(Dropout(0.5))     # [None, 64]
    model.add(Dense(classes, activation="softmax"))     # [None, 2]
    model.summary()
    return model
