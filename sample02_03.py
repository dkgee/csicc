from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from main.classify.model import LeNet
from config import Config
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

'''
【数据预处理+模型训练+可视化图表】
模型训练主类，此Demo演示模型训练及其中的数据可视化图表展示过程，

相比sample02、sample02_visual，此类使用同一个图像特征提取器处理训练集和测试集数据。

主要区别参数：
    batch=100
    steps_per_epoch=26

此文件外界输入：
    config.train_path（训练数据集目录）
    config.test_path（验证数据集目录）

输出：
    config.model_path（模型文件保存目录，最开始的创建模型）
    config.train_for_saved_model_path（模型文件保存目录，在现有的模型基础上迭代）
'''

root_dir = os.path.dirname(__file__)

config = Config()
# 创建训练图像数据增强器
image_data_gen = ImageDataGenerator(rescale=1. / 255,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.5,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

# 使用图像生成器提取训练集图像数据特征进行训练
train_pic_flow = image_data_gen.flow_from_directory(config.train_path,
                                                    target_size=(config.resize, config.resize),
                                                    batch_size=config.batch_size,
                                                    class_mode='categorical')
# 使用图像生成器提取测试集图像数据特征进行验证
test_pic_flow = image_data_gen.flow_from_directory(config.test_path,
                                                   target_size=(config.resize, config.resize),
                                                   batch_size=config.batch_size,
                                                   class_mode='categorical')


def train_for_blank():
    # 设置类别（此处是2分类，即每个目录下是两个分类）
    classes = 2
    model = LeNet(config.resize, classes)
    # 创建一个检查点，该检查点配置模型保存的路径，并会监测训练损失率，还会保存最优模型，输出过程日志，自动训练一个周期
    check_point = ModelCheckpoint(config.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                  period=1)
    # 创建一个最早停止点
    early_stop = EarlyStopping(monitor='val_loss', patience=200, verbose=1)
    # 创建一个随机梯度下降优化器，概念参考 https://blog.csdn.net/liming89/article/details/111059213
    sgd = SGD(lr=config.lr, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit_generator(train_pic_flow,
                                  steps_per_epoch=config.steps_per_epoch,
                                  epochs=config.epoch,
                                  verbose=1,
                                  validation_data=test_pic_flow,
                                  validation_steps=500,
                                  workers=6,  # 运行线程数
                                  callbacks=[early_stop, check_point])
    return history, model


# 可视化模型训练过程, 将训练集与测试集的损失率和准确率合并进一张图片中，对比分析
# Matplotlib参数介绍   https://www.csdn.net/tags/MtTaEgxsMTkxMTI4LWJsb2cO0O0O.html
def visual_train(history, epochs):
    plt.figure()
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    # 记录输出训练及测试准确率
    print('train_acc:', train_acc)
    print('val_acc:', val_acc)
    # 绘图
    plt.plot(np.arange(epochs), train_acc, color='b', label="Train")  # 训练模型的准确率
    plt.plot(np.arange(epochs), val_acc, color='g', label="Test")  # 模型验证的损失率
    values = range(0, config.epoch + 10, 10)  # 等间距绘制
    plt.xticks(values)  # 修改x轴刻度数值
    plt.legend(loc='upper left', fontsize=10)  # 指定标签位置
    # plt.show()
    plt.savefig("data/model/visual_train_batch04.png", format="png")


if __name__ == '__main__':
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history, model = train_for_blank()
    visual_train(history, config.epoch)
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start_time:', start_time, 'end_time:', end_time)

