from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from main.classify.model import LeNet
from config import Config
import os

'''
【数据预处理+模型训练】
模型训练主类，此Demo演示模型训练过程，能顺利跑起来

此文件外界输入：
    config.train_path（训练数据集目录）
    config.test_path（验证数据集目录）

输出：
    config.model_path（模型文件保存目录，最开始的创建模型）
    config.train_for_saved_model_path（模型文件保存目录，在现有的模型基础上迭代）
'''


# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, default='lenet', required=True,
#                     help='choose a model: lenet, alexnet, zfnet, model')
# args = parser.parse_args()
# x = import_module('models.' + args.model)

# model = x.Model(224, 2)
root_dir = os.path.dirname(__file__)

config = Config()
# 创建训练图像数据增强器
train_pic_gen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# 创建测试图像数据增强器
test_pic_gen = ImageDataGenerator(rescale=1. / 255)

# 利用图像数据增强器从配置的训练数据集目录中批量读取图像数据流（在此过程中，对图片的尺寸进行了归一化处理）
train_pic_flow = train_pic_gen.flow_from_directory(config.train_path,
                                                   target_size=(config.resize, config.resize),
                                                   batch_size=config.batch_size,
                                                   # save_to_dir='./data/predata',  # 输出图像目录
                                                   class_mode='categorical')
# 同上，对验证数据集进行处理
validate_pic_flow = test_pic_gen.flow_from_directory(config.test_path,
                                                     target_size=(config.resize, config.resize),
                                                     batch_size=config.batch_size,
                                                     class_mode='categorical')

# 只训练模型，但不保存模型文件
'''
训练100个周期，每个周期进行迭代，模型准确率从
'''
def train_for_blank():
    # 【定义模型】
    model = LeNet(config.resize, 2)
    # 创建一个检查点，该检查点配置模型保存的路径，并会监测训练损失率，还会保存最优模型，输出过程日志，自动训练一个周期
    check_point = ModelCheckpoint(config.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                  period=1)
    # 创建一个最早停止点
    early_stop = EarlyStopping(monitor='val_loss', patience=200, verbose=1)
    # 创建一个随机梯度下降优化器，概念参考 https://blog.csdn.net/liming89/article/details/111059213
    sgd = SGD(lr=config.lr, decay=1e-5, momentum=0.9, nesterov=True)
    # 【编译模型】，编译过程中要指定损失函数、优化方法、模型评估指标
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    his = model.fit_generator(train_pic_flow,
                              steps_per_epoch=config.steps_per_epoch,
                              epochs=config.epoch,
                              verbose=1,
                              validation_data=validate_pic_flow,
                              validation_steps=500,
                              callbacks=[early_stop, check_point])
    return his


# 训练并保存模型文件
def train_for_save():
    model = load_model(config.model_path)
    check_point = ModelCheckpoint(config.train_for_saved_model_path, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='auto', period=1)
    callback_list = [check_point]
    sgd = SGD(lr=config.lr, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    his = model.fit_generator(train_pic_flow,
                              steps_per_epoch=config.steps_per_epoch,
                              epochs=config.epoch,
                              verbose=1,
                              validation_data=validate_pic_flow,
                              validation_steps=500,
                              callbacks=callback_list)
    return his

# 模型训练
train_for_blank()
# 在原来模型的基础上迭代
# train_for_saved()