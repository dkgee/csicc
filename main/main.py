from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from config import Config
import os
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='lenet', required=True,
                    help='choose a model: lenet, alexnet, zfnet, model')
args = parser.parse_args()
x = import_module('models.' + args.model)

model = x.Model(224, 2)
root_dir = os.path.dirname(__file__)


config = Config()
# 对训练图像进行数据增强
train_pic_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# 对测试图像进行数据增强
test_pic_gen = ImageDataGenerator(rescale=1./255)
# 利用 .flow_from_directory 函数生成训练数据
train_flow = train_pic_gen.flow_from_directory(config.train_path,
                                               target_size=(config.resize, config.resize),
                                               batch_size=config.batch_size,
                                               # save_to_dir="./dataset",
                                               class_mode='categorical',
                                               )
# 利用 .flow_from_directory 函数生成测试数据
test_flow = test_pic_gen.flow_from_directory(config.test_path,
                                             target_size=(config.resize, config.resize),
                                             batch_size=config.batch_size,
                                             class_mode='categorical')


def train_for_blank():

    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(config.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                 period=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=200, verbose=1)

    sgd = SGD(lr=config.lr, decay=1e-5, momentum=0.9, nesterov=True)

    model.compile(loss="categorical_crossentropy",
                  optimizer='adam',
                  metrics=['accuracy'])

    his = model.fit_generator(train_flow,
                              steps_per_epoch=config.steps_per_epoch,
                              epochs=config.epoch,
                              verbose=1,
                              validation_data=test_flow,
                              validation_steps=500,
                              callbacks=[early_stop, checkpoint])
    return his


def train_for_saved():
    model = load_model(config.model_path)

    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(config.train_for_saved_model_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint]

    sgd = SGD(lr=config.lr, decay=1e-5, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    his = model.fit_generator(train_flow,
                              steps_per_epoch=config.steps_per_epoch,
                              epochs=config.epoch,
                              verbose=1,
                              validation_data=test_flow,
                              validation_steps=500,
                              callbacks=callbacks_list)
    return his


# 模型训练
train_for_blank()
# 在原来模型的基础上迭代
# train_for_saved()
