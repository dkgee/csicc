import shutil

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from config import Config
import os
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='lenet', required=True,
                    help='choose a model: lenet, alexnet, zfnet, model')
parser.add_argument('--classes', type=int, required=True, help='template classes')
parser.add_argument('--acc', type=float, default=0.9, required=True, help='expected accuracy')
args = parser.parse_args()
x = import_module('models.' + args.model)
print(args.acc)

model = x.Model(224, 2)
root_dir = os.path.dirname(__file__)

# 模型存储地址
model_path = os.path.join(root_dir, 'data/save_model/ss_model_{}-weights.best.hdf5'.format(args.classes))
# 训练数据地址
train_path = os.path.join(root_dir, 'dataset/CT_{}/train'.format(args.classes))
# 测试数据地址
test_path = os.path.join(root_dir, 'dataset/CT_{}/test'.format(args.classes))

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
train_flow = train_pic_gen.flow_from_directory(train_path,
                                               target_size=(config.resize, config.resize),
                                               batch_size=config.batch_size,
                                               # save_to_dir="./dataset",
                                               class_mode='categorical')
# 利用 .flow_from_directory 函数生成测试数据
test_flow = test_pic_gen.flow_from_directory(test_path,
                                             target_size=(config.resize, config.resize),
                                             batch_size=config.batch_size,
                                             class_mode='categorical')


def train_for_blank():

    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
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


# 模型训练
s = train_for_blank()
print(s.history['val_acc'])

model_npath = os.path.join(root_dir, "data/save_model/")

if s.history['val_acc'][-1] > args.acc:
    shutil.move(model_path, model_npath)
    print("The accuracy rate of the new model is {} "
          "and it has been automatically deployed and launched !".format(s.history['val_acc']))
else:
    print("The accuracy of the model is {}, which is not up to standard, please retrain!".format(s.history['val_acc']))
    i = input("Whether to delete substandard models y/n ?")
    if i == "y":
        os.remove(model_path)
        print("successfully deleted !")
    elif i == "n":
        print("model is stored in {}".format(model_path))
    else:
        print("wrong input, model is stored in {}".format(model_path))
