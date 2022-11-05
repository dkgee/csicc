from keras.optimizers import SGD
from main.classify.model import LeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from config import Config
from keras.models import load_model
config = Config()

# 对训练图像进行数据增强
train_pic_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# 对测试图像进行数据增强
test_pic_gen = ImageDataGenerator(rescale=1./255)
# 利用 .flow_from_directory 函数生成训练数据
train_flow = train_pic_gen.flow_from_directory(config.train_path_01,
                                               target_size=(224, 224),
                                               batch_size=64,
                                               class_mode='categorical')
# 利用 .flow_from_directory 函数生成测试数据
test_flow = test_pic_gen.flow_from_directory(config.test_path_01,
                                             target_size=(224, 224),
                                             batch_size=64,
                                             class_mode='categorical')


def train_for_blank():
    model = LeNet(config.resize, 2)

    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(config.model_path_01, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                 period=1)
    callbacks_list = [checkpoint]

    sgd = SGD(lr=config.lr, decay=1e-6, momentum=0.8, nesterov=True)

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


def train_for_saved():
    model = load_model(config.model_path_01)

    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(config.train_for_saved_model_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='auto', period=1)
    callbacks_list = [checkpoint]

    sgd = SGD(lr=config.lr, decay=1e-6, momentum=0.9, nesterov=True)

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


# 重新开始训练
# train_for_blank()
# 继续训练
train_for_saved()
