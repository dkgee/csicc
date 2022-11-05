import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from main.data_preprocess.data_process import drop_invaild_data, get_picture_name
from config import Config
import time
import re
from pathlib import Path
root_dir = os.path.dirname(__file__)
config = Config()
s = time.time()

# TODO 模型路径参数不对
modelA = load_model(config.model_path_A)
modelB = load_model(config.model_path_B)

"""
导入你的模型
导入你的参数
"""


# 预测图片复制到指定位置
def copy_picture_to_folder(old_path, new_path, picture_name):
    shutil.copyfile(old_path, os.path.join(new_path, picture_name))


# 将数据转换为list
def to_list(predicted_class_indices):
    l = str(predicted_class_indices)
    l = l.replace('[', '').replace(']', '')
    m = []
    label_0 = 0
    label_1 = 0
    for j in l.split(' '):
        if j == "0":
            m.append(0)
            label_0 += 1
        else:
            m.append(1)
            label_1 += 1
    return m


# 获取真实的list标签
def get_true_label(filenames, label):
    true_label = []
    for j in filenames:
        if re.search(label[0], j):
            true_label.append(0)
        else:
            true_label.append(1)
    return true_label


def get_predict_label(predict_list):
    pre_list = []
    for i in predict_list:
        if i[1] > config.percent:
            pre_list.append(1)
        else:
            pre_list.append(0)
    return pre_list


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


def write2dir(filenames, pred, path, ty):
    # 将待预测图片进行预测，并写入指定文件夹
    current_time = time.strftime("%Y-%m-%d")
    predict_1 = 0
    predict_0 = 0
    names = []
    for idx, name in zip(range(len(filenames)), filenames):
        name = str(name)
        name = name.split('\\')[1]
        old_path = os.path.join(config.predict_dir_path, '{}'.format(str(filenames[idx])))
        new_name = round(pred[idx][1], 4)
        if pred[idx][1] < config.percent:
            # copy_picture_to_folder(old_path, config.predictUNKpath, '{}_{}_{}'.format(str(current_time),
            #                                                                               str(new_name),
            #                                                                               name))
            predict_0 += 1
        else:
            copy_picture_to_folder(old_path, path, '{}_{}_{}'.format(str(current_time),
                                                                         str(new_name),
                                                                         name))
            names.append(name)
            predict_1 += 1
        # print("picture name:{:50}".format(filenames[idx]), "label=1 precent:", pred[idx][1])
    print('predict {}:'.format(ty), predict_1)
    print('predict not {}:'.format(ty), predict_0)
    return names


myfile = Path(config.predict_dir_path)
if myfile.exists():
    # 数据预处理
    drop_num = drop_invaild_data(config.predict_dir_path)
    print('drop data number:', drop_num)
    # 对需要预测文件进行预处理（待预测文件应该属于二级文件夹如在 ./predict/unk/+.png）
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(config.predict_dir_path,
                                                      target_size=(224, 224), batch_size=1,
                                                      class_mode='categorical', shuffle=False)

    test_generator.reset()

    test_datagenB = ImageDataGenerator(rescale=1. / 255)
    test_generatorB = test_datagenB.flow_from_directory(config.predict_dir_path,
                                                        target_size=(224, 224), batch_size=1,
                                                        class_mode='categorical', shuffle=False)

    test_generatorB.reset()

    # 预测结果
    predA = modelA.predict_generator(test_generator, verbose=1, steps=len(test_generator))
    predB = modelB.predict_generator(test_generatorB, verbose=1, steps=len(test_generatorB))

    # 将预测概率转换为标签
    # predicted_class_indices = np.argmax(pred, axis=1)
    predicted_class_indices = get_predict_label(predA)
    filenames = test_generator.filenames
    filenamesB = test_generatorB.filenames
    del_files(config.predictApath)
    del_files(config.predictBpath)
    del_files(config.predictUNKpath)
    namesA = write2dir(filenames, predA, config.predictApath, 'A')
    print(namesA)
    namesB = write2dir(filenamesB, predB, config.predictBpath, "B")
    print(namesB)
    names = get_picture_name(config.predict_dir_path)
    for idx, name in zip(range(len(filenames)), filenames):
        name = str(name)
        name = name.split('\\')[1]
        old_path = os.path.join(config.predict_dir_path, '{}'.format(str(filenames[idx])))
        if name not in namesA:
            if name not in namesB:
                old_path = os.path.join(config.predict_dir_path, '{}'.format(str(filenames[idx])))
                # print(old_path)
                copy_picture_to_folder(old_path, config.predictUNKpath, name)

    print('model predict done !')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "cleaning the source dir...")
    # del_files(config.predict_dir_path)
    e = time.time()
    print('using time:{} s'.format(e-s), "\n")
else:
    print("target dir not exit ...", "\n")
