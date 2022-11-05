from main.data_preprocess.data_process import aHash, dHash, pHash, get_thum, get_picture_name
from PIL import Image
import cv2
import os
from numpy import average
from config import Config
from keras.preprocessing.image import ImageDataGenerator


'''
【数据预处理】
该用例主要是测试数据预处理的过程
具体为hash（ahash、phash、dhash）、cos处理前图片像素平均值、conv卷积前图像处理向量

'''

root_dir = os.path.dirname(__file__)

sample_img_path = os.path.join(root_dir, 'data/dataset/hash/sample')
sample_img_name = get_picture_name(sample_img_path)


# 打印图片文件的哈希值
def hash_str(hash):
    for i in sample_img_name:
        img2 = cv2.imread(os.path.join(sample_img_path, i))
        if hash == "aHash":
            hash1 = aHash(img2, 8)
        elif hash == "dHash":
            hash1 = dHash(img2, 8)
        elif hash == "pHash":
            hash1 = pHash(img2, 8)
        else:
            break
        print('{}:\t'.format(i), hash1)


# 打印图片的余弦值（）
def cos_str():
    for i in sample_img_name:
        vector = []
        img2 = Image.open(os.path.join(sample_img_path, i))
        image = get_thum(img2)
        # 平均像素值
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        print('{}:\t'.format(i), vector)

# 打印图片的余弦向量
def conv_str(index_pic):
    config = Config()

    # 对测试图像进行数据增强
    test_pic_gen = ImageDataGenerator(rescale=1. / 255)

    # 利用 .flow_from_directory 函数生成测试数据
    test_flow = test_pic_gen.flow_from_directory(os.path.join(root_dir, 'data/dataset/hash'),
                                                 target_size=(config.resize, config.resize),
                                                 batch_size=1,
                                                 class_mode='categorical')
    index_pic -= 1
    print(test_flow[index_pic])


# 对图片进行预处理后，输出到指定目录，具体就是将网页处理层224x224的尺寸的标准输入图像。(每张图片会生成2个标准输入图像)
def output_img():
    config = Config()
    train_pic_gen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.5,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    train_flow = train_pic_gen.flow_from_directory(os.path.join(root_dir, 'data/dataset/ct_17/valid'),
                                                   target_size=(config.resize, config.resize),
                                                   batch_size=config.batch_size,
                                                   save_to_dir=os.path.join(root_dir, 'data/dataset/output'),
                                                   class_mode='categorical',
                                                   )
    # 控制数据输出的量，此处设置为输出两个样本图片，i=0,1,2
    i = 0
    for s in train_flow:
        i += 1
        if i > 2:
            break


if __name__ == "__main__":
    # hash
    hash_str(hash='aHash')  # aHash, dHash, pHash
    # # cos
    # cos_str()
    # # conv
    # conv_str(1)  # 1, 2, 3
    # # 图像增强()
    # output_img()
