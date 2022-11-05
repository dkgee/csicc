
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

'''
【数据预处理】
该文件主要用于图片分类单元测试
'''
# seed = 1
# train_pic_gen = ImageDataGenerator(rescale=1. / 255,
#                                    rotation_range=20,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.5,
#                                    horizontal_flip=True,
#                                    fill_mode='nearest')

# import keras
# print(keras.__version__)

# 设置简单的图片处理器即可，不用太复杂（不可以，还是需要裁剪、缩放处理）
# 在图片处理方式上，采用宽高进行裁剪处理即可，不用旋转、缩放、翻转处理，这种一方面会增加模型的复杂度，占用大量空间，实际上并没有太大作用，
# 因为网页图片内容为了方便用户阅读，都是按顺序展示，不会像其他空间对象提取的图像出现旋转，或远近的放大缩小现象。在后续处理的过程中，除了
# 才开始对原始的图像进行尺寸的归一化处理进行缩放外，后续不再进行缩放、旋转等处理。
# 默认会将图像统一转换为256×256的尺寸。
train_pic_gen = ImageDataGenerator(rescale=1. / 255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.4,  # 因为有的网页头部会有广告或置顶消息
                                   shear_range=0.2,
                                   fill_mode='nearest')

train_pic_flow = train_pic_gen.flow_from_directory('data/dataset/ct_17/train',
                                                   target_size=(224, 224),
                                                   batch_size=4,
                                                   save_prefix='image',
                                                   save_to_dir='data/predata',  # 处理的图像输出目录
                                                   class_mode='categorical')

print(type(train_pic_flow))
# 通过循环迭代每一次的数据，并进行查看

'''
图片数据流会无限迭代，迭代时从数据集获取一批数据，每个批次从数据集中随机选取一批处理后数据再进行处理（按照之前的设置规则方式处理）
原始图片 ——> [ImageDataGenerator] ——> 处理完成的图片
'''
count = 1
for x_batch_image, y_class in train_pic_flow:
    if count > 10:
        break
    print(F"------------开始第{count}次迭代-----------------------------")
    print(F"------------x_batch的形状如下----------------------")
    print(np.shape(x_batch_image))
    print('-------------y_class打印结果如下-----------------------------')
    print(y_class)
    # print('-------------x_batch_image[0]打印结果如下（对象太大）-----------------------------')
    # print(x_batch_image[0])
    print('============================================================')
    # 将每次augmentation之后的2幅图像显示出来(每批次取3时，此处最大值不能超过3个)
    for i in range(4):
        plt.subplot(2, 2, i + 1)  # 分割成2行，2列，展示在水平第i个位置
        plt.imshow(x_batch_image[i])  # 每个批次数据集中的第i张图片(i <= 单批次数量 )
        # 使用plt预览的6张图片的文件夹
        plt.savefig("data/predata/iter_" + str(count) + ".png", format="png")
    # plt.show()
    count += 1

