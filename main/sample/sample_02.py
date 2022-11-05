
from keras.models import load_model
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

'''
基于现有的模型，输出预测的图片数据，
【测试通过】，可加载现有的模型对图片文件进行预测，并将类别标签与图像之间关系输出来
'''

root_dir = os.path.dirname(__file__)
# model_path = os.path.join(root_dir, "../../data/model/lenet_model_17-weights.best.hdf5")    # 模型文件所在位置
# test_path = os.path.join(root_dir, '../../data/dataset/ct_17/test')     # 测试文件所在位置
test_pic_save_path = os.path.join(root_dir, '../../data/dataset/output')    # 提取的测试图片保存位置

model_path = os.path.join(root_dir, "../../data/model/lenet_model_tk_01-weights.best.hdf5")    # 模型文件所在位置
test_path = os.path.join(root_dir, '../../data/dataset02/test')     # 测试文件所在位置
batch_size = 1  # 只取1批数据

# 根据模型地址加载模型文件
model = load_model(model_path)

# 从测试文件所在位置提取一张图片，使用图像数据生成器进行处理
test_pic_gen = ImageDataGenerator(rescale=1./255)
test_pic_flow = test_pic_gen.flow_from_directory(test_path,
                                                 target_size=(224, 224), batch_size=batch_size,
                                                 # save_to_dir=test_pic_save_path,
                                                 class_mode='categorical',
                                                 shuffle=False)
# 将生成器重置，避免识别过程中不对应
test_pic_flow.reset()
# 对数据进行预测
labels = test_pic_flow.class_indices   # 此处有问题，文件流中的目录不一定就是标签
filenames = test_pic_flow.filenames
# 预测出来是一个集合（包含该类型的概率阀值，0~1之间），遍历集合名称，如果大于该概率阀值，则加入结果
predict_result = model.predict_generator(test_pic_flow, verbose=1, steps=len(test_pic_flow))
# score = model.evaluate_generator(test_pic_flow, verbose=1, steps=len(test_pic_flow))

print('图片类别标签：', labels)
print('图片文件名称：', filenames)
# 0.95

predicted_class_indices = np.argmax(predict_result, axis=1)
label = dict((v, k) for k, v in labels.items())

# 建立代码标签与真实标签的关系
predictions = [label[i] for i in predicted_class_indices]

# 建立预测结果和文件名之间的关系
for idx in range(len(filenames)):
    print('图片文件：%s' % (filenames[idx]), ',预测结果：%s' % (predictions[idx]))

# print('评估得分：', score)

