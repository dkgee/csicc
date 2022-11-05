
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, confusion_matrix
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from main.data_preprocess.data_process import drop_invaild_data, get_predict_label, get_true_label, to_list
from config import Config
import time
import os

'''
【数据预处理+模型预测】
该用例主要是测试模型预测的过程

'''

root_dir = os.path.dirname(__file__)

# step01: 加载配置及模型文件
config = Config()
model = load_model(config.predict_model_path)
s = time.time()

# 注意：对于待预测数据，要提前对数据进行检查后，清理无效的数据（譬如可能由于网络或磁盘故障，导致图片数据完整，从而无法打开图片）
drop_invaild_data(config.test_path)

# step02：创建图片生成器，加载待预测的图片
test_pic_gen = ImageDataGenerator(rescale=1./255)

# 注意此处的数据输入要和模型训练时相匹配
test_pic_flow = test_pic_gen.flow_from_directory(config.test_path,
                                                 target_size=(config.resize, config.resize), batch_size=config.batch_size,
                                                 class_mode='categorical',
                                                 shuffle=False)

# 重置批索引大小为0
test_pic_flow.reset()

# step03：开始预测（结果是百分比，即属于该类别的可能性有多大，）
predict_result = model.predict_generator(test_pic_flow, verbose=1, steps=len(test_pic_flow))


# step04: 预测结果转换可读标签

# 读取预设的阀值，当超过该阀值时，即表明属于该类别
if config.num < 9:
    ss_key = 'per_0{}'.format(config.num)
else:
    ss_key = 'per_{}'.format(config.num)
# 预测结果归属类别
predicted_class_indices = get_predict_label(predict_result, config.predict_params[ss_key][1])

print('预测类别索引：', predicted_class_indices)

# 模型评估
def model_eval():
    '''
    对已知数据进行评估
    :return:
    '''

    score = model.evaluate_generator(test_pic_flow, verbose=1, steps=len(test_pic_flow))
    print('评估得分：', score)

    # 获取目录标签名
    labels_map = test_pic_flow.class_indices
    # print('标签字典1：', labels_map)
    label_list = []
    for k, v in labels_map.items():
        label_list.append(k)
    print('标签数组：', label_list)
    filenames = test_pic_flow.filenames
    true_label = get_true_label(filenames, label_list)
    # 预测标签
    m = predicted_class_indices
    # 获取评估结果
    precision_val = precision_score(true_label, m)
    accuracy_val = accuracy_score(true_label, m)
    recall_val = recall_score(true_label, m)
    f1_val = f1_score(true_label, m)
    confusion_val = confusion_matrix(true_label, m)

    # 计算评估指标
    print('\n')
    print("--------------------评估指标--------------------")
    print("\n")
    print('设置预测百分比阀值:', config.predict_params[ss_key][1])
    print("混淆矩阵图")
    print(confusion_val)
    print('{:10}: {}%'.format("Precision", round(precision_val * 100, 3)))
    print("{:10}: {}%".format("Accuracy", round(accuracy_val * 100, 3)))
    print('{:10}: {}%'.format("Recall", round(recall_val * 100, 3)))
    print('{:10}: {}%'.format("F1", round(f1_val * 100, 3)))


if __name__ == '__main__':
    model_eval()
    e = time.time()
    print('评估耗时（s）：', e - s)



