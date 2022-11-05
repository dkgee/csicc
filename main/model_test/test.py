import os
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from main.data_preprocess.data_process import drop_invaild_data, get_predict_label, get_true_label
from config import Config
import time
import pandas as pd

root_dir = os.path.dirname(__file__)
config = Config()
s = time.time()

model = load_model(config.predict_model_path)

df = pd.read_csv("E:/data/doc/ct_site_text_202202151150.csv", encoding="gb18030")
# key_text = "Powered by SSPANEL Theme by editXY"

"""
导入你的模型
导入你的参数
"""

# 数据预处理
drop_num = drop_invaild_data(config.test_path)
print('drop data number:', drop_num)
# 对需要预测文件进行预处理（待预测文件应该属于二级文件夹如在 ./predict/unk/+.png）
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(config.test_path,
                                                  target_size=(224, 224), batch_size=1,
                                                  class_mode='categorical', shuffle=False)

test_generator.reset()

# 预测结果
pred = model.predict_generator(test_generator, verbose=1, steps=len(test_generator))

# 将预测概率转换为标签
# predicted_class_indices = np.argmax(pred, axis=1)
if config.num <= 9:
    ss_key = 'per_0{}'.format(config.num)
else:
    ss_key = 'per_{}'.format(config.num)
predicted_class_indices = get_predict_label(pred, config.predict_params[ss_key][1])
filenames = test_generator.filenames


def model_eval():
    # 模型评估(仅对已知数据进行评估)

    score = model.evaluate_generator(test_generator, verbose=1, steps=len(test_generator))
    print(score)

    labels = test_generator.class_indices
    label = dict((v, k) for k, v in labels.items())
    # print('label', label)
    true_label = get_true_label(filenames, label)
    # m = to_list(predicted_class_indices)
    m = predicted_class_indices
    _val_recall = recall_score(true_label, m)
    f1 = f1_score(true_label, m)
    _val_precision = precision_score(true_label, m)

    # 计算评估指标
    print('\n')
    print("--------------------评估指标--------------------")
    print("\n")
    print('set percent:', config.predict_params[ss_key][1])
    print("混淆矩阵图")
    print(confusion_matrix(true_label, m))
    print('{:10}: {}%'.format("Precision", round(_val_precision*100, 3)))
    print("{:10}: {}%".format("Accuracy", round(accuracy_score(true_label, m)*100, 3)))
    print('{:10}: {}%'.format("Recall", round(_val_recall*100, 3)))
    print('{:10}: {}%'.format("F1", round(f1*100, 3)))


model_eval()
e = time.time()
print(e-s)
