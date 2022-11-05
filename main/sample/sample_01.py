
import keras
from keras.layers import Dense
import numpy as np


'''
理解：神经网络可以学习任何输入和输出之间的映射，并且可用作函数逼近器，一旦学习了映射，它就可以为我们挺高给它的任何输入
生成近似输出
'''

model = keras.models.Sequential()

model.add(Dense(units=1, use_bias=False, input_shape=(1,)))   # 此处只有1个权重
# MSE:均方损失函数
model.compile(loss='mse', optimizer='adam')

# 创建训练数据
data_input = np.random.normal(size=100000) # 生成随机数
data_label = -(data_input)  # 数据标签


# 训练网络
# 检查权重(w)大小
print('模型随机权重分配为：%s' % (model.layers[0].get_weights()))    # [array([[1.6690887]], dtype=float32)]
# 进行训练（10万个样本训练），fitting是training的别称, fit：有计划地锻炼，与训练是一个意思
model.fit(data_input, data_label, epochs=1, batch_size=1, verbose=1)

# 使用训练后的模型，验证响应
predict_data = model.predict(np.array([2.5]))

print('模型进行预测：', predict_data)  # [[-2.4999993]]

# 网络预测一个非常接近的数值，并且带有负号，随着将更多的数据训练它，将输出越来越接近目标的值，此时再检查模型中的权重，
# 可以看到权重值从一开始的随机数变为大约-1，这很明显，因为必须将数字乘以-1才能更改其符号。
print('训练完成后权重分配为：%s' % (model.layers[0].get_weights()))    # [array([[-1.0000031]], dtype=float32)]

