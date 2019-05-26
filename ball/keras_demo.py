from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

'''
keras实现神经网络分类模型
'''
# 读取数据
path = 'df2.csv'
train_df = pd.read_csv(path)
# 删掉不用字符串字段
dataset = train_df.drop('jh', axis=1)
# df转array
values = dataset.values

y = values[:, -1]
X = values[:, 0:-1]
# 必须标准化，否则难以收敛
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# 随机拆分训练集与测试集
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, stratify=y)
# 多分类转换label
# nb_classes = 3
# train_Y = np_utils.to_categorical(train_y, nb_classes)
# test_Y = np_utils.to_categorical(test_y, nb_classes)
# 全连接神经网络
model = Sequential()
input = X.shape[1]
# 隐藏层128
model.add(Dense(128, input_shape=(input,)))
model.add(Activation('relu'))
# Dropout层用于防止过拟合
model.add(Dropout(0.2))
# 隐藏层128
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# 没有激活函数用于输出层，二分类问题，用sigmoid激活函数进行变换，多分类用softmax。
model.add(Dense(1))
model.add(Activation('sigmoid'))
# 使用高效的 ADAM 优化算法以，二分类损失函数binary_crossentropy，多分类的损失函数categorical_crossentropy
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# early stoppping
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_acc', patience=50, verbose=2)
# 训练
history = model.fit(train_X, train_y, epochs=400, batch_size=20, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False, callbacks=[early_stopping])  # loss曲线
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# 预测
y_pre = model.predict_classes(test_X)
#
print(classification_report(test_y, y_pre, labels=[0, 1]))
print(confusion_matrix(test_y, y_pre))
