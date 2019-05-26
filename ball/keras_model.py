import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sqlalchemy import create_engine


def max_min_norm(array):
    maxcols = array.max(axis=0)
    mincols = array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_cols):
        t[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t


sess = tf.Session()

engine_ball = create_engine("mysql+pymysql://root:1122@10.93.53.244:3306/db_ball_dataset")
train_x = pd.read_sql_table('b1_train_x_array', con=engine_ball).values
train_y = pd.read_sql_table('b1_train_y_array', con=engine_ball).values
train_y_categorical = keras.utils.to_categorical(train_y, 33)
# train_y_one_hot = sess.run(tf.one_hot(train_y, 33)).to_list()
# print(train_y_one_hot)
model = keras.Sequential()

model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation=tf.nn.tanh))
# model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(33, activation=tf.nn.softmax))

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy', 'mse']  # 矩阵
)

hist = model.fit(x=train_x, y=train_y_categorical, batch_size=5, epochs=500, validation_split=0.2)

model.save('my-model.h5')
