#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session

engine_ball = create_engine("mysql+pymysql://root:1122@10.93.53.244:3306/db_ball_dataset")

# print(os.path.abspath(__file__))

# df_std = pd.read_csv('projects/ball/dataset/process_dataset.csv')
# df_data = pd.read_csv('projects/ball/dataset/ball.csv')
df_std = pd.read_csv('./dataset/process_dataset.csv')  # 运行时写法，python console模式下会提示找不到路径r
df_data = pd.read_csv('./dataset/ball.csv')


def get_y(df_data_p=None, n=0):
    row_length = df_data_p.shape[0]
    y_list = []
    for i in range(row_length):
        try:
            y = df_data_p.iat[i + 5, n]
            y_list.append(y)
        except Exception as e:
            pass
    return np.array([y_list]).T


def get_x(df_data_p=None):
    row_length = df_data_p.shape[0]
    x_list = []
    x_l = []
    for i in range(row_length):
        try:
            x = df_data_p.iloc[i:i + 5, :].values
            x_list.append(np.reshape(x, 35))
            x_l.append(x)
        except Exception as e:
            pass
    x_list.pop()
    return np.array(x_list)


b1_train_x_array = get_x(df_data)
d1 = pd.DataFrame(b1_train_x_array)
d1.to_sql(name='b1_train_x_array', con=engine_ball, index=False, if_exists='replace')

b1_train_y_array = get_y(df_data, n=0)
d1 = pd.DataFrame(b1_train_y_array)
d1.to_sql(name='b1_train_y_array', con=engine_ball, index=False, if_exists='replace')


b2_train_x_array = np.concatenate((b1_train_x_array, b1_train_y_array), axis=1)
b2_train_y_array = get_y(df_data, n=1)
pd.DataFrame(b2_train_x_array).to_sql(name='b2_train_x_array', con=engine_ball, index=False, if_exists='replace')
pd.DataFrame(b2_train_y_array).to_sql(name='b2_train_y_array', con=engine_ball, index=False, if_exists='replace')

b3_train_x_array = np.append(b2_train_x_array, b2_train_y_array, axis=1)
b3_train_y_array = get_y(df_data, n=2)
pd.DataFrame(b3_train_x_array).to_sql(name='b3_train_x_array', con=engine_ball, index=False, if_exists='replace')
pd.DataFrame(b3_train_y_array).to_sql(name='b3_train_y_array', con=engine_ball, index=False, if_exists='replace')

b4_train_x_array = np.append(b3_train_x_array, b3_train_y_array, axis=1)
b4_train_y_array = get_y(df_data, n=3)
pd.DataFrame(b4_train_x_array).to_sql(name='b4_train_x_array', con=engine_ball, index=False, if_exists='replace')
pd.DataFrame(b4_train_y_array).to_sql(name='b4_train_y_array', con=engine_ball, index=False, if_exists='replace')

b5_train_x_array = np.append(b4_train_x_array, b4_train_y_array, axis=1)
b5_train_y_array = get_y(df_data, n=4)
pd.DataFrame(b5_train_x_array).to_sql(name='b5_train_x_array', con=engine_ball, index=False, if_exists='replace')
pd.DataFrame(b5_train_y_array).to_sql(name='b5_train_y_array', con=engine_ball, index=False, if_exists='replace')

b6_train_x_array = np.append(b5_train_x_array, b5_train_y_array, axis=1)
b6_train_y_array = get_y(df_data, n=5)
pd.DataFrame(b6_train_x_array).to_sql(name='b6_train_x_array', con=engine_ball, index=False, if_exists='replace')
pd.DataFrame(b6_train_y_array).to_sql(name='b6_train_y_array', con=engine_ball, index=False, if_exists='replace')

b7_train_x_array = np.append(b6_train_x_array, b6_train_y_array, axis=1)
b7_train_y_array = get_y(df_data, n=6)
pd.DataFrame(b7_train_x_array).to_sql(name='b7_train_x_array', con=engine_ball, index=False, if_exists='replace')
pd.DataFrame(b7_train_y_array).to_sql(name='b7_train_y_array', con=engine_ball, index=False, if_exists='replace')

print('oj')
