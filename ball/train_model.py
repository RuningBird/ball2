import pandas as pd
import tensorflow as tf
import numpy as np
from sqlalchemy import create_engine


class TrainModel:
    engine_ball = create_engine("mysql+pymysql://root:1122@10.93.53.244:3306/db_ball_dataset")

    def __init__(self):
        self.train_x = pd.read_sql_table('b1_train_x_data', con=self.engine_ball)
        self.train_y = pd.read_sql_table('b1_train_y_data', con=self.engine_ball)

    def fit(self):
        pass

    def build_model(self, train_x, train_yl):
        pass
