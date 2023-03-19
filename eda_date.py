import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# 雨量のデータの読み込み
path = 'run_test/train/'

#2年目のデータを探索
rain_data = pd.read_csv('rain_data_time.csv')


#1次元配列(numpy)
rain_data_cos = np.cos(2 * np.pi * (rain_data["date"]+1)/365)
rain_data_sin = np.sin(2 * np.pi * (rain_data["date"]+1)/365)

#2次元配列
A = np.array([rain_data_cos, rain_data_sin])
C = A.T
# print(rain_data_sin.shape)
rain_data_d_trigo = pd.DataFrame(C)
rain_data_d_trigo.columns = ["D_COS","D_SIN"]
print(rain_data_d_trigo)





# def make_dataset():
#     values = []
#     for i in range(0, 365):
#         now = datetime.datetime(2021, 1, 1) + datetime.timedelta(days=i)
# #       season = SEASON[now.month]
#         values.append((now.date(), i))

#     df = pd.DataFrame(values, columns=["date", "yday", "season"])
#     df = encode(df, "yday")
#     return df
    
# df = make_dataset()
# print(df)