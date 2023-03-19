import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pickle

#%matplotlib inline
import japanize_matplotlib

path = 'run_test/train/'

################水位のデータの読み込み######################
water_data = pd.read_csv(path+'waterlevel/data.csv')
water_stations = pd.read_csv(path+'waterlevel/stations.csv')


########東西の判定########
def ew_juge(ew) :
    if ew <= 132.775 :
        return "w"
    else:
        return "e"
    
water_stations.loc[:,"東西"]  = water_stations.loc[:,"経度"].apply(ew_juge)

########南北の判定##########
def sn_juge(sn) :
    if sn <= 34.5362 :
        return "n"
    else:
        return "s"
    
water_stations.loc[:,"南北"]  = water_stations.loc[:,"緯度"].apply(sn_juge)

#water_stations.to_csv("water_stations_a.csv", encoding="UTF-8")

#########雨量のデータの読み込み############################
rain_data = pd.read_csv(path+'rainfall/data.csv')
rain_stations = pd.read_csv(path+'rainfall/stations.csv')

#######東西の判定#########
rain_stations.loc[:,"東西"]  = rain_stations.loc[:,"経度"].apply(ew_juge)
#######南北の判定#########
rain_stations.loc[:,"南北"]  = rain_stations.loc[:,"緯度"].apply(sn_juge)

#確認用CSV出力
#rain_stations.to_csv("rain_stations_a.csv", encoding="UTF-8")

# # #1年目のデータを探索
# rain_data = rain_data[0:150060]
# water_data = water_data[0:65514]

#2年目のデータを探索
rain_data = rain_data[150061:300122]
water_data = water_data[65515:131030]

#水位観測所のを入力

#例　json ファイルのinput["stations"]から

#stations = input['stations'] 

in_data = "大谷池"

water_stations_select = water_stations[water_stations["観測所名称"] == in_data]

#評価対象チェック処理

water_eval = water_stations_select[["評価対象"]]
water_eval = water_eval.to_numpy().tolist()
water_eval = water_eval[0][0]

print(water_eval)

if  water_eval == 1:
    
    pass

else:
    print("評価対象ではありません")
    exit()
# ここで検索不可なら「マスタに存在しません　もう一度確認して入力してください」

print(water_stations_select)

#水系名を抽出
water_type =water_stations_select[["水系名"]]
water_type = water_type.to_numpy().tolist()
water_type = water_type[0][0]

#河川名を抽出
water_river =water_stations_select[["河川名"]]
water_river = water_river.to_numpy().tolist()
water_river = water_river[0][0]

#東西を抽出
water_ew =water_stations_select[["東西"]]
water_ew = water_ew.to_numpy().tolist()
water_ew = water_ew[0][0]

# print("東西")
# print(water_ew)

#南北を抽出

water_sn =water_stations_select[["南北"]]
water_sn = water_sn.to_numpy().tolist()
water_sn = water_sn[0][0]

# print("南北")
# print(water_sn)

rain_stations_select = rain_stations[rain_stations["水系名"] == water_type ] 
rain_stations_select[rain_stations_select["河川名"] == water_river]
   
#河川名のチェック処理
        
if water_river in rain_stations_select["河川名"] :
      
    print('河川名は存在します')
    print(water_river)
    
    rain_stations_select1 = rain_stations_select[rain_stations_select["河川名"] == water_river]
    rain_stations_select2 = rain_stations_select1[rain_stations_select1["東西"] == water_ew ] 
    rain_stations_select3 = rain_stations_select2[rain_stations_select2["南北"] == water_sn ] 
    
else:
    print('河川名が存在しません')
    print(water_river)
    
    rain_stations_select2 = rain_stations_select[rain_stations_select["東西"] == water_ew ] 
    rain_stations_select3 = rain_stations_select2[rain_stations_select2["南北"] == water_sn ] 

# # print("最西")
# rain_stations_select4 = rain_stations_select3.loc[[rain_stations_select3["経度"].idxmin()]]

#print("最東")
print(rain_stations_select3)
rain_stations_select4 = rain_stations_select3.loc[[rain_stations_select3["経度"].idxmax()]]

print("最東")
print(rain_stations_select4)

rain_stations_name   = rain_stations_select4[["観測所名称"]]
rain_stations_name_np   = rain_stations_name.to_numpy().tolist()
print(rain_stations_name_np)

rain_stations_name =  rain_stations_name_np[0][0]

print("雨量観測所名称")
print(rain_stations_name)

#雨量観測所の市町を抽出
rain_stations_city   = rain_stations_select4[["市町"]]
rain_stations_city   = rain_stations_city.to_numpy().tolist()

rain_stations_city =  rain_stations_city[0][0]

print("雨量観測所市町")
print(rain_stations_city)

rain_data_w = rain_data[rain_data["station"] == rain_stations_name]
rain_data =  rain_data_w[rain_data_w["city"] == rain_stations_city]
water_data = water_data[water_data["station"] == in_data]

#雨量観測所
#時間軸の追加
rain_data = rain_data[rain_data['station']== rain_stations_name].set_index('date')
rain_data_time= rain_data[rain_data.columns[2:]].stack()

print(rain_data_time.shape)
rain_data_time.to_csv("rain_data_time.csv", encoding="UTF-8")

#水位観測所
##時間軸の追加
in_water = water_data[water_data['station']== in_data].set_index('date')
water_data_time = in_water[in_water.columns[2:]].stack()

print(water_data_time.shape)
water_data_time.to_csv("water_data_time.csv", encoding="UTF-8")

#2次元配列
A = np.array([rain_data_time, water_data_time])
C = A.T

#データフレームへ変換
water_rain = pd.DataFrame(C)
water_rain.columns = ["rain_level","water_level"]

#一旦、数値へ変換（測定していないデータは考慮せず）1
water_rain= water_rain.replace('M', '0')
water_rain= water_rain.replace('*', '0')
water_rain= water_rain.replace('-', '0')


water_rain["rain_level"] = pd.to_numeric(water_rain["rain_level"], errors='coerce')
water_rain["water_level"] = pd.to_numeric(water_rain["water_level"], errors='coerce')


print(water_rain)

#日付データの変換
#rain_data = pd.read_csv('rain_data_time.csv')

#データフレームへ変換
date_chg= pd.DataFrame(rain_data_time)

date_chg = date_chg.reset_index()
#date_chg.columns = ["date"]

print(date_chg)

# # #1次元配列(numpy)
rain_data_cos = np.cos(2 * np.pi * (date_chg['date']+1)/365)
rain_data_sin = np.sin(2 * np.pi * (date_chg['date']+1)/365)

# #2次元配列
A = np.array([rain_data_cos, rain_data_sin])
C = A.T
print(rain_data_sin.shape)
rain_data_d_trigo = pd.DataFrame(C)
rain_data_d_trigo.columns = ["D_COS","D_SIN"]
print(rain_data_d_trigo)


#日付（三角関数）と水量・雨量の3列を結合
train = pd.concat([rain_data_d_trigo, water_rain], axis=1, sort =True)

#特徴量を選定
features = ["D_COS","D_SIN","rain_level"]

#学習データとテストデータ カラムの分割
train_X = train[features]
train_y = train["water_level"]

#学習データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, random_state=0)

#lightGBM　採用
model = lgb.LGBMRegressor(boosting_type='goss', max_depth=5, random_state=0)

#'goss' Gradient-based One-Side Sampling 
#勾配(残差)の小さいデータはランダムサンプリングすることによって学習データを減らし高速化を実現

eval_set = [(x_test, y_test)]
callbacks = []
callbacks.append(lgb.early_stopping(stopping_rounds=20))
callbacks.append(lgb.log_evaluation())
model.fit(x_train, y_train, eval_set = eval_set, callbacks = callbacks)


#提出用のフォルダに保存
path2 = 'sample_submit/sample_submit/model/'
filename = (path2+'finalized_model.sav')				
pickle.dump(model, open(filename, 'wb'))					

#学習モデルの予測
y_pred = model.predict(x_test)

print("学習モデル")
print(x_test.head())
print(y_pred)

# RMSE関数
def RMSE(var1, var2):
    
    # MSEを計算
    mse = mean_squared_error(var1,var2)
    
    # 平方根を取った値を返す
    return np.sqrt(mse)

#RMSE
var = RMSE(y_test, y_pred)

#学習モデルのRMSE
print("学習モデルのRMSE")
print(var)

############一旦　プログラムは282行で終了させ、広島県防災WEB雨量・水位観測所のデータをダウンロード============
#当日データの読み込み
rain_data_today = pd.read_csv(path+'rainfall/today_data.csv', encoding="UTF-8")
water_data_today = pd.read_csv(path+'waterlevel/today_data.csv', encoding="UTF-8")

print(rain_data_today)

##雨量時間軸の追加
rain_data_today= rain_data_today.set_index('date')


rain_data_today = rain_data_today[rain_data_today.columns[2:]].stack()

rain_data_today = pd.DataFrame(rain_data_today)

rain_data_today.columns = ["rainlevel"]

print(rain_data_today)

##水位時間軸の追加
water_data_today= water_data_today.set_index('date')


water_data_today = water_data_today[water_data_today.columns[2:]].stack()

water_data_today = pd.DataFrame(water_data_today)

water_data_today.columns = ["waterlevel"]

print(water_data_today)

##緊急事態の宣言の可否

# 最大水位
water_data_today_max = water_data_today["waterlevel"].max()
print("最大1時間水位")
print(water_data_today_max)
# 最小水位
water_data_today_min = water_data_today["waterlevel"].min()
print("最大1時水位")
print(water_data_today_min)

#比率
ratio = water_data_today_max / water_data_today_min

#緊急事態の数値
# 最大雨量
rain_data_today_max = rain_data_today["rainlevel"].max()
print("最大1時間雨量")
print(rain_data_today_max)

# # 合計雨量
# rain_data_today_sum = rain_data_today["rainlevel"].sum()
# print("24時間合計雨量")
# print(rain_data_today_sum)

print("3時間合計雨量")
rain_data_today_TOP3  = rain_data_today.sort_values(by='rainlevel', ascending=False).head(3)
rain_data_today_TOP3 =  rain_data_today_TOP3["rainlevel"].sum()
print(rain_data_today_TOP3)

if  ratio > 1.0 and rain_data_today_max > 26  and  rain_data_today_TOP3 > 40 :
    
    print("!!!! 緊急事態の可能性があります。予測を停止します!!!!")
    print("3時間雨量")
    print(rain_data_today_TOP3)
    
    exit()
    pass

else:
    pass


#一旦、数値へ変換（測定していないデータは考慮せず）
rain_data_today["rainlevel"]= rain_data_today["rainlevel"].replace('M', '0')
rain_data_today["rainlevel"]= rain_data_today["rainlevel"].replace('*', '0')
rain_data_today["rainlevel"]= rain_data_today["rainlevel"].replace('-', '0')


rain_data_today["rainlevel"] = pd.to_numeric(rain_data_today["rainlevel"], errors='coerce')

#データフレームへ変換
rain_date_chg= pd.DataFrame(rain_data_today)

rain_date_chg_to = rain_date_chg.reset_index()
#date_chg.columns = ["date"]

print(rain_date_chg_to)


print(rain_date_chg_to.shape)

# # #1次元配列(numpy)
rain_data_cos = np.cos(2 * np.pi * (rain_date_chg_to['date']+1)/365)
rain_data_sin = np.sin(2 * np.pi * (rain_date_chg_to['date']+1)/365)

# #2次元配列
A = np.array([rain_data_cos, rain_data_sin])
C = A.T

rain_data_d_trigo_to = pd.DataFrame(C)
rain_data_d_trigo_to.columns = ["D_COS","D_SIN"]
print(rain_data_d_trigo_to)

#インデックス
rain_data_today= rain_data_today.reset_index()

print(rain_data_today["rainlevel"])

#日付（三角関数と水位・雨量の3列を結合
x_test2 = pd.concat([rain_data_d_trigo_to, rain_data_today["rainlevel"]], axis=1, sort =True)

print(x_test2)

y_pred2 = model.predict(x_test2)


print(y_pred2)

#翌日の水位を読み込む
water_data_tomo = pd.read_csv(path+'waterlevel/tomo_data.csv')

water_data_tomo= water_data_tomo[water_data_tomo.columns[3:]].stack()


water_data_tomo= pd.DataFrame(water_data_tomo)

water_data_tomo.columns = ["waterlevel"]

#一旦、数値へ変換（測定していないデータは考慮せず）
water_data_tomo["waterlevel"]= water_data_tomo["waterlevel"].replace('M', '0')
water_data_tomo["waterlevel"]= water_data_tomo["waterlevel"].replace('*', '0')
water_data_tomo["waterlevel"]= water_data_tomo["waterlevel"].replace('-', '0')

water_data_tomo["waterlevel"] = pd.to_numeric(water_data_tomo["waterlevel"], errors='coerce')

#1次元配列
y_test2 = np.array([water_data_tomo["waterlevel"]])
y_test2 = y_test2[0]


print(y_test2)
print(y_pred2)

#RMSE
var2 = RMSE(y_test2, y_pred2)
#実際のRMSE
print("検証用RMSE")
print(var2)
