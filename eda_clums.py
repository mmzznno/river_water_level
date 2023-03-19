import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import japanize_matplotlib

path = 'run_test/train/'

# 水位のデータの読み込み
water_data = pd.read_csv(path+'waterlevel/data.csv')
water_stations = pd.read_csv(path+'waterlevel/stations.csv')

# 潮位のデータの読み込み
tide_data = pd.read_csv(path+'tidelevel/data.csv')


#東西の判定
def ew_juge(ew) :
    if ew <= 132.775 :
        return "w"
    else:
        return "e"
    
water_stations.loc[:,"東西"]  = water_stations.loc[:,"経度"].apply(ew_juge)

#南北の判定
def sn_juge(sn) :
    if sn <= 34.5362 :
        return "n"
    else:
        return "s"
    
water_stations.loc[:,"南北"]  = water_stations.loc[:,"緯度"].apply(sn_juge)

#water_stations.to_csv("water_stations_a.csv", encoding="UTF-8")

# 雨量のデータの読み込み
rain_data = pd.read_csv(path+'rainfall/data.csv')
rain_stations = pd.read_csv(path+'rainfall/stations.csv')


#東西の判定
rain_stations.loc[:,"東西"]  = rain_stations.loc[:,"経度"].apply(ew_juge)
#南北の判定
rain_stations.loc[:,"南北"]  = rain_stations.loc[:,"緯度"].apply(sn_juge)

#確認用CSV出力
#rain_stations.to_csv("rain_stations_a.csv", encoding="UTF-8")

# #1年目のデータを探索
# rain_data = rain_data[0:150060]
# water_data = water_data[0:65514]

# #2年目のデータを探索
rain_data = rain_data[150061:300122]
water_data = water_data[65515:131030]
tide_data =tide_data[4758:9502]

# print('水位観測データ: ', water_data.shape)
# print('水位観測データ所: ', water_stations.shape)

# print('雨量観測データ: ', rain_data.shape)
# print('雨量観測データ所: ', rain_stations.shape)


#水位観測所のを入力

#例
in_data = "草津(国)"

water_stations_select = water_stations[water_stations["観測所名称"] == in_data]

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

#print("最東")

print(rain_stations_select3)
rain_stations_select4 = rain_stations_select3.loc[[rain_stations_select3["経度"].idxmax()]]

print(rain_stations_select4)

rain_stations_name   = rain_stations_select4[["観測所名称"]]
rain_stations_name_np   = rain_stations_name.to_numpy().tolist()
print(rain_stations_name_np)

rain_stations_name =  rain_stations_name_np[0][0]

print("雨量観測所名称")
print(rain_stations_name)

rain_data_W = rain_data[rain_data["station"] == rain_stations_name ]
water_data = water_data[water_data["station"] == in_data]

# #時間軸の追加 (グラフを出力時はこのコード不要)
# rain_data = rain_data[rain_data['station']== rain_stations_name].set_index('date')
# rain_data_time= rain_data[rain_data.columns[2:]].stack()

# print(rain_data_time.shape)
# rain_data_time.to_csv("rain_data_time.csv", encoding="UTF-8")

#雨量観測所の比較

rain_data[pd.to_numeric(rain_data['00:00:00'], errors='coerce').notna()==False]['00:00:00'].unique()

for i in range(24):
     rain_data[f'{str(i).zfill(2)}:00:00'] = pd.to_numeric(rain_data[f'{str(i).zfill(2)}:00:00'], errors='coerce')
    
#1日あたりの平均)
rain_data['mean'] = rain_data[rain_data.columns[3:]].mean(axis=1)
rain_data_in1 = rain_data[rain_data['station']== rain_stations_name ][['date', 'mean']].set_index('date')

#元コード（1日あたりの平均)

# rain_data_time.to_csv("rain_data_time.csv", encoding="UTF-8")

plt.figure(figsize=(10, 5))
plt.plot(rain_data_in1 , label= rain_stations_name )
plt.title('all day')
plt.xlabel('date')
plt.ylabel('rain level')
plt.legend()
plt.grid()
plt.show()

# #水位観測所

water_data[pd.to_numeric(water_data['00:00:00'], errors='coerce').notna()==False]['00:00:00'].unique()

for i in range(24):
     
     water_data[f'{str(i).zfill(2)}:00:00'] = pd.to_numeric(water_data[f'{str(i).zfill(2)}:00:00'], errors='coerce')
     
     print(water_data)


# ##元コード(1日あたりの平均)
water_data['mean'] = water_data[water_data.columns[3:]].mean(axis=1)
water_data_in = water_data[water_data['station']== in_data][['date', 'mean']].set_index('date')
# in_water = water_data[water_data['station']== in_data].set_index('date')
##元コード

# # ####時間軸の追加1日あたりの平均)
# in_water = water_data[water_data['station']== in_data].set_index('date')
# water_data_time = in_water[in_water.columns[2:]].stack()

# # ####

# print(water_data_time.shape)
# water_data_time.to_csv("water_data_time.csv", encoding="UTF-8")

plt.figure(figsize=(10, 5))
plt.plot(water_data_in, label= in_data)
plt.title('all day')
plt.xlabel('date')
plt.ylabel('water level')
plt.legend()
plt.grid()
plt.show()



# 潮位観測所

tide_data[pd.to_numeric(tide_data['00:00:00'], errors='coerce').notna()==False]['00:00:00'].unique()

for i in range(24):
     
     tide_data[f'{str(i).zfill(2)}:00:00'] = pd.to_numeric(tide_data[f'{str(i).zfill(2)}:00:00'], errors='coerce')
     
     print(tide_data)


# ##元コード(1日あたりの平均)
tide_data['mean'] = tide_data[tide_data.columns[3:]].mean(axis=1)
tide_data_in = tide_data[tide_data['station']== "広島港"][['date', 'mean']].set_index('date')
# in_water = water_data[water_data['station']== in_data].set_index('date')
##元コード

# # ####時間軸の追加1日あたりの平均)
# in_water = water_data[water_data['station']== in_data].set_index('date')
# water_data_time = in_water[in_water.columns[2:]].stack()

# # ####

# print(water_data_time.shape)
# water_data_time.to_csv("water_data_time.csv", encoding="UTF-8")

plt.figure(figsize=(10, 5))
plt.plot(tide_data_in, label= "広島港")
plt.plot(water_data_in, label= in_data)
plt.title('all day')
plt.xlabel('date')
plt.ylabel('tide-water level')
plt.legend()
plt.grid()
plt.show()







