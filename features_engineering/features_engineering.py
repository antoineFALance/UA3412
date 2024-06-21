import pandas as pd
import os
import sys
is_windows = hasattr(sys, 'getwindowsversion')
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random

# FONCTIONS
def seasonality_plot(serie,period):
    result = seasonal_decompose(serie, model='additive', period=period)
    result.plot()
    plt.show()

# Données météo
if is_windows:
    PATH_TO_WEATHER_FORECAST=os.path.dirname(os.getcwd())+"\\data_\\weather_forecast\\"
    PATH_TO_OUTPUT_DIR_DATA = os.path.dirname(os.getcwd())+"\\Time serie\\"
    PATH_TO_INPUT_HOUSE_DATA=os.path.dirname(os.getcwd())+"\\data_\\main_dataset\\"
else:
    PATH_TO_WEATHER_FORECAST = os.path.dirname(os.getcwd()) + "/data_/weather_forecast/"
    PATH_TO_OUTPUT_DIR_DATA = os.path.dirname(os.getcwd()) + "/Time serie/"
    PATH_TO_INPUT_HOUSE_DATA = os.path.dirname(os.getcwd()) + "/data_/main_dataset\\"

df_weather = pd.read_csv(PATH_TO_WEATHER_FORECAST+"transformed_weather_dataset.csv",sep=",")
df_weather['datetime']=pd.to_datetime(df_weather[['year', 'month', 'day','hour']])
df_weather['timestamp']=df_weather['datetime'].values.tolist()

# CORRELATION MATRIX DONNEES METEO
# var=['temp','dewPt','rh','pressure','wind_x','wind_y']
# df=df_weather[var]
# f = plt.figure(figsize=(19, 15))
# print(df.corr())
# plt.matshow(df.corr(), fignum=f.number)
# plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
# plt.show()

# df=df_weather[var]
# pd.plotting.scatter_matrix(df)
# plt.show()

# CORRELATIONMATRIX DONNES LOCAL+METEO
fileList=[file for file in os.listdir(PATH_TO_INPUT_HOUSE_DATA)]
df_house=pd.DataFrame()
for file in fileList:
    filename = os.fsdecode(file)
    fullFileName=PATH_TO_INPUT_HOUSE_DATA+'/'+filename
    df_dataset=pd.read_csv(fullFileName,sep=";")
    df_dataset['gas_value'].fillna(df_dataset['gas_value'].min(),inplace=True)
    df_house=pd.concat([df_house,df_dataset])
print('ok')
pd.set_option('display.max_columns', None)
var=['Tint','hum_value','central_heat_flow_value','central_heat_return_value','Text','dewPt','rh','pressure','wind_x','wind_y','gas_value']
df_house=df_house[var]
df_house['delta']=df_house['central_heat_flow_value']-df_house['central_heat_return_value']
print(df_house.corr())
# pd.plotting.scatter_matrix(df_house)
# plt.show()
