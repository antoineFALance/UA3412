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
else:
    PATH_TO_WEATHER_FORECAST = os.path.dirname(os.getcwd()) + "/data_/weather_forecast/"
    PATH_TO_OUTPUT_DIR_DATA = os.path.dirname(os.getcwd()) + "/Time serie/"

df_weather = pd.read_csv(PATH_TO_WEATHER_FORECAST+"transformed_weather_dataset.csv",sep=",")
df_weather['datetime']=pd.to_datetime(df_weather[['year', 'month', 'day','hour']])
df_weather['timestamp']=df_weather['datetime'].values.tolist()

# plot_cols = ['temp', 'dewPt', 'rh','pressure','datetime']
# plot_features = df_weather[plot_cols]
# plot_features.set_index('datetime',inplace=True)

# for parameters in plot_cols:
#     for period in [24,30*24]:
#         seasonality_plot(plot_features[parameters]-plot_features[parameters].mean(),period)

# AUTOCORRELATION
year=random.randint(2016,2018)
month=random.randint(min(df_weather[df_weather['year']==year]['month'].unique()),max(df_weather[df_weather['year']==year]['month'].unique()))
day=random.randint(min(df_weather[(df_weather['year']==year) & (df_weather['month']==month) ]['day'].unique()),max(df_weather[(df_weather['year']==year) & (df_weather['month']==month) ]['day'].unique()))
serie=df_weather[(df_weather['year']==year) & (df_weather['month']==month) & (df_weather['day']==day)]['temp']
# pd.plotting.autocorrelation_plot(serie)
# plt.show()
print('ok')

# CORRELATION MATRIX
var=['temp','dewPt','rh','pressure','wind_x','wind_y']
df=df_weather[var]
f = plt.figure(figsize=(19, 15))
print(df.corr())
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

df=df_weather[var]
pd.plotting.scatter_matrix(df)
plt.show()