import pandas as pd
import os
import sys
is_windows = hasattr(sys, 'getwindowsversion')
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

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

df_weather = pd.read_csv(PATH_TO_WEATHER_FORECAST+"weather_edinburgh_final.csv",sep=";")
df_weather['datetime']=pd.to_datetime(df_weather[['year', 'month', 'day','hour']])
df_weather['timestamp']=df_weather['datetime'].values.tolist()

plot_cols = ['temp', 'dewPt', 'rh','pressure','wc','wdir','wspd','datetime']
plot_features = df_weather[plot_cols]
plot_features.set_index('datetime',inplace=True)

# for parameters in plot_cols:
#     for period in [24,30*24]:
#         seasonality_plot(plot_features[parameters]-plot_features[parameters].mean(),period)

