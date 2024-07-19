import pandas as pd
import os
import sys
is_windows = hasattr(sys, 'getwindowsversion')
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
# import tensorflow as tf
import numpy as np
import random
import seaborn as sns

# FONCTIONS
def seasonality_plot(serie,period):
    result = seasonal_decompose(serie, model='additive', period=period)
    result.plot()
    plt.show()

# Données météo
if is_windows:
    PATH_TO_WEATHER_FORECAST=os.path.dirname(os.getcwd())+"\\data_\\weather_forecast\\"
    PATH_TO_OUTPUT_DIR_DATA = os.path.dirname(os.getcwd())+"\\Time serie\\"
    PATH_TO_INPUT_HOUSE_DATA=os.path.dirname(os.getcwd())+"\\data_\\dataset_gas_value_corrected\\"
else:
    PATH_TO_WEATHER_FORECAST = os.path.dirname(os.getcwd()) + "/data_/weather_forecast/"
    PATH_TO_OUTPUT_DIR_DATA = os.path.dirname(os.getcwd()) + "/Time serie/"
    PATH_TO_INPUT_HOUSE_DATA = os.path.dirname(os.getcwd()) + "/data_/dataset_gas_value_corrected\\"


# CORRELATIONMATRIX DONNES LOCAL+METEO
varList=['hum_value','central_heat_flow_value','central_heat_return_value','Tint','Ti_1','Text','phi_corrected','phi_1','dewPt','rh','pressure','wind_x','wind_y','gas_value']
fileList=[file for file in os.listdir(PATH_TO_INPUT_HOUSE_DATA)]
df_house=pd.DataFrame()
for file in fileList:
    filename = os.fsdecode(file)
    fullFileName=PATH_TO_INPUT_HOUSE_DATA+'/'+filename
    df_dataset=pd.read_csv(fullFileName,sep=";")
    df_dataset=df_dataset[varList].dropna()
    df_dataset['gas_value_1']=df_dataset['gas_value'].shift(1)
    corr = df_dataset.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                annot=True)
    plt.show()

    print('ok')

