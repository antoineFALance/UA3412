import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as so
from  scipy.signal import correlate
import math


# Define a function to calculate mean
def mean(arr):
    return sum(arr) / len(arr)


# function to calculate cross-correlation
def cross_correlation(x, y):
    # Calculate means
    x_mean = mean(x)
    y_mean = mean(y)

    # Calculate numerator
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))

    # Calculate denominators
    x_sq_diff = sum((a - x_mean) ** 2 for a in x)
    y_sq_diff = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_sq_diff * y_sq_diff)
    correlation = numerator / denominator

    return correlation

PATH_TO_INPUT_DATA="/home/antoine/PycharmProjects/UA3412/data/weather_home62.csv"
input_=pd.read_csv(PATH_TO_INPUT_DATA,sep=";")
input_['datetime']=pd.to_datetime(input_[['year','month','day','hour']])
input_.sort_values(by=['datetime'],inplace=True)
input_['t']=range(len(input_['datetime']))
# DEBUG
# ax0=input_.plot(x='datetime',y='gas_value')
# plt.show()

ref_signal=input_[(input_['datetime']>='2017-02-05 7:00:00') &(input_['datetime']<='2017-02-05 17:00:00')]['temp_value'].to_numpy()

correlation_list,idxList=[],[]
for idx in range(input_['t'].shape[0]):
    window_signal=input_['temp_value'].to_numpy()[idx:idx+11]
    correlation_list.append(cross_correlation(window_signal,ref_signal))
    idxList.append([idx,idx+11])

bestIndex=[idxList[index] for index in range(len(idxList)) if correlation_list[index]>0.95]
output=[input_['temp_value'].to_numpy()[window[0]:window[1]] for window in bestIndex]
inputs=[input_[['t','temp','gas_value']].to_numpy()[window[0]:window[1],:] for window in bestIndex]
dfInput=pd.DataFrame()
for inp in inputs:
    dfTemp=pd.DataFrame(data=inp)
    dfInput=pd.concat([dfInput,dfTemp],axis=0)
print('ok')