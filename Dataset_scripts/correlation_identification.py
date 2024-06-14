import pandas as pd
import numpy as np
import math
import os

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

PATH_TO_INPUT_DIR_DATA="######"
PATH_TO_OUTPUT_DIR_DATA="######"
directory = os.fsencode(PATH_TO_INPUT_DIR_DATA)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    fullFileName=PATH_TO_INPUT_DIR_DATA+filename
    # PATH_TO_INPUT_DATA= "/data/weather_home62.csv"
    input_=pd.read_csv(fullFileName,sep=";")
    input_['datetime']=pd.to_datetime(input_[['year','month','day','hour']])
    input_.sort_values(by=['datetime'],inplace=True)
    input_['t']=range(len(input_['datetime']))

    # Essayer de créer un nuage de points depuis les équations pluitôt qu'un extrait
    ref_signal=input_[(input_['datetime']>='2017-02-05 7:00:00') &(input_['datetime']<='2017-02-05 17:00:00')]['temp_value'].to_numpy()

    correlation_list,idxList=[],[]
    for idx in range(input_['t'].shape[0]):
        window_signal=input_['temp_value'].to_numpy()[idx:idx+11]
        correlation_list.append(cross_correlation(window_signal,ref_signal))
        idxList.append([idx,idx+11])

    bestIndex=[idxList[index] for index in range(len(idxList)) if correlation_list[index]>0.95]
    output=[input_['temp_value'].to_numpy()[window[0]:window[1]] for window in bestIndex]
    inputs=[input_[['t','temp','gas_value']].to_numpy()[window[0]:window[1],:] for window in bestIndex]
    data=[input_[['t','temp','gas_value','temp_value']].to_numpy()[window[0]:window[1],:] for window in bestIndex]
    dfInput=pd.DataFrame()

    enum=0
    for inp in data:
        dfTemp=pd.DataFrame(data=inp)
        dfTemp['num']=enum
        test = [dfTemp[col].isnull().all() for col in dfTemp.columns]
        if all(test!=np.nan):
            dfInput=pd.concat([dfInput,dfTemp],axis=0)
            enum+=1

    dfInput.to_csv(PATH_TO_OUTPUT_DIR_DATA+"#####.csv")

print('ok')