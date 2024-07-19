import math
import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as so
from statistics import mean,stdev
import sys
is_windows = hasattr(sys, 'getwindowsversion')
import scipy.stats as stats

def windowing(dataset,windowRange):
    x_list,y_list=[],[]

    for day in dataset['cd_yearMonthDay'].unique().tolist():
        df_day = dataset[dataset['cd_yearMonthDay']==day]
        if not df_day.isnull().any().any() :
            df_day['t']=np.array(range(df_day.shape[0]))
            # ax=df_day.plot(x='t',y='Tint')
            # df_day.plot(x='t', y='Text',ax=ax)
            x_list.append(df_day[['gas_value','Text','Tint']].to_numpy())
            y_list.append(df_day[['Tint']].to_numpy())
        else:
            pass
    return x_list,y_list


    x=np.stack(x_ds)
    y=np.stack(y_ds).reshape(-1, 1)
    wdw=10
    x_,y_=[],[]
    for step in range(x.shape[0] - wdw):
        x_chunk = x[step:step+wdw,:]
        y_chunk = y[step:step+wdw,:]
        x_.append(x_chunk)
        y_.append(y_chunk)

    return np.array(x_),np.array(y_)

def simpleRC(x,R,C):
    ti_1=x[:,0][0]
    te_1=x[:,1][0]
    gas_value_1 = x[:, 2][0]
    delta_phi=x[:, 3][0]
    delta_Te=x[:, 4][0]
    gamma = R*C

    ti = -R*delta_phi*math.exp(-)
    return ti

# PATH DEFINITION
if is_windows:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\dataset_gas_value_corrected\\"
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\RC\\results\\"
else:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"/data_/dataset_gas_value_corrected/"
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.path.dirname(os.getcwd()))+"/Models/RC/results/"
directory = os.fsencode(PATH_TO_INPUT_DIR_DATA)

df_res=pd.DataFrame()
for file in os.listdir(directory):
    print(file)
    filename = os.fsdecode(file)
    home_id = re.search('(.*)_dataset.csv', filename).group(1)
    fullFileName=PATH_TO_INPUT_DIR_DATA+filename
    df_temp=pd.DataFrame()
    RList, CList = [], []
    Ri_List,R0_List,Ci_List,Ce_List=[],[],[],[]
    df_dataset = pd.read_csv(fullFileName, sep=";")
    df_dataset['ti_1']=df_dataset['Tint'].shift(1)
    df_dataset['te_1'] = df_dataset['Text'].shift(1)
    df_dataset['deltaPhi'] = df_dataset['gas_value']-df_dataset['gas_value'].shift(1)
    df_dataset['deltaTe'] = df_dataset['Text'] - df_dataset['Text'].shift(1)
    df_dataset['gas_value_1'] = df_dataset['gas_value'].shift(1)

    x=df_dataset[['ti_1','te_1','gas_value_1','deltaPhi','deltaTe']].dropna().to_numpy()
    y = df_dataset[['Tint']].dropna().to_numpy()[1:].flatten()

    for idx in range(x.shape[0]):
        x_test=x[[idx],:]
        y_test=y[idx]
        p_opt, p_cov = so.curve_fit(f=simpleRC,
                            xdata=x[[idx],:],
                            ydata=y[idx],
                            p0=(1.0, 1.0),
                            bounds=[[0.5,1],[1.5,3]]
                                )


        R, C = p_opt[0], p_opt[1]
        RList.append(R)
        CList.append(C)


    R_mean=mean(RList)
    C_mean=mean(CList)
    R_std = stdev(RList)
    C_std=stdev(CList)

    df_temp['home_id']=np.array([home_id])
    df_temp['Rmean']=np.array(R_mean)
    df_temp['Cmean'] = np.array(C_mean)
    df_temp['Rstd'] = np.array(R_std)
    df_temp['Cstd'] = np.array(C_std)
    df_res=pd.concat((df_res,df_temp))

df_res.to_csv(PATH_TO_OUTPUT_DIR_DATA+'RC_model.csv',sep=";",index=False)



