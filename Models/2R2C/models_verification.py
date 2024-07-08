import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from itertools import chain
import sys
from sklearn.metrics import mean_squared_error
is_windows = hasattr(sys, 'getwindowsversion')
import random

def simpleRC(T_o_t_1,T_i_t_1,phi_h_t_1, Ri, Ro, Ci):
    ti = T_i_t_1+1/Ci*(1/(Ri+Ro)*(T_o_t_1-T_i_t_1)+phi_h_t_1)
    return ti

def windowing(dataset,windowRange):
    indexRange=list(dataset.index)
    valid_index_range=[]
    discontinuousIndex =[a+1!=b for a, b in zip(indexRange, indexRange[1:])]
    discontinuousIndex2=[index  for index in range(len(discontinuousIndex)) if discontinuousIndex[index]==True]
    if discontinuousIndex2:
        continuousData=[dataset.to_numpy()[0:discontinuousIndex2[0]+1,:]]+[dataset.to_numpy()[discontinuousIndex2[idx]+1:discontinuousIndex2[idx+1]+1,:] for idx in range(len(discontinuousIndex2)-1)]
    else:
        continuousData=[dataset.to_numpy()]
    x_ds,y_ds=[],[]
    for continuousArr in continuousData:
        if continuousArr.shape[0]>=windowRange:
            for step in range(continuousArr.shape[0]-windowRange):
                ds=continuousArr[0+step:windowRange+step,:]
                test = np.hstack([ds[:-1,:-1].flatten(),ds[-2,-1]])
                x_ds.append(np.hstack([ds[:-1,:-1].flatten(),ds[-2,-1]]))
                # x_ds.append(ds[:-1, :-1].flatten())
                y_ds.append(ds[-1,-1])
    x=np.stack(x_ds)
    y=np.stack(y_ds).reshape(-1, 1)
    return x,y

if is_windows:
    PATH_TO_INPUTS_DIR=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\main_dataset\\"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\2R2C\\results\\"
    PATH_TO_MODEL_PARAMETER = os.path.dirname(os.path.dirname(os.getcwd())) + "\\data_\\2R2C_model\\2R2C_model.csv"
else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/main_dataset/"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/2R2C/results/"
    PATH_TO_MODEL_PARAMETER = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/2R2C_model/2R2C_model.csv"

df_results=pd.DataFrame()


directory = os.fsencode(PATH_TO_INPUTS_DIR)
df_error=pd.DataFrame()
parameters=pd.read_csv(PATH_TO_MODEL_PARAMETER,sep=";")
mseList,rmseList,home_id_list=[],[],[]

df_model_results=pd.DataFrame()
mse_list=[]
for file in os.listdir(directory):
    print(file)
    filename = os.fsdecode(file)
    fullFileName = PATH_TO_INPUTS_DIR + filename
    home_id = re.search('weather_(.*).csv', filename).group(1)
    home_id_list.append(home_id)
    mask=parameters['home_id']==home_id
    param_home_id=parameters[mask]
    ri=param_home_id['ri'].to_list()[0]
    ro = param_home_id['ro'].to_list()[0]
    ci = param_home_id['ci'].to_list()[0]
    ce = param_home_id['ce'].to_list()[0]

    df_dataset = pd.read_csv(fullFileName, sep=";")
    df_dataset['gas_value'].fillna(df_dataset['gas_value'].min(), inplace=True)
    x, y = windowing(df_dataset[['gas_value','Text','Tint']], windowRange=2)
    dataset = pd.DataFrame(x,columns=['gas_value_1','Text_1','Tint_1'])
    dataset['Text_1'].interpolate(inplace=True)
    ti_list,t_env_list=[],[]
    for index,row in dataset.iterrows():
        if index==0:
            T_env_1=(ri*row['Text_1']+ro*row['Tint_1'])/(ri+ro)
            t_env_list.append(T_env_1)
            ti=row['Tint_1']+1/ci*((T_env_1-row['Tint_1'])/ri+row['gas_value_1'])
            ti_list.append(ti)
        else:
            T_env_1=t_env_list[-1]+1/ce*((row['Tint_1']-t_env_list[-1])/ri+(row['Text_1']-t_env_list[-1])/ro)
            t_env_list.append(T_env_1)
            ti=row['Tint_1']+1/ci*((T_env_1-row['Tint_1'])/ri+row['gas_value_1'])
            ti_list.append(ti)
    dataset['ti_pred']=np.array(ti_list)
    dataset['ti_reel']=y
    dataset['t_env']=np.array(t_env_list)
    dataset['t']=dataset.index
    mse=mean_squared_error(y_true=y,y_pred=ti_list)
    mse_list.append(mse)
    index=random.randint(0,len(mse_list))

    ax=dataset.iloc[index:index+300].plot(x='t',y='ti_pred')
    ax = dataset.iloc[index:index+300].plot(x='t', y='ti_reel',ax=ax)
    # plt.show()
    fig = ax.get_figure()
    FILENAME = home_id+"_"+"2R2C"
    fig.savefig(PATH_TO_OUTPUT_RESULTS + FILENAME + "_result_plot.png")
    plt.close()

df_model_results['home_id']=np.array(home_id_list)
df_model_results['mse']=np.array(mse_list)
df_model_results.to_csv(PATH_TO_OUTPUT_RESULTS+"2R2C_mse.csv",sep=";")
print('ok')










