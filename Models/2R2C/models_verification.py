import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from itertools import chain
import sys
is_windows = hasattr(sys, 'getwindowsversion')

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
    PATH_TO_OUTPUT_MODEL=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\XGBoost\\models\\"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\XGBoost\\results\\"
    PATH_TO_MODEL_PARAMETER = os.path.dirname(os.path.dirname(os.getcwd())) + "\\data_\\2R2C_model\\2R2C_model.csv"
else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/main_dataset/"
    PATH_TO_OUTPUT_MODEL = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/XGBoost/models/"
    PATH_TO_OUTPUT_RESULTS = os.path.dirname(os.path.dirname(os.getcwd())) + "/Models/XGBoost/results/"
    PATH_TO_MODEL_PARAMETER = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/2R2C_model/2R2C_model.csv"

df_results=pd.DataFrame()


directory = os.fsencode(PATH_TO_INPUTS_DIR)
df_error=pd.DataFrame()
parameters=pd.read_csv(PATH_TO_MODEL_PARAMETER,sep=";")
mseList,rmseList,home_id_list=[],[],[]

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
    dataset['Tint']=y
    dataset['Tint_pred']=dataset.apply(lambda x:simpleRC(x.Text_1,x.Tint_1,x.gas_value_1,ri,ro,ci),axis=1)
    print('ok')






    resultList=[]
    for section in dataTocheck['num'].unique():
        mask=dataTocheck['num']==section
        dataTocheckMasked=dataTocheck[mask]
        inputs = dataTocheckMasked[['time', 'Text', 'gas']].to_numpy()
        output = dataTocheckMasked['Tint'].to_numpy()
        ti_results=list(simpleRC(inputs=inputs,output=output,Ri=ri,Ro=ro,Ce=ce,Ci=ci))
        resultList.append(ti_results)
    dataTocheck['result_model']=np.array(chain.from_iterable(resultList))
    dataTocheck['x_plot']=np.array(range(dataTocheck.shape[0]))
    dataTocheck['MSE']=dataTocheck.apply(lambda x:(x['Tint']-x['result_model'])**2,axis=1)
    mseList.append(dataTocheck['MSE'].mean())
    rmseList.append(sqrt(dataTocheck['MSE'].mean()))
    ax1 = dataTocheck.plot(x='x_plot',y=['Tint'],color='r')
    ax2 = dataTocheck.plot(x='x_plot',y=['result_model'],ax=ax1,color='green')
    fig = ax2.get_figure()
    result_png_name = PATH_TO_OUTPUT_DIR_DATA + home_id + "_result_model.png"
    fig.savefig(result_png_name)
    plt.close()
    print(home_id)
df_error['home_id'] = np.array(home_id_list)
df_error['mse']=np.array(mseList)
df_error['rmse']=np.array(rmseList)
df_error.to_csv(PATH_TO_OUTPUT_DIR_DATA+'error_rate.csv',sep=";",index=False)



