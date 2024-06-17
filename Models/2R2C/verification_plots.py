import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
import scipy.optimize as so
from statistics import mean
from itertools import chain
import sys
is_windows = hasattr(sys, 'getwindowsversion')

# FUNCTIONS
def simpleRC(inputs,output, Ri, Ro, Ci, Ce):
    time = range(inputs.shape[0])
    to = inputs[:,1].tolist()
    phi_h = inputs[:,2].tolist()
    ti = np.zeros(len(time))
    te = np.zeros(len(time))
    # Initial temperatures
    ti[0] = output[0]
    te[0] = (Ri * to[0] + Ro * ti[0]) / (Ri + Ro)
    # Loop for calculating all temperatures
    for t in range(1, len(output)):
        dt = (time[t] - time[t - 1])
        ti[t] = ti[t - 1] + dt / Ci * ((te[t - 1] - ti[t - 1]) / Ri + phi_h[t - 1] )
        te[t] = te[t - 1] + dt / Ce * ((ti[t - 1] - te[t - 1]) / Ri + (to[t - 1] - te[t - 1]) / Ro)
    return ti

# PATH DEFINITION
if is_windows:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\signal_dataset\\"
    PATH_TO_MODEL_PARAMETER = os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\2R2C_model\\2R2C_model.csv"
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\2R2C_model\\"
else:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"/data_/signal_dataset/"
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.path.dirname(os.getcwd()))+"/data_/2R2C_model/"
directory = os.fsencode(PATH_TO_INPUT_DIR_DATA)
df_error=pd.DataFrame()
parameters=pd.read_csv(PATH_TO_MODEL_PARAMETER,sep=";")
mseList,rmseList,home_id_list=[],[],[]
for file in os.listdir(directory):
    print(file)
    filename = os.fsdecode(file)
    fullFileName = PATH_TO_INPUT_DIR_DATA + filename
    home_id = re.search('(.*)heat_signal.csv', filename).group(1)
    home_id_list.append(home_id)
    mask=parameters['home_id']==home_id
    param_home_id=parameters[mask]
    ri=param_home_id['ri'].to_list()[0]
    ro = param_home_id['ro'].to_list()[0]
    ci = param_home_id['ci'].to_list()[0]
    ce = param_home_id['ce'].to_list()[0]
    dataTocheck=pd.read_csv(fullFileName,sep=";")
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



