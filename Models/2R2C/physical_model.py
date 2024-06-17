import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as so
from statistics import mean
import sys
is_windows = hasattr(sys, 'getwindowsversion')

def simpleRC(inputs, Ri, Ro, Ci, Ce):
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
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\2R2C_model\\"
else:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"/data_/signal_dataset/"
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.path.dirname(os.getcwd()))+"/data_/2R2C_model/"
directory = os.fsencode(PATH_TO_INPUT_DIR_DATA)

df_res=pd.DataFrame()
for file in os.listdir(directory):
    print(file)
    filename = os.fsdecode(file)
    home_id = re.search('(.*)heat_signal.csv', filename).group(1)
    fullFileName=PATH_TO_INPUT_DIR_DATA+filename
    heat_sequence=pd.read_csv(fullFileName,sep=";")
    numHeatSeqList=heat_sequence['num'].to_list()
    df_temp=pd.DataFrame()
    riList, roList, ciList, ceList = [], [], [], []
    for numSeq in numHeatSeqList:
        mask=heat_sequence['num']==numSeq
        heat_sequence_masked=heat_sequence[mask]
        inputs = heat_sequence_masked[['time', 'Text', 'gas']].to_numpy()
        output = heat_sequence_masked['Tint'].to_numpy()
        try:
            p_opt, p_cov = so.curve_fit(f=simpleRC,
                                    xdata=inputs,
                                    ydata=output,
                                    p0=(0.01, 0.01, 1e6, 1e7),
                                    bounds=[[0,0,0,0],[1,1,np.inf,np.inf]])
            ri, r0, ci, ce = p_opt[0], p_opt[1], p_opt[2], p_opt[3]
            riList.append(ri)
            roList.append(r0)
            ciList.append(ci)
            ceList.append(ce)
        except:
            pass

    df_temp['home_id']=np.array([home_id])
    df_temp['ri']=np.array([mean(riList)])
    df_temp['ro'] = np.array([mean(roList)])
    df_temp['ci'] = np.array([mean(ciList)])
    df_temp['ce'] = np.array([mean(ceList)])
    df_res=pd.concat((df_res,df_temp))
df_res.to_csv(PATH_TO_OUTPUT_DIR_DATA+'2R2C_model.csv',sep=";",index=False)



