import math

import pandas as pd
import numpy as np
import scipy.optimize as so
from matplotlib import pyplot as plt
import pyswarm
from scipy.optimize import differential_evolution as de
from sklearn.metrics import mean_squared_error

df= pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_Min.csv',sep=";",decimal=",")
df['rad_soleil']=df.apply(lambda x: max(0,-2.0*math.cos(math.pi/12*x.hour)),axis=1)
df['rad_soleil']=np.where(df['min']==0,df['rad_soleil'],np.nan)
df['rad_soleil'].interpolate(inplace=True)
# df1=df[(df['year']==2018)&(df['month']==2) & (df['day']==4) &((df['hour']>=2) & (df['hour']<=6))]
df1=df[(df['year']==2018)&(df['month']==2) & (df['day']==4)]
df1['gas_value'].fillna(0,inplace=True)
df1['time']=np.array(range(df1.shape[0]))
T0=df1['Tint_living'].to_list()[0]
Cwater=4.18

def errorFunction(pred,real):
    return mean_squared_error(real, pred)

# def RCfunction2(R1,R2,R3,C1,C2,Hrad,As):
def RCfunction2(q):
    R1=q[0]
    R2=q[1]
    R3=q[2]
    C1=q[3]
    C2=q[4]
    Hrad=q[5]
    As=q[6]

    time = inputs[:, 0]
    to = inputs[:, 1]
    Qheat = inputs[:, 2]
    Qs=inputs[:, 3]
#
    ti = np.zeros(len(time))
    tret = np.zeros(len(time))
    tc1 = np.zeros(len(time))
    tc2 = np.zeros(len(time))

    # Initial temperatures
    ti[0] = df1['Tint_living'].to_list()[0]
    tret[0]=df1['T_output_rad_living'].to_list()[0]

    # Loop for calculating all temperatures
    for t in range(1, inputs.shape[0]):
        dt = time[t] - time[t - 1]
        tc1[t]=tc1[t-1]+dt*(to[t-1]/(R1*C1)-tc1[t-1]/(R1*C1)-tc1[t-1]/(R2*C1)+tc2[t-1]/(R2*C1))
        tc2[t]=tc2[t-1]+dt*(tc1[t-1]/(R2*C2)-tc2[t-1]/(R2*C2)-tc2[t-1]/(R3*C2)+ti[t-1]/(R3*C2)+1/C2*(Hrad*(tret[t-1]-ti[t-1])+Qs[t-1]*As))
        tret[t]=tret[t-1]+dt/Cwater*(Qheat[t-1]-Hrad*(tret[t-1]-ti[t-1]))
        ti[t]=1/(1/R3-Hrad)*(tc1[t]*(1/R1-1/R2)+tc2[t]*(1/R3-1/R2)-to[t]/R1-tret[t]*Hrad-Qs[t]*As)

    ti[np.isnan(ti)]=0
    ti[np.isinf(ti)] = 0

    # return tret
    return mean_squared_error(ti,df1['Tint_living'].to_numpy())


def RCfunction3(q):
    R1 = q[0]
    R2 = q[1]
    R3 = q[2]
    C1 = q[3]
    C2 = q[4]
    Hrad = q[5]
    As = q[6]

    time = inputs[:, 0]
    to = inputs[:, 1]
    Qheat = inputs[:, 2]
    Qs = inputs[:, 3]
    ti = inputs[:, 4]
    #
    ti_ = np.zeros(len(time))
    tret = np.zeros(len(time))
    tc1 = np.zeros(len(time))
    tc2 = np.zeros(len(time))

    # Initial temperatures
    ti_[0] = df1['Tint_living'].to_list()[0]
    tret[0] = df1['T_output_rad_living'].to_list()[0]

    # Loop for calculating all temperatures
    for t in range(1, inputs.shape[0]):
        dt = time[t] - time[t - 1]
        tc1[t] = tc1[t - 1] + dt * (
                    to[t - 1] / (R1 * C1) - tc1[t - 1] / (R1 * C1) - tc1[t - 1] / (R2 * C1) + tc2[t - 1] / (R2 * C1))
        tc2[t] = tc2[t - 1] + dt * (
                    tc1[t - 1] / (R2 * C2) - tc2[t - 1] / (R2 * C2) - tc2[t - 1] / (R3 * C2) + 1 / C2 * (
                        Hrad * (tret[t - 1] - ti[t - 1]) + Qs[t - 1] * As))
        tret[t] = tret[t - 1] + dt / Cwater * (Qheat[t - 1] - Hrad * (tret[t - 1] - ti[t - 1]))
        ti[t] = 1 / (1 / R3 - Hrad) * (
                    tc1[t] * (1 / R1 - 1 / R2) + tc2[t] * (1 / R3 - 1 / R2) - to[t] / R1 - tret[t] * Hrad - Qs[t] * As)

    ti[np.isnan(ti)] = 0
    ti[np.isinf(ti)] = 0
    return ti

inputs=df1[['time','Text','gas_value','rad_soleil','Tint_living']].to_numpy()
outputs=df1['Tint_living'].to_numpy()

bounds = ([0.01,1e04], [0.01,1e04], [0.01,1e04],[ 0.01,1e04], [0.01,1e04], [0.01,1e04],[0.01,1e04])
result= de(RCfunction2,bounds,args=())
print(result)

print('ok')
df1=df[(df['year']==2018)&(df['month']==1) & (df['day']==4)]
df1['gas_value'].fillna(0,inplace=True)
df1['time']=np.array(range(df1.shape[0]))

inputs=df1[['time','Text','gas_value','rad_soleil','Tint_living']].to_numpy()

df1['T_int_hat']=RCfunction3(result.x)
# df1['Tint_hat']=RCfunction3([ 3.078e+01,  9.702e+01,  1.413e+01,  7.950e+01,  2.878e-03,1.757e-01])
figure,axs =plt.subplots(5)
axs[0].plot(df1['gas_value'])
axs[1].plot(df1['T_output_rad_living'],c='black')
axs[2].plot(df1['T_int_hat'],c='red')
axs[2].plot(df1['Tint_living'],c='black')
axs[3].plot(df1['rad_soleil'])
axs[4].plot(df1['Text'])
plt.show()
print('ok')