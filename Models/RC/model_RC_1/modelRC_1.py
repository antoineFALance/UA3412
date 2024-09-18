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

def RCfunction(inputs,Hrad,Hv,Cbldg,As):
    time = inputs[:, 0]
    to = inputs[:, 1]
    Qheat = inputs[:, 2]
    Qs=inputs[:,3]

    ti = np.zeros(len(time))
    tret = np.zeros(len(time))

    # Initial temperatures
    ti[0] = T0

    # Loop for calculating all temperatures
    for t in range(1, inputs.shape[0]):
        dt = time[t] - time[t - 1]
        ti[t]=ti[t-1]+dt/Cbldg*(Hrad*(tret[t-1]-ti[t-1])-Hv*(ti[t-1]-to[t-1])+As*Qs[t-1])
        tret[t]=tret[t-1]+dt/Cwater*(Qheat[t-1]-Hrad*(tret[t-1]-ti[t-1]))

    return ti

# def RCfunction2(Ci,Ce,Ri,Re,Hrad,As):
def RCfunction2(q):
    time = inputs[:, 0]
    to = inputs[:, 1]
    Qheat = inputs[:, 2]
    Qs=inputs[:, 3]
#
    ti = np.zeros(len(time))
    tret = np.zeros(len(time))
    te = np.zeros(len(time))

    # Initial temperatures
    ti[0] = df1['Tint_living'].to_list()[0]
    tret[0]=df1['T_output_rad_living'].to_list()[0]
    te[0] = (q[2] * to[0] + q[3] * ti[0]) / (q[2] + q[3])

    # Loop for calculating all temperatures
    for t in range(1, inputs.shape[0]):
        dt = time[t] - time[t - 1]
        te[t]=te[t-1]+dt/q[1]*((ti[t-1]-te[t-1])/q[2]+(to[t-1]-te[t-1])/q[3])
        ti[t]=ti[t-1]+dt/q[0]*((te[t-1]-ti[t-1])/q[2]+q[4]*(tret[t-1]-ti[t-1])+q[5]*Qs[t-1])
        tret[t] = tret[t - 1] + dt / Cwater * (Qheat[t - 1] - q[5] * (tret[t - 1] - ti[t - 1]))

    ti[np.isnan(ti)]=0
    ti[np.isinf(ti)] = 0
    return mean_squared_error(ti,df1['Tint_living'].to_numpy())


def RCfunction3(q):
    time = inputs[:, 0]
    to = inputs[:, 1]
    Qheat = inputs[:, 2]
    Qs=inputs[:, 3]
#
    ti = np.zeros(len(time))
    tret = np.zeros(len(time))
    te = np.zeros(len(time))

    # Initial temperatures
    ti[0] = df1['Tint_living'].to_list()[0]
    tret[0]=df1['T_output_rad_living'].to_list()[0]
    te[0] = (q[2] * to[0] + q[3] * ti[0]) / (q[2] + q[3])

    # Loop for calculating all temperatures
    for t in range(1, inputs.shape[0]):
        dt = time[t] - time[t - 1]
        te[t]=te[t-1]+dt/q[1]*((ti[t-1]-te[t-1])/q[2]+(to[t-1]-te[t-1])/q[3])
        ti[t]=ti[t-1]+dt/q[0]*((te[t-1]-ti[t-1])/q[2]+q[4]*(tret[t-1]-ti[t-1])+q[5]*Qs[t-1])
        tret[t] = tret[t - 1] + dt / Cwater * (Qheat[t - 1] - q[5] * (tret[t - 1] - ti[t - 1]))
    return ti

inputs=df1[['time','Text','gas_value','rad_soleil']].to_numpy()
outputs=df1['Tint_living'].to_numpy()

# test= so.curve_fit(f=RCfunction2,
#                          xdata=inputs,
#                          ydata=outputs,
#                          # p0=[0.2,0.2,8,10,2000,10],
#                          bounds=([1,1,1,1,1,1],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]),
#                          maxfev=100000,
#                         full_output=True
#                          )
bounds = ([0,0.8e02], [0,1e02], [0,0.8e02],[ 0,0.8e02], [0,1e00], [0,1e00])
# result= de(RCfunction2,bounds,args=())
# print(result)

print('ok')
df1=df[(df['year']==2018)&(df['month']==1) & (df['day']==21)]
df1['gas_value'].fillna(0,inplace=True)
df1['time']=np.array(range(df1.shape[0]))

inputs=df1[['time','Text','gas_value','rad_soleil']].to_numpy()

# df2['Tint_hat']=RCfunction(inputs,2.61594e-01,9.351127e-01,2.0e+03,7)
# figure,axs =plt.subplots(5)
# axs[0].plot(df2['gas_value'])
# axs[1].plot(df2['T_output_rad_living'])
# axs[2].plot(df2['Tint_hat'],c='red')
# axs[2].plot(df2['Tint_living'],c='black')
# axs[3].plot(df2['rad_soleil'])
# axs[4].plot(df2['Text'])
# plt.show()
# print('ok')

# df1['Tint_hat']=RCfunction3(result.x)
df1['Tint_hat']=RCfunction3([ 3.078e+01,  9.702e+01,  1.413e+01,  7.950e+01,  2.878e-03,1.757e-01])
figure,axs =plt.subplots(5)
axs[0].plot(df1['gas_value'])
axs[1].plot(df1['T_output_rad_living'])
axs[2].plot(df1['Tint_hat'],c='red')
axs[2].plot(df1['Tint_living'],c='black')
axs[3].plot(df1['rad_soleil'])
axs[4].plot(df1['Text'])
plt.show()
print('ok')