import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as so

PATH_TO_INPUT_DATA="/home/antoine/PycharmProjects/UA3412/data/weather_home62.csv"
input_=pd.read_csv(PATH_TO_INPUT_DATA,sep=";")
input_['datetime']=pd.to_datetime(input_[['year','month','day','hour']])
input_.sort_values(by=['datetime'],inplace=True)
# DEBUG
# ax0=input_.plot(x='datetime',y='gas_value')
# plt.show()

input=input_[(input_['datetime']>='2017-02-05 06:00:00') &(input_['datetime']<='2017-02-05 16:00:00')]
input_test=input_[(input_['datetime']>='2017-01-01 00:00:00') &(input_['datetime']<='2017-01-21 23:00:00')]
input['t']=range(input.shape[0])
input_test['t']=range(input_test.shape[0])
inputs=input[['t','temp','gas_value']].to_numpy()
inputs_test=input_test[['t','temp','gas_value']].to_numpy()
output=input['temp_value'].to_numpy()
output_test=input_test['temp_value'].to_numpy()

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

def simpleRC2(inputs, Ri, Ro, Ci, Ce,output):
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



p_opt, p_cov = so.curve_fit(f=simpleRC,
                            xdata=inputs,
                            ydata=output,
                            p0=(0.01, 0.01, 1e6, 1e7),
                            bounds=[[0,0,0,0],[np.inf,np.inf,np.inf,np.inf]])

# Saving results into a dataframe and displaying it
res1 = pd.DataFrame(index=['Ri', 'Ro', 'Ci', 'Ce'])
res1['avg'] = p_opt
res1['std'] = np.diag(p_cov)**0.5
print(res1)

ri,r0,ci,ce=p_opt[0],p_opt[1],p_opt[2],p_opt[3]
input_test['ti']=simpleRC2(inputs=inputs_test,Ri=ri,Ro=r0,Ci=ci,Ce=ce,output=output_test)
ax=input_test.plot(x='t',y='temp_value')
ax1=input_test.plot(x='t',y='ti',ax=ax,ls='--')
plt.show()
print('ok')
