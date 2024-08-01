import math
from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt
import sys
is_windows = hasattr(sys, 'getwindowsversion')
import os
import re
import pandas as pd
import sys
is_windows = hasattr(sys, 'getwindowsversion')
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

if is_windows:
    PATH_TO_INPUTS_DIR=os.path.dirname(os.getcwd())+"\\data_\\main_dataset\\"

else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.getcwd()) + "/data_/main_dataset/"

# DONNEES
dt = 1
time = 0.0
current_value = 0.0
R=1.5e-01
C=2e-02
current_value_list=[0]
gas_value_list=[0]

lastErrorList = [0]
integralList = [0]

Text_list=[]
gas_value_reel=[]

errorList=[]

filename=PATH_TO_INPUTS_DIR+"weather_home106.csv"
df=pd.read_csv(filename,sep=";")
df['yearmonth']=df['year']*100+df['month']

# FILTRE SUR MOIS HIVER
df=df[(df['yearmonth'] ==201711) | (df['yearmonth'] ==201712) | (df['yearmonth'] ==201801)| (df['yearmonth'] ==201802)]
df.fillna(method='ffill', inplace=True)
df.dropna(inplace=True)
# df['setpoint']=np.where((df['hour']>=8) & (df['hour']<=20),24,24)
df['setpoint']=df['Tint']
TextStepList=df['Text'].to_list()
Tint_list=[]
setpointList=df['setpoint'].to_list()
Tint_reel=df['Tint'].to_list()
Tint_reel_list=[]

def PID(Kp,Ki,Kd,integral,lasterror,error,max_rate=200):
    derivative = (error - lasterror) / dt
    integral += error * dt
    output = Kp * error + Ki * integral + Kd * derivative
    return max(output,0),integral

def system(t,Text,phi,T0):
    dTidt=math.exp(-t/(R*C))/C*(1/R*(Text-T0)+phi)
    return dTidt

def system2(t,Ti,phi,text):
    dTidt=1/(R*C)*(text-Ti)+phi
    return dTidt
# nb_days=60

for step in tqdm(range(df.shape[0])):
    text=TextStepList[step]
    setpoint=setpointList[step]
    tint_r=Tint_reel[step]
    # T0= current_value
    time_prev=0
    # phi=gas_value_list[-1]
    for time in range(1,300):
        gas_value_reel.append(df['gas_value'].to_list()[step])
        Text_list.append(text)
        Tint_list.append(df['Tint'].to_list()[step])
        error = setpoint - current_value
        errorList.append(error)
        gas_value,integral=PID(Kp=100, Ki=100.0, Kd=66,integral=integralList[-1],lasterror=lastErrorList[-1],error=error)
        integralList.append(integral)
        lastErrorList.append(error)
        gas_value_list.append(gas_value)
        Tint_reel_list.append(tint_r)
        tspan = np.linspace(time_prev, time, 2)
        current_value = odeint(system2,current_value_list[-1], tspan, args =(gas_value_list[-1],text), tfirst=True)[-1][0]
        # current_value += system(t=time,Text=text,phi=gas_value,T0=T0) * dt
        current_value_list.append(current_value)
        text_1=Text_list[-1]
        time_prev = time


# print('ok')
# df=pd.DataFrame()
# df['gas_value']=np.array(gas_value_list)
# df['current_value']=np.array(current_value_list)
# df['error']=np.array(errorList)
# df['Text']=np.array(Text_list)


fig, axs = plt.subplots(3)
axs[0].plot(current_value_list,c='blue')
axs[0].plot(Tint_reel_list,c='orange')
axs[1].plot(gas_value_list,c='blue')
axs[1].plot(gas_value_reel,c='orange')
axs[2].plot(Text_list)
plt.show()