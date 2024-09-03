import math

import pylab as pl
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
    PATH_TO_INPUTS_DIR='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv'

else:
    PATH_TO_INPUTS_DIR = os.path.dirname(os.getcwd()) + "/data_/main_dataset/"

def PID(Kp,Ki,Kd,integral,lasterror,error,dt=0.1):
    derivative = (error - lasterror) / dt
    integral += error * dt
    output = Kp * error + Ki * integral + Kd * derivative
    return max(output,0),integral

def system(t,Text,phi,T0):
    # dTidt=math.exp(-t/(R*C))/C*(1/R*(Text-T0)+phi)
    dTidt = math.exp(-t / (R * C)) / C * (1 / R * (Text - T0) + phi)
    return dTidt

def system2(t,Ti,phi,text):
    dTidt=1/C*(1/R*(text-Ti)+phi)
    # dTidt = math.exp(-t / (R * C)) / C * (1 / R * (text - T0) + phi)
    return dTidt
nb_days=60

# DONNEES
time = 0.0

R=0.1
C=500

lastErrorList = [0]
integralList = [0]
Text_list=[]
gas_value_reel=[]
errorList=[]
gas_hr=[]
df=pd.read_csv(PATH_TO_INPUTS_DIR,sep=";")
TextList=df['Text'].tolist()
setpointList=df['Tint'].tolist()
current_value_list=[0]
current_value = 0
t=0
k=1
nb_hours=100
for hour in tqdm(range(nb_hours)):
    gas_value_list = []
    text=TextList[hour]
    setpoint=setpointList[hour]
    lastErrorList = [0]
    integralList = [0]
    errorList = []
    for time in range(1,100):
        # CALCUL ERREUR
        error = setpoint - current_value_list[-1]
        errorList.append(error)
        phi,integral=PID(Kp=0.1, Ki=0.0, Kd=0.0,integral=integralList[-1],lasterror=lastErrorList[-1],error=error)
        phi=k*phi
        integralList.append(integral)
        lastErrorList.append(error)
        # CALCUL
        gas_value_list.append(phi)
        gamma=math.exp(-t/(R*C))
        current_value=R*phi*(1-gamma)+text*(1-gamma)+current_value_list[0]*gamma
        current_value_list.append(current_value)
        t+=1

    gas_hr.append(sum(gas_value_list)/500)
print('ok')

fig, axs = plt.subplots(2)
axs[0].plot(setpointList[:nb_hours],c='orange')
axs[1].plot(gas_hr,c='orange')
ax1_2=axs[1].twinx()
ax1_2.plot(df['gas_value_corrected'].tolist()[:nb_hours],c='green')
plt.show()











