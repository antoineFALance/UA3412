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
from scipy import integrate, optimize

df = pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_Min.csv',sep=";", decimal=",")
df = df[(df['year'] == 2018) & (df['month'] == 2) & ((df['day'] >= 13) & (df['day'] <= 14))]
df['gas_value'].fillna(0,inplace=True)

# ead soleil
qs=2.5
hour_start=11
hour_max=14
a=-qs/(hour_start**2-2*hour_max*hour_start+hour_max**2)
b=(2*hour_max*qs)/(hour_start**2-2*hour_max*hour_start+hour_max**2)
c=-(2*hour_max*hour_start-hour_start**2)*qs/(hour_start**2-2*hour_max*hour_start+hour_max**2)
df['rad_soleil'] = df.apply(lambda x: max(0,a*x.hour**2+b*x.hour+c), axis=1)
df['rad_soleil'] = np.where(df['min'] == 0, df['rad_soleil'], np.nan)
df['rad_soleil'].interpolate(inplace=True)
# df['rad_soleil']=np.where(((df['hour'] >= 12) & (df['hour'] <= 16)),df['rad_soleil'],0)

df['consigne'] = np.where((df['hour'] >= 10) & (df['hour'] <= 12), 18, np.nan)
df['consigne'] = np.where((df['hour'] >= 14) & (df['hour'] <= 17), 12, df['consigne'])
df['consigne'] = np.where((df['hour'] >= 19) & (df['hour'] <= 22), 10, df['consigne'])
df['consigne'] = np.where((df['hour'] >= 0) & (df['hour'] <= 1), 5, df['consigne'])
df['consigne'] = np.where((df['hour'] >= 3) & (df['hour'] <= 3), 7, df['consigne'])
df['consigne'] = np.where((df['hour'] >= 5) & (df['hour'] <= 7), 3, df['consigne'])
df['consigne'].interpolate(inplace=True)

to_list=df['Text'].to_list()
Q_list=[0]
Qs_list=df['rad_soleil'].to_list()
ti_list=df['Tint_living'].to_list()

t_end=df.shape[0]-2
n=(df.shape[0]-2)*10
t_eval=np.linspace(0,t_end,n)

def Qf(t):
    try :
        xlb_t = math.floor(t)
        xub_t=xlb_t+1
        yub=Q_list[xub_t]
        ylb=Q_list[xlb_t]
        q=-(xlb_t*yub-t*yub-xub_t*ylb+t*ylb)/(xub_t-xlb_t)
    except:
        q=0
    return q

def tif(t):
    xlb_t = math.floor(t)
    xub_t=xlb_t+1
    yub=ti_list[xub_t]
    ylb=ti_list[xlb_t]
    ti_interpolate=-(xlb_t*yub-t*yub-xub_t*ylb+t*ylb)/(xub_t-xlb_t)
    return ti_interpolate

def Qsf(t):
    xlb_t = math.floor(t)
    xub_t=xlb_t+1
    yub=Qs_list[xub_t]
    ylb=Qs_list[xlb_t]
    qs_interpolate=-(xlb_t*yub-t*yub-xub_t*ylb+t*ylb)/(xub_t-xlb_t)
    return qs_interpolate

def tof(t):
    xlb_t = math.floor(t)
    xub_t = xlb_t+1
    if (t-xlb_t)<(xub_t-t):
        return to_list[xlb_t]
    else:
        return to_list[xub_t]

def ode_system(t,x,re,ri,ce,ci,hrad):
    # tret,ti,te=x
    sol=np.array([
        1/4.18*(Qf(t)-hrad*(x[0]-x[1])),
        1/ci*((x[2]-x[1])/ri+hrad*(x[0]-x[1])),
        1/ce*((x[1]-x[2])/ri+(tof(t)-x[2])/re)])
    return sol

def solveOde(t,re,ri,ce,ci,hrad,As):
    ti0=df['Tint_living'].to_list()[0]
    tret0=df['T_output_rad_living'].to_list()[0]
    te0=(ri*to_list[0]+re*ti0)/(re+ri)
    res=integrate.solve_ivp(ode_system,[0,t_end],[tret0,ti0,te0],t_eval=t_eval,args=(re,ri,ce,ci,hrad,As))
    return res.y[1]

def PID(Kp,Ki,Kd,integral,lasterror,error,dt=1):
    maxAction=1000
    derivative = (error - lasterror) / dt
    integral += error * dt
    output = Kp * error + Ki * integral + Kd * derivative
    return min(max(output,0),maxAction),integral

def on_running(phi_values):
    plt.clf()
    plt.plot(df['consigne'].to_list(),c='red')
    plt.plot(phi_values,c='green')
    plt.draw()
    plt.pause(0.1)



re=2.5
ri=0.06
ce=3000
ci=90
hrad=0.06

ti0=5
tret0=0
te0=(ri*to_list[0]+re*ti0)/(re+ri)

ti_0_list=[ti0]
current_value_list=[ti0]
tret_0_list=[tret0]
te0_list=[te0]

gas_value_list = []
lastErrorList = [0]
integralList = [0]
errorList = []

for t in tqdm(range(1,df.shape[0])):
    text=tof(t)
    setpoint=df['consigne'].to_list()[t]

    # for time in range(1,10):
    # CALCUL ERREUR
    error = setpoint - current_value_list[-1]
    errorList.append(error)
    phi,integral=PID(Kp=300, Ki=0.0, Kd=0.5,integral=integralList[-1],lasterror=lastErrorList[-1],error=error)
    integralList.append(integral)
    lastErrorList.append(error)

    # CALCUL
    Q_list.append(phi)
    res = integrate.solve_ivp(ode_system, [0, t], [tret0, ti0,te0], t_eval=[t],args=(re, ri, ce, ci, hrad))
    tret_0_list.append(res.y[0][-1])
    ti_0_list.append(res.y[1][-1])
    te0_list.append(res.y[2][-1])
    current_value_list.append(res.y[1][0])
    if t%100==0:
        on_running(current_value_list)
plt.close()

deltaC=[current_value_list[idx]/df['consigne'].to_list()[idx] if current_value_list[idx]<0.95*df['consigne'].to_list()[idx] else 0 for idx in range(df.shape[0])]

figure,axs =plt.subplots(3)
axs[0].plot(ti_0_list,'r', label="Ti contrôlée PID")
axs[0].plot(df['consigne'].to_list(),'k--',label="Température consigne")
axs[1].plot([q/1 for q in Q_list],c='red',label="gaz")
axs[2].plot(deltaC,c='green',label="Delta consigne")


plt.show()

print("Energie gaz consommée: " +str(sum(Q_list)))
















