import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd
import math
from scipy.optimize import differential_evolution as de
from sklearn.metrics import mean_squared_error

df = pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_Min.csv',sep=";", decimal=",")
df = df[(df['year'] == 2018) & (df['month'] == 2) & ((df['day'] >= 13) & (df['day'] <= 13))]
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

df['consigne'] = np.where((df['hour'] >= 8) & (df['hour'] <= 12), 18, np.nan)
df['consigne'] = np.where((df['hour'] >= 14) & (df['hour'] <= 17), 12,  df['consigne'])
df['consigne'] = np.where((df['hour'] >= 19) & (df['hour'] <= 22), 15,df['consigne'])
df['consigne'] = np.where((df['hour'] >= 0) & (df['hour'] <= 6), 5,df['consigne'])
df['consigne'].interpolate(inplace=True)

to_list=df['Text'].to_list()
Q_list=df['gas_value'].to_list()
Qs_list=df['rad_soleil'].to_list()
ti_list=df['Tint_living'].to_list()

t_end=df.shape[0]-2
n=(df.shape[0]-2)*10
t_eval=np.linspace(0,t_end,n)

def Qf(t):
    xlb_t = math.floor(t)
    xub_t=xlb_t+1
    try:
        yub=Q_list[xub_t]
    except:
        print('ok')
    ylb=Q_list[xlb_t]
    q=-(xlb_t*yub-t*yub-xub_t*ylb+t*ylb)/(xub_t-xlb_t)
    return q

def tif(t):
    xlb_t = math.floor(t)
    xub_t=xlb_t+1
    try:
        yub=ti_list[xub_t]
    except:
        print('ok')
    ylb=ti_list[xlb_t]
    ti_interpolate=-(xlb_t*yub-t*yub-xub_t*ylb+t*ylb)/(xub_t-xlb_t)
    return ti_interpolate

def Qsf(t):
    xlb_t = math.floor(t)
    xub_t=xlb_t+1
    try:
        yub=Qs_list[xub_t]
    except:
        print('ok')
    ylb=Qs_list[xlb_t]
    qs_interpolate=-(xlb_t*yub-t*yub-xub_t*ylb+t*ylb)/(xub_t-xlb_t)
    return qs_interpolate

def tof(t):
    xub_t = round(t)
    xlb_t = math.ceil(t)
    return (to_list[xub_t]-to_list[xlb_t])/2

def ode_system(t,x,re,ri,ce,ci,hrad,As):
    # tret,ti,te=x
    sol=np.array([
        1/4.18*(Qf(t)-hrad*(x[0]-x[1])),
        1/ci*((x[2]-x[1])/ri+hrad*(x[0]-x[1])+As*Qsf(t)),
        1/ce*((x[1]-x[2])/ri+(tof(t)-x[2])/re)])
    return sol

def solveOde(t,re,ri,ce,ci,hrad,As):
    ti0=df['Tint_living'].to_list()[0]
    tret0=df['T_output_rad_living'].to_list()[0]
    te0=(ri*to_list[0]+re*ti0)/(re+ri)
    res=integrate.solve_ivp(ode_system,[0,t_end],[tret0,ti0,te0],t_eval=t_eval,args=(re,ri,ce,ci,hrad,As))
    return mean_squared_error(res.y[1],z)

z=[tif(t) for t in t_eval]
# plt.plot(t_eval,z,linestyle='None',marker='x')
# plt.plot(range(df.shape[0]),ti_list,c='red')
# plt.show()

bounds=[(0,0,0,0,0,0),(10,1,4000,200,1,20)]

# bounds = ([0,0.1e02], [0,5e03], [0,0.1e-01],[ 0,0.1e00], [0,1e01], [0,1e01])
# result= de(solveOde,bounds,args=())

# popt, pcov = optimize.curve_fit(solveOde, t_eval, z,bounds=bounds,p0=[2.5,0.06,3000,90,0.04,5])

re=2.5
ri=0.06
ce=3000
ci=90
hrad=0.06
As=0

ti0=df['Tint_living'].to_list()[0]
tret0=df['T_output_rad_living'].to_list()[0]
te0=(ri*to_list[0]+re*ti0)/(re+ri)

res=integrate.solve_ivp(ode_system, [0,t_end],[tret0,ti0,te0],t_eval=t_eval,args=(re,ri,ce,ci,hrad,As))

# plt.plot(t_eval,res.y[1],'r', label="optimized data")
# plt.plot(t_eval,z,'k--',label="data")
# plt.legend()
# plt.show()


figure,axs =plt.subplots(4)
axs[0].plot(t_eval,res.y[1],'r', label="optimized data")
axs[0].plot(t_eval,z,'k--',label="data")
axs[1].plot(df['gas_value'],c='red')
axs[2].plot(df['rad_soleil'],c='black')
axs[3].plot(df['Text'])
plt.show()

print('ok')



