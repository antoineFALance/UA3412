import pandas as pd
from scipy.optimize import milp
from scipy.optimize import LinearConstraint
from scipy.optimize import linprog
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import math
df_result=pd.DataFrame()
df= pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv',sep=";")
hour_range=500
df['consigne'] = np.where((df['hour'] >= 8) & (df['hour'] <= 17), 18, np.nan)
df['consigne'] = np.where((df['hour'] >= 20), 0, df['consigne'])
df['consigne'] = np.where((df['hour'] <= 5) & (df['hour'] <= 23), 0, df['consigne'])
df['consigne'].interpolate(inplace=True)
# Tc=df['consigne'].to_list()[:hour_range]
Tc=df['Tint'].to_list()[:hour_range]
Text=df['Text'].to_list()[:hour_range]
phi=df['gas_value'].to_list()[:hour_range]
T0=df['Tint'].to_list()[0]
# df[14:37].plot(x='hour',y='consigne')
# plt.show()
# print('ok')
#(0.4/125)
R=0.6
C_inertia=R/50
gamma=math.exp(-1/(R*C_inertia))
C=np.array([-1]*(hour_range-1)).reshape(-1,1)
C_col=np.array([gamma**index for index in range(0,hour_range-1)]).reshape(-1,1)
for idx in range(1,hour_range-1):
    test = C_col[:hour_range-idx-1,0].reshape(-1,1)
    C_col_i=np.vstack([np.array([0]*idx).reshape(-1,1),test.reshape(-1,1)])
    C_col=np.hstack([C_col,C_col_i])
A=-(1-gamma)*R*C_col
b_l=[]
for index in range(1,hour_range):
    b_l.append(T0*gamma**(index)-Tc[index]+sum([Text[idx]*(1-gamma)*(gamma**(index-idx-1)) for idx in range(index)]))

b_ub=np.array(b_l).reshape(-1,1)
bounds=[tuple([0,np.inf]) for _ in range(hour_range-1)]
res = linprog(-C,A,b_ub=b_ub,bounds=bounds)
action_resul=[action for action in list(res.x)]
Tint_result=[T0]
for index in range(hour_range-1):
    Tint_result.append(gamma*Tint_result[-1]+(R*action_resul[index]+Text[index])*(1-gamma))
figure,axs =plt.subplots(3)
axs[0].plot(action_resul,c='blue')
axs[0].plot(phi,c='red')
axs[1].plot(Tint_result)
axs[1].plot(Tc)
axs[2].plot(Text[:hour_range-1])
plt.show()
print(sum(action_resul))
df_result['phi_optim']=action_resul
df_result['Tint']=Tint_result[:len(action_resul)]
df_result['Tc']=Tc[:len(action_resul)]
df_result['gas_value']=phi[:df_result.shape[0]]
df_result.to_csv('optim_lin_results.csv')
#
figure,axs =plt.subplots(1)
axs.plot(action_resul,phi[:df_result.shape[0]],c='blue',linestyle='None',marker='x')
axs.plot(range(40),range(40),c='green')
# plt.axis('scaled')
plt.show()
print('ok')
