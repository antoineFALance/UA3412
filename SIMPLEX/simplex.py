import pandas as pd
from scipy.optimize import milp
from scipy.optimize import LinearConstraint
from scipy.optimize import linprog
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import math

df= pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv',sep=";")
hour_range=50
df['consigne'] = np.where((df['hour'] >= 8) & (df['hour'] <= 17), 18, np.nan)
df['consigne'] = np.where((df['hour'] >= 20), 0, df['consigne'])
df['consigne'] = np.where((df['hour'] <= 5) & (df['hour'] <= 23), 0, df['consigne'])
df['consigne'].interpolate(inplace=True)
Tc=df['consigne'].to_list()[:hour_range]
Text=df['Text'].to_list()[:hour_range]
T0=df['Tint'].to_list()[0]
#
R=0.1
C_inertia=100
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
figure,axs =plt.subplots(2)
axs[0].plot(action_resul)
axs[1].plot(Tint_result)
axs[1].plot(Tc)
plt.show()
print('ok')