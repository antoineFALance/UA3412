import pandas as pd
import scipy.linalg
from scipy.optimize import milp
from scipy.optimize import LinearConstraint
from scipy.optimize import linprog
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import math
from sklearn.metrics import r2_score,mean_squared_error
import tqdm

df_result=pd.DataFrame()
df= pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv',sep=";")
df=df[(((df['month']==11) & (df['day']>=25)) | (df['month']==12) | (df['month']==1)) & (df['year']==2017)]
hour_range=500
df['consigne'] = np.where((df['hour'] >= 8) & (df['hour'] <= 17), 18, np.nan)
df['consigne'] = np.where((df['hour'] >= 20), 0, df['consigne'])
df['consigne'] = np.where((df['hour'] <= 5) & (df['hour'] <= 23), 0, df['consigne'])
df['consigne'].interpolate(inplace=True)
# Tc=df['consigne'].to_list()[:hour_range]
Tc=df['Tint'].to_list()[:hour_range]
Text=df['Text'].to_list()[:hour_range]
Tint=df['Text'].to_list()[:hour_range]
phi=df['gas_value'].to_list()[:hour_range]
T0=df['Tint'].to_list()[0]
# df[14:37].plot(x='hour',y='consigne')
# plt.show()
# print('ok')
#(0.4/125)

# Combinaison R et C
RCresults={}
RList_=[]
CList_=[]
RMSEList_=[]
bestCfgList=[]
df_result=pd.DataFrame()

for RC in range(1,500):
    print('RC: '+str(RC))
    # RC=50
    RList=[value/10 for value in range(1,100,5)]
    C_inertia_list=[RC/R for R in RList]
    # R=0.3
    # C_inertia=50/R
    Combinations=[(RList[index],C_inertia_list[index]) for index in range(len(RList))]
    corrList=[]
    for index in tqdm.tqdm(range(len(Combinations))):
    # for cfg in Combinations:
        R, C_inertia = Combinations[index][0], Combinations[index][1]
        # R,C_inertia=cfg[0],cfg[1]
        gamma=math.exp(-1/(R*C_inertia))
        A=np.diag([gamma]*(hour_range-1))
        A=A+np.diag([-1]*(A.shape[0]-1),1)
        A=R*A[:-1,:]
        b_l=[]
        for index in range(1,hour_range-1):
            b_l.append(Text[index]-Tint[index-1]-Text[index-1]-Tc[index])

        b_ub=np.array(b_l).reshape(-1,1)
        bounds=[tuple([0,np.inf]) for _ in range(hour_range-1)]
        C=np.array([1]*(hour_range-1))
        res = linprog(C,A,b_ub=b_ub,bounds=bounds)
        action_resul=[action for action in list(res.x)]
        Tint_result=[T0]
        for index in range(hour_range-1):
            Tint_result.append(Text[index]+R*phi[index]+gamma*(Tint[index-1]-Text[index-1]-R*phi[index-1]))

        # correlation
        corrList.append(math.sqrt(mean_squared_error(action_resul,phi[:len(action_resul)])))

    bestConfig=Combinations[np.argmin(corrList)]
    RList_.append(bestConfig[0])
    CList_.append(bestConfig[1])
    RMSEList_.append(corrList[np.argmin(corrList)])
    RCresults[RC]=(bestConfig,corrList[np.argmin(corrList)])

print('ok')
df_result['R']=np.array(RList_)
df_result['C']=np.array(CList_)
df_result['RMSE']=np.array(RMSEList_)

df_result.to_csv('RC_parameters.csv')
