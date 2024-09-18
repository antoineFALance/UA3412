import scipy.signal as sig
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution as de

df= pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_Min.csv',sep=";",decimal=",")
df['rad_soleil']=df.apply(lambda x: max(0,-2.0*math.cos(math.pi/12*x.hour)),axis=1)
df['rad_soleil']=np.where(df['min']==0,df['rad_soleil'],np.nan)
df['rad_soleil'].interpolate(inplace=True)
# df1=df[(df['year']==2018)&(df['month']==2) & (df['day']==4) &((df['hour']>=2) & (df['hour']<=6))]
df1=df[(df['year']==2018)&(df['month']==2) & (df['day']==4)]
df1['gas_value'].fillna(0,inplace=True)
df1['time']=np.array(range(df1.shape[0]))
T0=df1['Tint_living'].to_list()[0]



# def objective_function(R1,R2,R3,C1,C2,Hrad,As):
def objective_function(x):
    # state space model
    R1=x[0]
    R2=x[1]
    R3=x[2]
    C1=x[3]
    C2=x[4]
    Hrad=x[5]
    As=x[6]

    Cwater = 4.18
    t = df1['time'].to_list()
    u=df1[['Text','gas_value','rad_soleil']].to_numpy()

    A = [[-(1/(R1*C1)+1/(R2*C1)),1/(R2*C1),0],
        [1/(R2*C2),-(1/(R2*C2+1/(R3*C2))),Hrad/C2],
         [0,0,-Hrad/Cwater]]

    B = [[1,0,0],[0,0,As/C2],[0,1/Cwater,0]]

    C = [[(1/R1-1/R2)/(1/R3-Hrad),(1/R3-1/R2)/(1/R3-Hrad),(-Hrad)/(1/R3-Hrad)]]
    D = [[-1/(R1*(1/R3-Hrad)),0,As/(1/R3-Hrad)]]

    sys = sig.StateSpace(A, B, C, D)
    t, y, x = sig.lsim(sys, u, t)
    y[np.isnan(y)]=0
    y[np.isinf(y)] = 0

    return mean_squared_error(y,df1['Tint_living'].to_numpy())


# bounds = ([0,1e03], [0,1e03], [0,1e03],[ 0,1e03], [0,1e03],[ 0,1e03], [0,1e03])
# result= de(objective_function,bounds,args=())
# print(result.x)
# print('ok')

def state_space_model(x):
    # state space model
    R1=x[0]
    R2=x[1]
    R3=x[2]
    C1=x[3]
    C2=x[4]
    Hrad=x[5]
    As=x[6]

    Cwater = 4.18
    t = df1['time'].to_list()
    u=df1[['Text','gas_value','rad_soleil']].to_numpy()

    A = [[-(1/(R1*C1)+1/(R2*C1)),1/(R2*C1),0],
        [1/(R2*C2),-(1/(R2*C2+1/(R3*C2))),Hrad/C2],
         [0,0,-Hrad/Cwater]]

    B = [[1,0,0],[0,0,As/C2],[0,1/Cwater,0]]

    C = [[(1/R1-1/R2)/(1/R3-Hrad),(1/R3-1/R2)/(1/R3-Hrad),(-Hrad)/(1/R3-Hrad)]]
    D = [[-1/(R1*(1/R3-Hrad)),0,As/(1/R3-Hrad)]]

    sys = sig.StateSpace(A, B, C, D)
    t, y, x = sig.lsim(sys, u, t)
    y[np.isnan(y)]=0
    y[np.isinf(y)] = 0

    return y

print('ok')
y=state_space_model([2.26363421e+00, 1.35858223e-01, 6.74932366e+02, 3.45213946e+00, 3.18077116e+02, 2.40709010e+01, 0.00000000e+00])
plt.plot(y)
plt.plot(df1['Tint_living'].to_numpy())
plt.show()
print('ok')


