import random

import numpy as np
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import pandas as pd
import math
from scipy.optimize import differential_evolution as de
from sklearn.metrics import mean_squared_error


df = pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_Min.csv',sep=";", decimal=",")
df = df[(df['year'] == 2018) & (df['month'] == 2) & ((df['day'] >= 13) & (df['day'] <= 13))]
df['gas_value'].fillna(0,inplace=True)
df['Tint_living'].interpolate(inplace=True)

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

to_list=df['Text'].to_list()
Q_list=df['gas_value'].to_list()
Qs_list=df['rad_soleil'].to_list()
ti_list=df['Tint_living'].to_list()

t_end=df.shape[0]-2
n=(df.shape[0]-2)*10
t_eval=np.linspace(0,t_end,n)

def kf_predict(X, P, A, Q, B, U):
    X = np.dot(A, X) + np.dot(B, U)
    P = np.dot(A, np.dot(P, A.T)) + Q
    return(X,P)

def kf_update(X, P, Y, H, R):
    IM = np.dot(H, X)
    IS = R + np.dot(H, np.dot(P, H.T))
    K = np.dot(P, np.dot(H.T, inv(IS)))
    X = X + np.dot(K, (Y-IM))
    P = P - np.dot(K, np.dot(IS, K.T))
    LH = gauss_pdf(Y, IM, IS)
    return (X,P,K,IM,IS,LH)

def gauss_pdf(X, M, S):
    X=np.atleast_2d(X)
    if M.shape[1] == 1:
        try:
            DX = X - tile(M, X.shape[1])
        except:
            print('ok')
        E = 0.5 * sum(DX * (np.dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * math.log(2 * math.pi) + 0.5 * math.log(linalg.det(S))
        P = math.exp(-E)
    elif X.shape[1] == 1:
        DX = tile(X, M.shape[1])- M
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
        E = E + 0.5 * M.shape[0] * math.log(2 * math.pi) + 0.5 * math.log(linalg.det(S))
        P = math.exp(-E)
    else:
        DX = X-M
        E = 0.5 * dot(DX.T, dot(inv(S), DX))
        E = E + 0.5 * M.shape[0] * math.log(2 * math.pi) + 0.5 * math.log(linalg.det(S))
        P = math.exp(-E)
    return (P,E[0])


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
    if np.isnan(ti_interpolate):
        print('ok')
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
    xlb_t = math.floor(t)
    xub_t = xlb_t+1
    test = to_list[xub_t]
    test1=to_list[xlb_t]
    test3=(to_list[xub_t]-to_list[xlb_t])/2
    return (to_list[xub_t]+to_list[xlb_t])/2

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

re=2.5
ri=0.06
ce=3000
ci=90
hrad=0.06
As=0
cw=4.18

ti0=df['Tint_living'].to_list()[0]
tret0=df['T_output_rad_living'].to_list()[0]
te0=(ri*to_list[0]+re*ti0)/(re+ri)
Qf0=df['gas_value'].to_list()[0]
To0=df['Text'].to_list()[0]

res=integrate.solve_ivp(ode_system, [0,t_end],[tret0,ti0,te0],t_eval=t_eval,args=(re,ri,ce,ci,hrad,As))
predictions,measurements=[],[]
dt=1/10
dTret0=1/cw*(Qf0-hrad*(tret0-ti0))
dti0=1/ci*((te0-ti0)/ri+hrad*(tret0-ti0))
dte0=1/ce*((ti0-te0)/ri+(To0-te0)/re)

X = np.array([[tret0], [ti0], [te0], [dTret0],[dti0],[dte0]])
P = np.diag((0.01, 0.01, 0.01, 0.01,0.01,0.01))
A=np.array([[1, 0,0, dt , 0,0], [0, 1, 0,0, dt,0], [0, 0, 1, 0,0,dt], [0, 0, 0,1,0,0],[0, 0, 0,0,1,0],[0, 0, 0,0,0,1]])
Q = np.eye(X.shape[1])
B = np.eye(X.shape[0])
U = np.zeros((X.shape[0],1))
# Measurement matrix
Y = np.array([[X[1][0]]])
H = np.array([[0, 1, 0, 0,0,0]])
R = np.eye(Y.shape[0])

for t in range((df.shape[0]-2)*10):
    (X, P) = kf_predict(X, P, A, Q, B, U)
    (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
    mes=tif(t/10)
    if t%30==0:
        Y=np.array([[mes]])
        measurements.append(mes)
    else:
        Y=np.array([[X[1][0]]])+random.uniform(-0.01,0.01)
        measurements.append(np.nan)
    predictions.append(Y[0][0])



figure,axs =plt.subplots(3)
axs[0].plot(predictions,c="red")
axs[0].plot(measurements,c='green',linestyle='',marker='x')
plt.show()

print('ok')



