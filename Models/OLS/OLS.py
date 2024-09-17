import math

import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize as so
from statistics import mean,stdev
import sys
is_windows = hasattr(sys, 'getwindowsversion')
import scipy.stats as stats
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_validate
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_squared_error

def save_figure(ax,filename,directory):
    fig = ax[0].get_figure()
    fig.savefig(directory + filename + ".png")

def windowing(dataset,windowRange):
    x_list,y_list=[],[]

    for day in dataset['cd_yearMonthDay'].unique().tolist():
        df_day = dataset[dataset['cd_yearMonthDay']==day]
        if not df_day.isnull().any().any() :
            df_day['t']=np.array(range(df_day.shape[0]))
            # ax=df_day.plot(x='t',y='Tint')
            # df_day.plot(x='t', y='Text',ax=ax)
            x_list.append(df_day[['gas_value','Text','Tint']].to_numpy())
            y_list.append(df_day[['Tint']].to_numpy())
        else:
            pass
    return x_list,y_list


    x=np.stack(x_ds)
    y=np.stack(y_ds).reshape(-1, 1)
    wdw=10
    x_,y_=[],[]
    for step in range(x.shape[0] - wdw):
        x_chunk = x[step:step+wdw,:]
        y_chunk = y[step:step+wdw,:]
        x_.append(x_chunk)
        y_.append(y_chunk)

    return np.array(x_),np.array(y_)


# PATH DEFINITION
if is_windows:
    PATH_TO_INPUT_DIR_DATA='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv'
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.path.dirname(os.getcwd()))+"\\Models\\OLS\\results\\"
else:
    PATH_TO_INPUT_DIR_DATA= os.path.dirname(os.path.dirname(os.getcwd()))+"/data_/dataset_gas_value_corrected/"
    PATH_TO_OUTPUT_DIR_DATA=os.path.dirname(os.path.dirname(os.getcwd()))+"/Models/OLS/results/"
directory = os.fsencode(PATH_TO_INPUT_DIR_DATA)

df_res=pd.DataFrame()


df_temp=pd.DataFrame()
RList, CList = [], []
Ri_List,R0_List,Ci_List,Ce_List=[],[],[],[]
df_dataset = pd.read_csv(PATH_TO_INPUT_DIR_DATA, sep=";")
df_dataset=df_dataset[(((df_dataset['month']==11) & (df_dataset['day']>=25)) | (df_dataset['month']==12) | (df_dataset['month']==1)) & (df_dataset['year']==2017)]

df_dataset['Tint1']=df_dataset['Tint'].shift(1)

df_ols = df_dataset[['Tint','gas_value','Text','Tint1']].dropna()

params={}

# df_ols = (df_ols-df_ols.mean())/(df_ols.std())
x_train,x_test,y_train,y_test=train_test_split(df_ols[['Tint','gas_value','Text']].to_numpy(),df_ols[['Tint1']].to_numpy().flatten(),test_size=0.2,random_state=123)
ols=LinearRegression()
scores = cross_validate(ols,x_train,y_train,cv=5,scoring='neg_mean_squared_error',return_estimator=True)
df_res=pd.concat([df_res,df_temp])
y_pred = scores['estimator'][0].predict(x_test)
mse = mean_squared_error(y_test,y_pred)
residuals = y_test-y_pred

ax=plt.plot(y_pred, residuals, c='red', marker='o', linestyle="")
plt.title("residuels")
plt.xlabel("y_pred")
plt.ylabel("residuels")
plt.show()
ax1=plt.plot(y_test,y_pred,c='black',marker='x',linestyle="")
plt.plot(range(14,30),range(14,30),c="green")
plt.show()
plt.xlabel("y_pred")
plt.ylabel("y_reel")
plt.close()



