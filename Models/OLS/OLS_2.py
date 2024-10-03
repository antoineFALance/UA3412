import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,Lasso
import numpy as np
import math
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_validate
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt



df_result=pd.DataFrame()
df= pd.read_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_Min.csv',sep=";",decimal=",")
df['rad_soleil']=df.apply(lambda x: max(0,-2.0*math.cos(math.pi/12*x.hour)),axis=1)
df['rad_soleil']=np.where(df['min']==0,df['rad_soleil'],np.nan)
df['rad_soleil'].interpolate(inplace=True)
# df1=df[(df['year']==2018)&(df['month']==2) & (df['day']==4) &((df['hour']>=2) & (df['hour']<=6))]
df=df[(df['year']==2018)&(df['month']==2)]
df['gas_value'].fillna(0,inplace=True)
df['time']=np.array(range(df.shape[0]))
df['T_output_rad_living'].interpolate(inplace=True)
df['Tint_living'].interpolate(inplace=True)
df['Text'].interpolate(inplace=True)

figure,axs =plt.subplots(4)
axs[0].plot(df['Tint_living'].to_list(),c='red')
axs[1].plot(df['gas_value'].to_list())
axs[2].plot(df['Text'].to_list())
axs[3].plot(df['T_output_rad_living'].to_list())
plt.show()
#

# print('ok')
df['ti_1']=df['Tint_living'].shift(1)
df['ti_2']=df['Tint_living'].shift(2)
df['tret_2']=df['T_output_rad_living'].shift(2)
df['to_2']=df['Text'].shift(2)
df['Qh_2']=df['gas_value'].shift(2)
df['Qs_1']=df['rad_soleil'].shift(1)

features = ['ti_1','ti_2','to_2','Qh_2','Qs_1']
y=['Tint_living']
df_ols=df[features+y]
# df_ols=df_ols[df_ols['Qh_2']!=0]
df_ols.dropna(inplace=True)



x_train,x_test,y_train,y_test=train_test_split(df_ols[features].to_numpy(),df_ols[y].to_numpy().flatten(),test_size=0.2,random_state=123)
ols = Ridge()
scores = cross_validate(ols,x_train,y_train,cv=5,scoring='neg_mean_squared_error',return_estimator=True)
coeffs=scores['estimator'][0].coef_
print(coeffs)
intercept=scores['estimator'][0].intercept_

# Preds
ti=[df['Tint_living'].to_list()[2]]
qh=df['gas_value'].to_list()
ti = [0.84*ti[-1]+5e-04*qh[t] for t in range(0,df.shape[0])]
y_pred=ti
# mse = mean_squared_error(y_test,y_pred)
# residuals = y_test-y_pred

df['y_pred']=y_pred
figure,axs =plt.subplots(4)
axs[0].plot(df['y_pred'].to_list(),c='blue')
axs[0].plot(df['Tint_living'].to_list(),c='red')
axs[1].plot(df['Qh_2'].to_list())
axs[2].plot(df['Text'].to_list())
axs[3].plot(df['T_output_rad_living'].to_list())
plt.show()

# ax=plt.plot(y_pred, residuals, c='red', marker='o', linestyle="")
# plt.title("residuels")
# plt.xlabel("y_pred")
# plt.ylabel("residuels")
# plt.show()
# ax1=plt.plot(y_test,y_pred,c='black',marker='x',linestyle="")
# plt.plot(range(14,30),range(14,30),c="green")
# plt.show()
# plt.xlabel("y_pred")
# plt.ylabel("y_reel")
# plt.close()