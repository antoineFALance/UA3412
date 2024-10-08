import math
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import r2_score,mean_squared_error
from matplotlib import  pyplot as plt
is_windows = hasattr(sys, 'getwindowsversion')
plt.rcParams["figure.figsize"] = (20, 10)
import tensorflow as tf
import joblib

# PATH DEFINITION
PATH_TO_INPUT = 'C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv'
PATH_TO_MODEL='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\Models\\DNN\\models\\home_id_106_3_128_0.02.h5'
scaler = joblib.load('DNN_scaler.gz')
df=pd.read_csv(PATH_TO_INPUT,sep=";")
df=df[(((df['month']==11) & (df['day']>=25)) | (df['month']==12) | (df['month']==1)) & (df['year']==2017)]
df_scaled=pd.DataFrame(data=scaler.transform(df),columns=df.columns)

dnn= tf.keras.models.load_model(PATH_TO_MODEL)
y_int_mean=scaler.mean_[5]
y_int_var=scaler.var_[5]
test = df_scaled[['Tint', 'Text', 'gas_value']].to_numpy()
df['y_pred']=dnn.predict(df_scaled[['Tint', 'Text', 'gas_value']].to_numpy())*y_int_var+y_int_mean
R,C=0.4,366
gamma=math.exp(-1/(R*C))
df['y_pred_RC']=df.apply(lambda x:x.Tint*gamma+(R*x.gas_value+x.Text)*(1-gamma),axis=1)
test=df_scaled.apply(lambda x:0.952113*x.Tint+0.014198*x.Text+0.069*x.gas_value,axis=1)
df['y_pred_VAR']=np.array(test)*y_int_var+y_int_mean
df['y_pred__']=df.apply(lambda x:x.Tint,axis=1)
# df.plot.scatter(x='Tint',y='y_pred')
# plt.show()
print('ok')

figure,axs =plt.subplots(3)
axs[0].plot(df['Tint'].to_list(),c='blue')
axs[0].plot(df['y_pred_RC'].to_list(),c='red',linestyle='--')
axs[0].plot(df['y_pred__'].to_list(),c='orange',linestyle='dashdot',linewidth=1.5)
# axs[0].plot(df['y_pred_VAR'].to_list(),c='black',linestyle='dashdot',linewidth=2)
axs[1].plot(df['gas_value'].to_list(),c='orange')
axs[2].plot(df['Text'].to_list(),c='violet')
plt.show()
print('ok')

# plt.plot(df['y_pred_RC'].to_list(),df['Tint'].to_list(),linestyle='None', marker='x',c='black')
#
# plt.plot(range(10,28),range(10,28),c='green')
# plt.title('Tint,RC vs Tint relle')
# plt.xlabel('Tint,RC')
# plt.ylabel('Tint,relle')
# plt.show()

print(mean_squared_error(df['y_pred_VAR'].to_list(),df['Tint'].to_list()))