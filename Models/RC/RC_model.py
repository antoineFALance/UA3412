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
from tqdm import tqdm
from functools import reduce

PATH_TO_LIVING_TINT_FILE='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_livingroom1087_sensor4902_room_temperature.csv'
PATH_TO_BEDROOM_TINT_FILE='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_bedroom1088_sensor4908_room_temperature.csv'
PATH_TO_HALL_TINT_FILE='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_hall1086_sensor4898_room_temperature.csv'
PATH_TO_KITCHEN_TINT_FILE='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_hall1086_sensor4898_room_temperature.csv'
PATH_TO_GAS_PULSE_FILE='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_kitchen1085_sensor4884_gas-pulse_gas.csv'
PATH_TO_WEATHER_FILE='C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\data_\\weather_forecast\\transformed_weather_dataset.csv'


# FUNCTION
def Ti_function(input,R,C):
    t=list(input[:,0])
    gas_value=list(input[:,1])
    Text = list(input[ :,2])
    T0=input[0,3]
    Ti=[R*gas_value[i]*(1-math.exp(-t[i]/(R*C)))+Text[i]*(1-math.exp(-t[i]/(R*C)))+T0*math.exp(-t[i]/(R*C)) for i in range(len(t))]
    return np.array(Ti)

df_gas_pulse = pd.read_csv(PATH_TO_GAS_PULSE_FILE,sep=";",header=None)

df_Text=pd.read_csv(PATH_TO_WEATHER_FILE,sep=";")
df_gas_pulse.columns=['time_stamp','value']

df_gas_pulse['time_stamp']=pd.to_datetime(df_gas_pulse['time_stamp'])
df_gas_pulse['dateHour']=df_gas_pulse['time_stamp'].dt.floor('H')

# CREATION DES DONNEES CHRONO POUR JOINTURE
df_gas_pulse['year']=df_gas_pulse['time_stamp'].dt.year
df_gas_pulse['month']=df_gas_pulse['time_stamp'].dt.month
df_gas_pulse['day']=df_gas_pulse['time_stamp'].dt.day
df_gas_pulse['hour']=df_gas_pulse['time_stamp'].dt.hour
df_gas_pulse['min']=df_gas_pulse['time_stamp'].dt.minute
df_gas_pulse_min=df_gas_pulse[['year','month','day','hour','min','value']].groupby(by=['year','month','day','hour','min'],as_index=False).sum()

# LIVING ROOM
df_Tint_living = pd.read_csv(PATH_TO_LIVING_TINT_FILE,sep=";",header=None)
df_Tint_living.columns=['time_stamp','Tint_living']
df_Tint_living['time_stamp']=pd.to_datetime(df_Tint_living['time_stamp'])
df_Tint_living['year']=df_Tint_living['time_stamp'].dt.year
df_Tint_living['month']=df_Tint_living['time_stamp'].dt.month
df_Tint_living['day']=df_Tint_living['time_stamp'].dt.day
df_Tint_living['hour']=df_Tint_living['time_stamp'].dt.hour
df_Tint_living['min']=df_Tint_living['time_stamp'].dt.minute
df_Tint_living_min=df_Tint_living[['year','month','day','hour','min','Tint_living']].groupby(by=['year','month','day','hour','min'],as_index=False).mean()

# BEDROOM
df_Tint_bed = pd.read_csv(PATH_TO_BEDROOM_TINT_FILE,sep=";",header=None)
df_Tint_bed.columns=['time_stamp','Tint_bedroom']
df_Tint_bed['time_stamp']=pd.to_datetime(df_Tint_bed['time_stamp'])
df_Tint_bed['year']=df_Tint_bed['time_stamp'].dt.year
df_Tint_bed['month']=df_Tint_bed['time_stamp'].dt.month
df_Tint_bed['day']=df_Tint_bed['time_stamp'].dt.day
df_Tint_bed['hour']=df_Tint_bed['time_stamp'].dt.hour
df_Tint_bed['min']=df_Tint_bed['time_stamp'].dt.minute
df_Tint_bed_min=df_Tint_bed[['year','month','day','hour','min','Tint_bedroom']].groupby(by=['year','month','day','hour','min'],as_index=False).mean()

# HALL
df_Tint_hall = pd.read_csv(PATH_TO_HALL_TINT_FILE,sep=";",header=None)
df_Tint_hall.columns=['time_stamp','Tint_hall']
df_Tint_hall['time_stamp']=pd.to_datetime(df_Tint_hall['time_stamp'])
df_Tint_hall['year']=df_Tint_hall['time_stamp'].dt.year
df_Tint_hall['month']=df_Tint_hall['time_stamp'].dt.month
df_Tint_hall['day']=df_Tint_hall['time_stamp'].dt.day
df_Tint_hall['hour']=df_Tint_hall['time_stamp'].dt.hour
df_Tint_hall['min']=df_Tint_hall['time_stamp'].dt.minute
df_Tint_hall_min=df_Tint_hall[['year','month','day','hour','min','Tint_hall']].groupby(by=['year','month','day','hour','min'],as_index=False).mean()

# KITCHEN
df_Tint_kitchen= pd.read_csv(PATH_TO_KITCHEN_TINT_FILE,sep=";",header=None)
df_Tint_kitchen.columns=['time_stamp','Tint_kitchen']
df_Tint_kitchen['time_stamp']=pd.to_datetime(df_Tint_kitchen['time_stamp'])
df_Tint_kitchen['year']=df_Tint_kitchen['time_stamp'].dt.year
df_Tint_kitchen['month']=df_Tint_kitchen['time_stamp'].dt.month
df_Tint_kitchen['day']=df_Tint_kitchen['time_stamp'].dt.day
df_Tint_kitchen['hour']=df_Tint_kitchen['time_stamp'].dt.hour
df_Tint_kitchen['min']=df_Tint_kitchen['time_stamp'].dt.minute
df_Tint_kitchen_min=df_Tint_kitchen[['year','month','day','hour','min','Tint_kitchen']].groupby(by=['year','month','day','hour','min'],as_index=False).mean()

# AGGREGATION TINT
dfs = [df_Tint_living_min, df_Tint_bed_min, df_Tint_hall_min,df_Tint_kitchen_min]
df_Tint_min=reduce(lambda  left,right: pd.merge(left,right,on=['year','month','day','hour','min'],how='inner'), dfs)
df_Tint_min['Tint']=df_Tint_min[['Tint_kitchen','Tint_hall','Tint_bedroom','Tint_living']].mean(axis=1)

#CREATION D'UNE PLAGE DE TEMPS
df_main=pd.DataFrame()
df_main['time_stamp']=pd.date_range(start =str(df_gas_pulse['dateHour'].min()),end =str(df_gas_pulse['dateHour'].max()), freq ='min')
df_main['time_stamp']=df_main['time_stamp'].dt.floor('min')
df_main['year']=df_main['time_stamp'].dt.year
df_main['month']=df_main['time_stamp'].dt.month
df_main['day']=df_main['time_stamp'].dt.day
df_main['hour']=df_main['time_stamp'].dt.hour
df_main['min']=df_main['time_stamp'].dt.minute

# JOINTURE
df=df_main.merge(df_gas_pulse_min,on=['year','month','day','hour','min'],how='left')\
    .merge(df_Tint_min[['year','month','day','hour','min','Tint']],on=['year','month','day','hour','min'],how='left')\
    .merge(df_Text[['year','month','day','hour','Text']],on=['year','month','day','hour'],how='left')

df.rename(columns={'value': 'gas_value'},inplace=True)
df['gas_value']=df['gas_value']/10
df['gas_value']=df['gas_value']-df['gas_value'].min()
df['Tint']=df['Tint']/10
df['Tint'].interpolate(inplace=True)
df_1=df[['year','month','day','hour','min','Tint','Text']]
df_1.dropna(inplace=True)
df_1['t']=np.array(range(1,df_1.shape[0]+1))
R,C,=1.5,800
T0=df_1['Tint'].tolist()[0]
TiList = df_1['Tint'].tolist()
TeList = df_1['Text'].tolist()
gas_value_corrected=[]
for t in tqdm(range(df_1.shape[0])):
    gamma=math.exp(-t/(R*C))
    try:
        Ti=TiList[t]
        Te=TeList[t]
        gas_value_corrected.append(max(0,Ti/(R*(1-gamma))-Te/R-gamma/(R*(1-gamma))*T0))
    except:
        gas_value_corrected.append(0)

df_1['gas_value_corrected']=np.array(gas_value_corrected)

# fig, axs = plt.subplots(3)
# axs[0].plot(df_1['Text'].tolist(),c='blue')
# axs[1].plot(df_1['Tint'].tolist(),c='orange')
# ax1_twin=axs[1].twinx()
# axs[1].plot(df_1['gas_value_corrected'].tolist(),c='green')
# plt.show()
#
# df_1.to_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set.csv',sep=";",index=False)
# df_1_hr=df_1.groupby(['year','month','day','hour'],as_index=False).agg({'Text':'mean','Tint':'mean','gas_value_corrected':'sum'})
# df_1_hr.to_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_HR.csv',sep=";",index=False)

