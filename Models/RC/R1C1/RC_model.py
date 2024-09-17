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

PATH_TO_LIVING_TINT_FILE= '/IDEAL_home106/home106_livingroom1087_sensor4902_room_temperature.csv'
PATH_TO_BEDROOM_TINT_FILE= '/IDEAL_home106/home106_bedroom1088_sensor4908_room_temperature.csv'
PATH_TO_HALL_TINT_FILE= '/IDEAL_home106/home106_hall1086_sensor4898_room_temperature.csv'
PATH_TO_KITCHEN_TINT_FILE= '/IDEAL_home106/home106_hall1086_sensor4898_room_temperature.csv'

PATH_TO_GAS_PULSE_FILE= '/IDEAL_home106/home106_kitchen1085_sensor4884_gas-pulse_gas.csv'
PATH_TO_WEATHER_FILE= '/data_/weather_forecast/transformed_weather_dataset.csv'

PATH_TO_CENTRAL_HEAT_FLOW= '/IDEAL_home106/home106_kitchen1085_sensor4870_tempprobe_central-heating-flow.csv'
PATH_TO_CENTRAL_HEAT_RETURN= '/IDEAL_home106/home106_kitchen1085_sensor4869_tempprobe_central-heating-return.csv'

PATH_TO_MAIN_ELEC_COMBINED= '/IDEAL_home106/home106_hall1086_sensor4891c4895_electric-mains_electric-combined.csv'
PATH_TO_MAIN_ELEC_SUBCIRCUIT= '/IDEAL_home106/home106_utility1130_sensor5313_electric-subcircuit_mains.csv'
PATH_TO_ELEC_KETTLE= '/IDEAL_home106/home106_kitchen1085_sensor5210_electric-appliance_kettle.csv'
PATH_TO_ELEC_MICROWAVE= '/IDEAL_home106/home106_kitchen1085_sensor5211_electric-appliance_microwave.csv'
PATH_TO_ELEC_WASHING_MACHINE= '/IDEAL_home106/home106_kitchen1085_sensor5212_electric-appliance_washingmachine.csv'
PATH_TO_ELEC_FRIDGE= '/IDEAL_home106/home106_kitchen1085_sensor5213_electric-appliance_fridgefreezer.csv'
PATH_TO_ELEC_SHOWER= '/IDEAL_home106/home106_utility1130_sensor5312_electric-subcircuit_shower.csv'
PATH_TO_ELEC_COOKER= '/IDEAL_home106/home106_utility1130_sensor13191_electric-subcircuit_cooker.csv'

PATH_TO_OUTPUT_RAD_LIVING= '/IDEAL_home106/home106_livingroom1087_sensor5154_tempprobe_radiator-output.csv'
PATH_TO_INPUT_RAD_LIVING= '/IDEAL_home106/home106_livingroom1087_sensor5155_tempprobe_radiator-input.csv'
PATH_TO_OUTPUT_RAD_BEDROOM= '/IDEAL_home106/home106_bedroom1089_sensor5174_tempprobe_radiator-output.csv'
PATH_TO_INPUT_RAD_BEDROOM= '/IDEAL_home106/home106_bedroom1089_sensor5175_tempprobe_radiator-input.csv'
PATH_TO_OUTPUT_RAD_HALL= '/IDEAL_home106/home106_hall1086_sensor5162_tempprobe_radiator-output.csv'
PATH_TO_INPUT_RAD_HALL= '/IDEAL_home106/home106_hall1086_sensor5163_tempprobe_radiator-input.csv'



# FUNCTION
def create_min_df(PATH_TO_DF,value_name,agg_function):
    try:
        df = pd.read_csv(PATH_TO_DF, sep=",", header=None)
        df.columns = ['time_stamp', value_name]
    except:
        df = pd.read_csv(PATH_TO_DF, sep=";", header=None)
        df.columns = ['time_stamp', value_name]

    df[value_name]=df[value_name]/10
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df['year'] = df['time_stamp'].dt.year
    df['month'] = df['time_stamp'].dt.month
    df['day'] = df['time_stamp'].dt.day
    df['hour'] = df['time_stamp'].dt.hour
    df['min'] = df['time_stamp'].dt.minute
    df = df[['year', 'month', 'day', 'hour', 'min', value_name]].groupby(by=['year', 'month', 'day', 'hour', 'min'], as_index=False).agg({value_name:agg_function})
    return df

# CREATION DES DONNEES CHRONO POUR JOINTURE

# TEXT
df_Text=pd.read_csv(PATH_TO_WEATHER_FILE,sep=";")
df_Text=df_Text.groupby(by=['year','month','day','hour'],as_index=False).mean()

# GAZ
df_gas_pulse = pd.read_csv(PATH_TO_GAS_PULSE_FILE,sep=";",header=None)
df_gas_pulse.columns=['time_stamp','value']
df_gas_pulse['time_stamp']=pd.to_datetime(df_gas_pulse['time_stamp'])
df_gas_pulse['dateHour']=df_gas_pulse['time_stamp'].dt.floor('H')
df_gas_pulse['year']=df_gas_pulse['time_stamp'].dt.year
df_gas_pulse['month']=df_gas_pulse['time_stamp'].dt.month
df_gas_pulse['day']=df_gas_pulse['time_stamp'].dt.day
df_gas_pulse['hour']=df_gas_pulse['time_stamp'].dt.hour
df_gas_pulse['min']=df_gas_pulse['time_stamp'].dt.minute
df_gas_pulse_min=df_gas_pulse[['year','month','day','hour','min','value']].groupby(by=['year','month','day','hour','min'],as_index=False).sum()
df_gas_pulse_min.rename(columns={"value":"gas_value"},inplace=True)

#CREATION D'UNE PLAGE DE TEMPS
df_main=pd.DataFrame()
df_main['time_stamp']=pd.date_range(start =str(df_gas_pulse['dateHour'].min()),end =str(df_gas_pulse['dateHour'].max()), freq ='min')
df_main['time_stamp']=df_main['time_stamp'].dt.floor('min')
df_main['year']=df_main['time_stamp'].dt.year
df_main['month']=df_main['time_stamp'].dt.month
df_main['day']=df_main['time_stamp'].dt.day
df_main['hour']=df_main['time_stamp'].dt.hour
df_main['min']=df_main['time_stamp'].dt.minute

# ELEC
ELEC_RELATED_DATA=[(PATH_TO_MAIN_ELEC_COMBINED,'power_elec_combined','sum'),
                   (PATH_TO_MAIN_ELEC_SUBCIRCUIT,'power_elec_subcircuit','sum'),
                   (PATH_TO_ELEC_KETTLE,'power_elec_kettle','sum'),
                   (PATH_TO_ELEC_MICROWAVE,'power_elec_mircowave','sum'),
                   (PATH_TO_ELEC_WASHING_MACHINE,'power_elec_washMach','sum'),
                   (PATH_TO_ELEC_FRIDGE,'power_elec_fridge','sum'),
                   (PATH_TO_ELEC_SHOWER,'power_elec_shower','sum'),
                   (PATH_TO_ELEC_COOKER,'power_elec_cooker','sum')]

df_list=[df_main]+[create_min_df(data[0],data[1],data[2]) for data in ELEC_RELATED_DATA]
df_elec_min=reduce(lambda  left,right: pd.merge(left,right,on=['year','month','day','hour','min'],how='left'), df_list)


# CENTRAL HEAT
CENTRAL_HEAT_RELATED_DATA=[(PATH_TO_CENTRAL_HEAT_FLOW,'central_heat_flow_temp','mean'),
                           (PATH_TO_CENTRAL_HEAT_RETURN,'central_heat_return_temp','mean')
                           ]

df_list=[df_main]+[create_min_df(data[0],data[1],data[2]) for data in CENTRAL_HEAT_RELATED_DATA]
df_central_heat_min=reduce(lambda  left,right: pd.merge(left,right,on=['year','month','day','hour','min'],how='left'), df_list)

# RADIATOR TEMP



# TINT
TINT_RELATED_DATA=[(PATH_TO_LIVING_TINT_FILE,'Tint_living','mean'),
                   (PATH_TO_BEDROOM_TINT_FILE,'Tint_bedroom','mean'),
                   (PATH_TO_HALL_TINT_FILE,'Tint_hall','mean'),
                   (PATH_TO_KITCHEN_TINT_FILE,'Tint_kitchen','mean')
                   ]

df_list=[df_main]+[create_min_df(data[0],data[1],data[2]) for data in TINT_RELATED_DATA]
df_Tint_min=reduce(lambda  left,right: pd.merge(left,right,on=['year','month','day','hour','min'],how='left'), df_list)

# RADIATORS
T_RAD_RELATED_DATA=[(PATH_TO_OUTPUT_RAD_LIVING,'T_output_rad_living','mean'),
                   (PATH_TO_INPUT_RAD_LIVING,'T_intput_rad_living','mean'),
                   (PATH_TO_OUTPUT_RAD_BEDROOM,'T_output_rad_bedroom','mean'),
                   (PATH_TO_INPUT_RAD_BEDROOM,'T_intput_rad_bedroom','mean'),
                   (PATH_TO_OUTPUT_RAD_HALL, 'T_output_rad_hall', 'mean'),
                   (PATH_TO_INPUT_RAD_HALL, 'T_intput_rad_hall', 'mean')
                   ]

df_list=[df_main]+[create_min_df(data[0],data[1],data[2]) for data in T_RAD_RELATED_DATA]
df_T_rad_min=reduce(lambda  left,right: pd.merge(left,right,on=['year','month','day','hour','min'],how='left'), df_list)

# JOINTURE
df=df_main.merge(df_gas_pulse_min,on=['year','month','day','hour','min'],how='left')\
    .merge(df_Tint_min[['year','month','day','hour','min','Tint_living','Tint_bedroom','Tint_hall','Tint_kitchen']],on=['year','month','day','hour','min'],how='left')\
    .merge(df_Text[['year','month','day','hour','Text']],on=['year','month','day','hour'],how='left')\
    .merge(df_central_heat_min[['year','month','day','hour','min','central_heat_flow_temp','central_heat_return_temp']],on=['year','month','day','hour','min'],how='left')\
    .merge(df_elec_min[['year','month','day','hour','min',
                        'power_elec_combined',
                        'power_elec_subcircuit',
                        'power_elec_kettle',
                        'power_elec_mircowave',
                        'power_elec_washMach',
                        'power_elec_fridge',
                        'power_elec_shower',
                        'power_elec_cooker'
                        ]],on=['year','month','day','hour','min'],how='left') \
    .merge(df_T_rad_min[['year', 'month', 'day', 'hour', 'min',
                        'T_output_rad_living',
                        'T_intput_rad_living',
                        'T_output_rad_bedroom',
                        'T_intput_rad_bedroom',
                        'T_output_rad_hall',
                        'T_intput_rad_hall'
                        ]], on=['year', 'month', 'day', 'hour', 'min'], how='left')


# df.dropna(inplace=True)
df.to_csv('C:\\Users\\a.lance\\PycharmProjects\\UA3412_\\IDEAL_home106\\home106_main_data_set_Min.csv',sep=";",index=False, decimal=',')


