import pandas as pd
import os
import sys
import numpy as np
is_windows = hasattr(sys, 'getwindowsversion')
cwd=os.path.dirname(os.getcwd())

if is_windows:
    PATH_TO_INPUT_DATA_SET=cwd+"\\data_\\old_dataset\\"
    PATH_TO_WEATHER_DATA=cwd+"\\data_\\weather_forecast\\weather_edinburgh_final.csv"
    PATH_TO_OUTPUT_DIRECTORY=cwd+"\\data_\\main_dataset\\"
    PATH_TO_OUTPUT_WEATHER_DIRECTORY = cwd + "\\data_\\weather_forecast\\"
else:
    PATH_TO_INPUT_DATA_SET=cwd+"/data_/old_dataset/"
    PATH_TO_WEATHER_DATA=cwd+"/data_/weather_forecast/weather_edinburgh_final.csv"
    PATH_TO_OUTPUT_DIRECTORY=cwd+"/data_/main_dataset/"
    PATH_TO_OUTPUT_WEATHER_DIRECTORY = cwd + "/data_/weather_forecast/"

df_weather = pd.read_csv(PATH_TO_WEATHER_DATA,sep=";")
df_weather['datetime']=pd.to_datetime(df_weather[['year', 'month', 'day','hour']])
df_weather['timestamp']=df_weather['datetime'].values.tolist()
# Vecteur vent
wind_speed = df_weather.pop('wspd')
wind_direction=df_weather.pop('wdir')*np.pi/180
df_weather['wind_x']=wind_speed*np.cos(wind_direction)
df_weather['wind_y']=wind_speed*np.sin(wind_direction)
# INFO FREQUENCE
day = 24
year =365.2425*day

df_weather['Day sin'] = np.sin(df_weather['timestamp'] * (2 * np.pi / day))
df_weather['Day cos'] = np.cos(df_weather['timestamp'] * (2 * np.pi / day))
df_weather['Year sin'] = np.sin(df_weather['timestamp'] * (2 * np.pi / year))
df_weather['Year cos'] = np.cos(df_weather['timestamp'] * (2 * np.pi / year))
final_features=['year','month','day','hour','Day sin','Day cos','Year sin','Year cos','temp','dewPt','rh','pressure','wind_x','wind_y']
df_weather=df_weather[final_features]
df_weather.to_csv(PATH_TO_OUTPUT_WEATHER_DIRECTORY+'transformed_weather_dataset.csv')

for file in os.listdir(PATH_TO_INPUT_DATA_SET):
    filename = os.fsdecode(file)
    fullFileName=PATH_TO_INPUT_DATA_SET+'/'+filename
    df_data=pd.read_csv(fullFileName,sep=",")
    final_df=pd.merge(df_data,df_weather,on=['year','month','day','hour'],how='left')
    final_df=final_df.rename(columns={'temp':'Text','temp_value':'Tint'})
    outputFilename=PATH_TO_OUTPUT_DIRECTORY+"weather_"+filename
    final_df.iloc[:,1:].to_csv(outputFilename,sep=";",index=False)
    print(outputFilename)