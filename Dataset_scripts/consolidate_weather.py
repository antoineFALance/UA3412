import pandas as pd
import os
from sqlalchemy import create_engine
from functools import reduce

path_to_weather_directory="/home/antoine/Documents/MASTER/UA3412/data/weather/additional_weather_data/"
path_to_output_directory="/home/antoine/Documents/MASTER/UA3412/data/weather/"
path_to_initial_weather_file="/home/antoine/Documents/MASTER/UA3412/data/weather/weather_edinburgh.csv"
df_weather=pd.DataFrame()
for file in os.listdir(path_to_weather_directory):
    filename = os.fsdecode(file)
    fullFileName=path_to_weather_directory+'/'+filename
    df=pd.read_csv(fullFileName,sep=";")
    df_weather=pd.concat([df_weather,df])
    df_weather['temp_celcius']=(df_weather['temp']-32)*5/9
    print(filename)

df_weather['datetime']=pd.to_datetime(df_weather['valid_time_gmt'], unit='s')
df_weather['year']=df_weather['datetime'].dt.year
df_weather['month']=df_weather['datetime'].dt.month
df_weather['day']=df_weather['datetime'].dt.day
df_weather['hour']=df_weather['datetime'].dt.hour
df_weather=df_weather[['year','month','day','hour','obs_name','temp_celcius','dewPt','rh','pressure','wc','wdir','wspd']]
df_weather.rename(columns={'temp_celcius':'temp'}, inplace=True)
df_initial_weather_file=pd.read_csv(path_to_initial_weather_file,sep=";")
df_final=pd.concat([df_initial_weather_file,df_weather],axis=0)
df_final.drop_duplicates(inplace=True)
df_final.sort_values(by=['year','month','day','hour'],inplace=True)
df_final.to_csv(path_to_output_directory+'weather_edinburgh_final.csv',sep=";",index=False)
print('ok')