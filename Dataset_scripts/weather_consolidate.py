import pandas as pd
import os
PATH_TO_INPUT_DIRECTORY=os.getcwd()+"\\weather\\"

df=pd.DataFrame()
for file in os.listdir(PATH_TO_INPUT_DIRECTORY):
    filename = os.fsdecode(file)
    # print(filename)
    df_temp = pd.read_csv(PATH_TO_INPUT_DIRECTORY + filename, sep=";")
    df_temp['temp']=(df_temp['temp']-32)*5/9
    df_temp['feels_like'] = (df_temp['feels_like'] - 32) * 5 / 9
    df_temp['dateTime']= pd.to_datetime(df_temp['valid_time_gmt'], unit='s')
    df_temp['year']=df_temp['dateTime'].dt.year
    df_temp['month'] = df_temp['dateTime'].dt.month
    df_temp['day'] = df_temp['dateTime'].dt.day
    df_temp['hour'] = df_temp['dateTime'].dt.hour
    var=['year','month','day','hour','obs_name','temp','dewPt','rh','pressure','wc','wdir','wspd']
    df_agg=df_temp[var].groupby(['year','month','day','hour','obs_name'],as_index=False).mean()
    df=pd.concat([df,df_agg],axis=0)
    print(filename)
df.to_csv('weather_edinburgh.csv',sep=";",index=False)
