import pandas as pd
import os
PATH_TO_INPUT_DATA_SET="/home/antoine/Documents/MASTER/UA3412/data/dataset/"
PATH_TO_WEATHER_DATA="/home/antoine/Documents/MASTER/UA3412/data/weather/weather_edinburgh_final.csv"
PATH_TO_OUTPUT_DIRECTORY="/home/antoine/Documents/MASTER/UA3412/data/dataset_consolidate/"
df_weather = pd.read_csv(PATH_TO_WEATHER_DATA,sep=";")

for file in os.listdir(PATH_TO_INPUT_DATA_SET):
    filename = os.fsdecode(file)
    fullFileName=PATH_TO_INPUT_DATA_SET+'/'+filename
    df_data=pd.read_csv(fullFileName,sep=",")
    final_df=pd.merge(df_data,df_weather,on=['year','month','day','hour'],how='left')
    outputFilename=PATH_TO_OUTPUT_DIRECTORY+"weather_"+filename
    final_df.iloc[:,1:].to_csv(outputFilename,sep=";",index=False)
    print(outputFilename)