import pandas as pd
import os
from sqlalchemy import create_engine

path_to_sensor_directory="/home/antoine/Documents/MASTER/UA3412/data/auxiliarydata/hourly_readings/"
directory = os.fsencode(path_to_sensor_directory)

conn_string = 'postgresql://postgres:admin@localhost/IDEAL'
db = create_engine(conn_string)
conn = db.connect()

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    fullFileName=path_to_sensor_directory+'/'+filename
    split= filename.split("_")
    home = split[0]
    room = split[1]
    sensor_num = split[2]
    sensor_type = split[3]
    appliance = split[4].replace(".csv", "")
    if appliance in ['gas','temperature','humidity','radiator-input','radiator-output','central-heating-flow','central-heating-return']:

        df = pd.read_csv(fullFileName, sep=",",header=None)
        df.columns=['datetime','value']
        df['datetime']=pd.to_datetime(df['datetime'],format='%Y-%m-%d %H:%M:%S')
        df['year']= df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['home_id'] = home
        df['room_id'] = room
        df['sensor_id'] = sensor_num
        df['sensor_type'] = sensor_type
        df['appliance'] = appliance.replace(".csv", "")
        df=df[['sensor_id','home_id','room_id','sensor_type','appliance','year','month','day','hour','value']]
        if 'temp' in sensor_type or 'temp' in appliance:
            df['value'] = df['value'] / 10
        df.to_sql('sensor', con=conn, if_exists='append', index=False)
        print(file)
