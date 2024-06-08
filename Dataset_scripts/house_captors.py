import pandas as pd
import os
from sqlalchemy import create_engine

path_to_sensor_directory="/home/antoine/Documents/MASTER/UA3412/data/home_appliance_sensor_hr/"
directory = os.fsencode(path_to_sensor_directory)
gasApplianceList=[]

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
    appliance = split[4]
    df = pd.read_csv(fullFileName, sep=";")
    df['home_id'] = home.replace("home", "")
    df['room_id'] = room.replace("room", "")
    df['sensor_id'] = sensor_num.replace("sensor", "")
    df['sensor_type'] = sensor_type
    df['appliance'] = appliance.replace(".csv", "")

    if appliance.replace(".csv", "")=='humidity':
        df['unit'] = 'humidity'
        df = df[['sensor_id', 'home_id', 'room_id', 'sensor_type', 'appliance', 'year', 'month', 'day', 'time_extract', 'unit','value_hr']]
        df.columns = ['sensor_id', 'home_id', 'room_id', 'sensor_type', 'appliance', 'year', 'month', 'day', 'hour', 'unit','value']
        df.to_sql('sensor', con=conn, if_exists='append', index=False)
        print(file)
    elif sensor_type in ('tempprobe') or appliance in ('temperature.csv'):
        df['unit'] = 'celcius'
        df['value_hr']=df['value_hr']/10
        df = df[['sensor_id', 'home_id', 'room_id', 'sensor_type', 'appliance', 'year', 'month', 'day', 'time_extract', 'unit','value_hr']]
        df.columns = ['sensor_id', 'home_id', 'room_id', 'sensor_type', 'appliance', 'year', 'month', 'day', 'hour', 'unit','value']
        df.to_sql('sensor', con=conn, if_exists='append', index=False)
        print(file)
    else:
        print('ok')
    print(file)
