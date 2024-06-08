import pandas as pd
import os
from sqlalchemy import create_engine

path_to_sensor_directory="/home/antoine/Documents/MASTER/UA3412/data/gas_pulse_hr/"
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
    df['unit'] = 'watthour'
    df = df[
        ['sensor_id', 'home_id', 'room_id', 'sensor_type', 'appliance', 'year', 'month', 'day', 'hour', 'unit',
         'value']]
    df.to_sql('sensor', con=conn, if_exists='append', index=False)
    print(file)