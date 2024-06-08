import pandas as pd
import os
from sqlalchemy import create_engine
from functools import reduce

path_to_sensor_config_directory="/home/antoine/Documents/MASTER/UA3412/home_list.csv"

df =pd.read_csv(path_to_sensor_config_directory,sep=",")
var=['home_id',
     'room_id',
     'temperature_sensor',
     'humidity_sensor',
     'central_heating_flow_sensor',
     'central_heating_return_sensor',
     'gas_sensor',
     'radiator_input_sensor',
     'radiator_output_sensor']
sensorConfigList=df[var].to_numpy().tolist()

conn_string = 'postgresql://postgres:admin@localhost/IDEAL'
db = create_engine(conn_string)
conn = db.connect()
query = "SELECT year,month,day,hour,value FROM sensor where sensor_id={sensor_id}"

for config in sensorConfigList:
     df_temperature = pd.read_sql(query.format(sensor_id="'"+str(config[2])+"'"),conn)

     df_humidity= pd.read_sql(query.format(sensor_id="'" + str(config[3]) + "'"), conn)
     df_central_heating_flow= pd.read_sql(query.format(sensor_id="'" + str(config[4]) + "'"), conn)
     df_central_heating_return = pd.read_sql(query.format(sensor_id="'" + str(config[5]) + "'"), conn)
     df_gas= pd.read_sql(query.format(sensor_id="'" + str(config[6]) + "'"), conn)
     df_radiator_input = pd.read_sql(query.format(sensor_id="'" + str(config[7]) + "'"), conn)
     df_radiator_output = pd.read_sql(query.format(sensor_id="'" + str(config[8]) + "'"), conn)

     dfList=[df_temperature,df_humidity,df_central_heating_flow,df_central_heating_return,df_gas,df_radiator_input,df_radiator_output]
     keysJoin=['year','month','day','hour']
     suffixes=['temp_','hum_','central_heat_flow_','central_heat_return_','gas_','radiator_input_','radiator_output_']
     for index in range(len(dfList)):
          dfList[index].rename(columns={'value': suffixes[index]+'value'},inplace=True)


     df_merged = reduce(lambda left, right: pd.merge(left,right, on=keysJoin,how='left'), dfList)
     fileName=config[0]+'.csv'
     df_merged.to_csv(fileName)
     print(fileName)