import pandas as pd
import os
import sys
is_windows = hasattr(sys, 'getwindowsversion')

# Données météo
if is_windows:
    PATH_TO_WEATHER_FORECAST=os.path.dirname(os.path.dirname(os.getcwd()))+"\\data_\\weather_forecast\\"
else:
    PATH_TO_WEATHER_FORECAST = os.path.dirname(os.path.dirname(os.getcwd())) + "/data_/weather_forecast/"

df_weather = pd.read_csv(PATH_TO_WEATHER_FORECAST+"#####.csv")
