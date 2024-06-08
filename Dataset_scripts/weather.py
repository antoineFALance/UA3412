import json
import requests as rq
import pandas as pd
import hashlib
import datetime
# import urllib.request as urllib2
import urllib.parse
import http.cookiejar as cookielib
import time

URL_AV = "https://api.weather.com/v1/location/EGPH:9:GB/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate={startDate}&endDate={endDate}"

dateRange=[d.strftime('%Y%m%d') for d in pd.date_range('20180101','20181231')]
timeStampList,tempList,pressureList,humidityList,wsList,wDirList,precipList=[],[],[],[],[],[],[]

for dt in dateRange:
    df = pd.DataFrame()
    url= URL_AV.format(startDate=dt,endDate=dt)
    download = rq.get(url=url,verify=False)
    jsonContent=download.json()
    for obs in jsonContent['observations']:
        dfTemp=pd.DataFrame([obs])
        df = pd.concat([df,dfTemp],axis=0)
    filename=dt+'.csv'
    df.to_csv(filename,sep=";")
    print(dt)
    time.sleep(5)

print('ok')
